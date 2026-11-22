"""Bounded navigation, independent legs/torso and survival priorities."""
import gc
import math
from pathlib import Path
import random
import signal
import threading
import time

from brain import DESIRED, Memory, pick_offer, safe_point, take_weapon
from combat import best_weapon, dodge, fire_control, unit
from geo import BudgetExceeded, DecisionTimeout, Geo
import mechfight as mf
from navigation import Navigator


def decision_alarm(signum, frame):
    raise DecisionTimeout


class Bot:
    name = 'Builders Royale'

    def __init__(self, hello, geo=None):
        root = Path(__file__).resolve().parent
        map_dir = Path(hello.get('map', {}).get('dir', root / 'arena'))
        if not (map_dir / 'v2/world.json').is_file():
            map_dir = root / 'arena'
        self.geo = geo if geo is not None else Geo(map_dir,root / 'navgrid.npz')
        gc.collect()
        gc.freeze()
        self.config = hello.get('config',{})
        self.rng = random.Random(hello.get('seed',0))
        self.nav = Navigator(self.geo)
        self.memory = Memory()
        self.mode = 'hide'
        self.target_id = None
        self.strafe = 1
        self.switch_at = 0
        self.cover = None
        self.cover_at = -math.inf
        self.cover_key = None
        self.stuck_since = None
        self.last_move = [0,0]
        self.regenerating = False
        self.explored = set()
        self.explore_goal = None
        self.explore_at = 0
        self.nav_cell = None
        self.nav_label = None

    def _reachable(self, point):
        if self.nav_label is None:
            return True
        cell = self.geo.nearest(point,radius=3)
        return cell is not None and self.geo.components[cell[1],cell[0]] == self.nav_label

    def _zone_goal(self, point, circle):
        if self._reachable(point):
            return point
        for radius in (12,24,48,80):
            candidates = []
            for i in range(12):
                yaw = i*math.pi/6
                trial = [point[0]+math.cos(yaw)*radius,point[1]+math.sin(yaw)*radius]
                if math.dist(trial,circle['center'])+3 > circle['radius'] or not self._reachable(trial):
                    continue
                cell = self.geo.nearest(trial,radius=3)
                trial = self.geo.point(cell)
                if math.dist(trial,circle['center'])+3 <= circle['radius']:
                    candidates.append(trial)
            if candidates:
                return min(candidates,key=lambda p:math.dist(p,point))
        return point

    def _explore(self, pos, circle, now):
        self.explored.add((int(pos[0]//30),int(pos[1]//30)))
        if self.explore_goal is not None and now < self.explore_at and math.dist(pos[:2],self.explore_goal) > 8 and math.dist(self.explore_goal,circle['center']) < circle['radius']-5:
            return self.explore_goal
        candidates = []
        for _ in range(24):
            yaw = self.rng.uniform(-math.pi,math.pi)
            radius = self.rng.uniform(40,100)
            point = [pos[0]+math.cos(yaw)*radius,pos[1]+math.sin(yaw)*radius]
            if math.dist(point,circle['center']) > circle['radius']-8:
                continue
            cell = self.geo.nearest(point,radius=3)
            if cell is None:
                continue
            point = self.geo.point(cell)
            if math.dist(point,circle['center']) > circle['radius']-5 or not self._reachable(point):
                continue
            key = (int(point[0]//30),int(point[1]//30))
            candidates.append((key not in self.explored,self.rng.random(),point))
        self.explore_goal = max(candidates)[2] if candidates else circle['center']
        self.explore_at = now+12
        return self.explore_goal

    def _cover(self, me, threats, circle, now, deadline):
        key = tuple((t['id'],int(t['pos'][0]/8),int(t['pos'][1]/8)) for t in threats)
        invalid = key != self.cover_key or (self.cover is not None and math.dist(self.cover,circle['center'])+2.5 > circle['radius'])
        if (invalid or now >= self.cover_at) and time.perf_counter() < deadline-.004:
            budget = min(4,max(0,(deadline-time.perf_counter())*1000-3))
            self.cover = self.geo.find_cover(me['pos'],[t['pos'] for t in threats],circle,time_budget_ms=budget)
            self.cover_at = now+1.5
            self.cover_key = key
        return self.cover

    def _detour_safe(self, point, me, zone, now):
        if math.dist(point[:2],zone['center'])+3 > zone['radius']:
            return False
        travel = (math.dist(me['pos'][:2],point[:2])+max(0,math.dist(point[:2],zone['next_center'])-.6*zone['next_radius']))/max(1,me.get('stats',{}).get('walk_speed',11.7))
        return now+travel+10 < zone['shrink_start'] or math.dist(point[:2],zone['next_center'])+3 <= .6*zone['next_radius']

    def _loot_goal(self, me, zone, now, weapons_only=False, cubes_only=False):
        held = {w['kind'] for w in me.get('weapons',[]) if w}
        candidates = []
        items = [] if cubes_only else [(p,.65 if p['kind'] == 'laser' else .85) for p in self.memory.pickups.values() if p['kind'] in DESIRED-held]
        if not weapons_only:
            items += [(p,1.2 if not held & DESIRED else .8) for p in self.memory.cubes.values()]
            items += [(p,1.) for p in self.memory.harvests.values()]
        for item,weight in items:
            point = item['pos']
            distance = math.dist(me['pos'][:2],point[:2])
            if distance > (140 if weapons_only else 100) or not self._detour_safe(point,me,zone,now):
                continue
            if not self._reachable(point):
                continue
            if abs(point[2]-me['pos'][2]) > 4 and point[2]-float(self.geo.ground(*point[:2])) > 3:
                continue
            if any(math.dist(point[:2],enemy['pos'][:2]) < 90 for enemy in self.memory.enemies.values()):
                continue
            candidates.append((distance*weight,item['id'],point[:2]))
        return min(candidates)[2] if candidates else None

    def _avoid_npcs(self, pos, move, target):
        if not any(move):
            return move
        speed = math.hypot(*move)
        for npc in self.memory.npcs.values():
            if target is not None and target['id'] == npc['id']:
                continue
            dx,dy = pos[0]-npc['pos'][0],pos[1]-npc['pos'][1]
            distance = math.hypot(dx,dy)
            ahead = math.dist([pos[0]+move[0]*12,pos[1]+move[1]*12],npc['pos'][:2])
            if distance < 50 or ahead < 50:
                away = unit(dx,dy)
                tangent = [-away[1],away[0]]
                if tangent[0]*move[0]+tangent[1]*move[1] < 0:
                    tangent = [-v for v in tangent]
                move = [v*speed for v in unit(away[0]+tangent[0],away[1]+tangent[1])]
        return move

    def act(self, obs):
        previous = getattr(self.geo,'deadline',math.inf)
        self.geo.deadline = min(previous,time.perf_counter()+.013)
        handler = None
        try:
            try:
                if hasattr(signal,'setitimer') and threading.current_thread() is threading.main_thread() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.):
                    handler = signal.getsignal(signal.SIGALRM)
                    if handler is not None:
                        signal.signal(signal.SIGALRM,decision_alarm)
                        signal.setitimer(signal.ITIMER_REAL,.014)
                if hasattr(self.geo,'collect_retired'):
                    self.geo.collect_retired(min(self.geo.deadline,time.perf_counter()+.0005))
                return self._act(obs)
            finally:
                if handler is not None:
                    try:
                        signal.setitimer(signal.ITIMER_REAL,0)
                    finally:
                        signal.signal(signal.SIGALRM,handler)
        except DecisionTimeout:
            self.nav.invalidate()
            return self._coast(obs)
        except BudgetExceeded:
            return self._coast(obs)
        finally:
            self.geo.deadline = previous

    def _coast(self, obs):
        return {'tick':obs.get('tick',0),'move':self.last_move[:],
                'aim':obs['self'].get('aim',[0,0])[:],'fire':[False,False]}

    def _act(self, obs):
        deadline = self.geo.deadline
        me,now,zone = obs['self'],obs['t'],obs['zone']
        pos = me['pos']
        if hasattr(self.geo,'components'):
            self.nav_cell = self.geo.nearest(pos)
            self.nav_label = self.geo.components[self.nav_cell[1],self.nav_cell[0]] if self.nav_cell is not None else None
        self.memory.update(obs)
        action = {'move':[0,0],'fire':[False,False],'boost':False}
        pick = pick_offer(obs.get('offer'))
        if pick is not None:
            action['pick'] = pick
        action.update(take_weapon(me,list(self.memory.pickups.values())))
        safe,urgent,circle,_ = safe_point(me,zone,now)
        threats = list(self.memory.enemies.values())
        target = self.memory.primary(me,now)
        visible_mechs = [a for a in obs.get('mechs',[]) if a.get('kind') == 'mech']
        visible_ids = {a['id'] for a in obs.get('mechs',[])}
        hp_ratio = me.get('hp',1400)/max(1,me.get('max_hp',1400))
        near = [t for t in threats if now-t['seen'] < 3 and math.dist(pos[:2],t['pos'][:2]) <= 120]
        stronger = bool(near) and me.get('hp',1400) > sum(e.get('hp',1400) for e in near)*.85 and me.get('cubes',0) >= max(e.get('cubes',0) for e in near)+2
        engage = hp_ratio > .4 and len(near) <= (2 if stronger else 1)
        if hp_ratio < .35 and threats:
            self.regenerating = True
        if hp_ratio >= .8:
            self.regenerating = False
        goal,move,speed = None,None,1
        equipment = self._loot_goal(me,zone,now,weapons_only=True) if now < 150 and not any(w and w['kind'] in DESIRED for w in me.get('weapons',[])) else None
        cubes = self._loot_goal(me,zone,now,cubes_only=True) if hp_ratio > .6 and not self.regenerating and all(math.dist(pos[:2],enemy['pos'][:2]) > 100 for enemy in threats) else None
        if urgent:
            self.mode,goal = 'zone',self._zone_goal(safe,circle)
            action['boost'] = me.get('energy',0) > 20
        elif cubes is not None:
            self.mode,goal,target = 'loot',cubes,None
        elif equipment is not None and hp_ratio > .7 and all(math.dist(pos[:2],enemy['pos'][:2]) > 100 for enemy in threats):
            self.mode,goal = 'equip',equipment
        elif threats and (not engage or self.regenerating):
            self.mode = 'disengage'
            goal = self._cover(me,threats,circle,now,deadline)
            if goal is None:
                away = unit(pos[0]-target['pos'][0],pos[1]-target['pos'][1])
                goal = [pos[0]+away[0]*30,pos[1]+away[1]*30]
                if math.dist(goal,circle['center']) > circle['radius']-3:
                    goal = safe
            action['boost'] = me.get('energy',0) > 20
        elif threats:
            self.mode = 'engage'
            if now >= self.switch_at:
                self.strafe = self.rng.choice((-1,1))
                self.switch_at = now+self.rng.uniform(.6,1.4)
            dx,dy = target['pos'][0]-pos[0],target['pos'][1]-pos[1]
            distance = math.hypot(dx,dy)
            weapon = best_weapon(me,distance)
            low,high = (25,38) if weapon and weapon['kind'] == 'lightning' else (50,110)
            radial = 1 if distance > high else (-1 if distance < low else 0)
            toward = unit(dx,dy)
            move = unit(toward[0]*radial-toward[1]*self.strafe,toward[1]*radial+toward[0]*self.strafe)
        elif self.regenerating and (hp_ratio < .8 or now-self.memory.last_damage < me.get('stats',{}).get('regen_delay',5)):
            self.mode = 'regen'
            goal = self._cover(me,[],circle,now,deadline)
            speed = .4
        else:
            npc = self.memory.npcs.get(self.memory.attacker)
            if npc and now-self.memory.last_damage < 8:
                target = npc
                self.mode = 'npc'
                distance = math.dist(pos[:2],npc['pos'][:2])
                move = unit(npc['pos'][0]-pos[0],npc['pos'][1]-pos[1]) if distance > 70 else [0,0]
            elif now-self.memory.last_mech >= 3:
                loot = self._loot_goal(me,zone,now)
                if loot is not None:
                    self.mode,goal = 'loot',loot
                else:
                    muzzle = [pos[0],pos[1],pos[2]+me.get('stats',{}).get('muzzle_height',7)]
                    harvest = [(c,'crate') for c in obs.get('crates',[])]
                    harvest += [(n,'npc') for n in obs.get('mechs',[]) if n.get('kind') == 'npc' and hp_ratio > .6]
                    harvest.sort(key=lambda pair:math.dist(pos[:2],pair[0]['pos'][:2]))
                    for crate,kind in harvest:
                        if time.perf_counter() >= deadline-.012:
                            break
                        if math.dist(pos[:2],crate['pos'][:2]) < 60 and self._reachable(crate['pos']) and abs(crate['pos'][2]-float(self.geo.ground(*crate['pos'][:2]))) < 4 and self._detour_safe(crate['pos'],me,zone,now) and self.geo.los(muzzle,[crate['pos'][0],crate['pos'][1],crate['pos'][2]+1.2]):
                            target = {**crate,'kind':kind}
                            self.mode,goal = kind,crate['pos'][:2]
                            if math.dist(pos[:2],goal) < 12:
                                move = [0,0]
                            visible_ids.add(crate['id'])
                            break
                    if target is None:
                        remembered = [c for c in self.memory.crates.values() if math.dist(pos[:2],c['pos'][:2]) < 100 and self._reachable(c['pos']) and abs(c['pos'][2]-float(self.geo.ground(*c['pos'][:2]))) < 4 and self._detour_safe(c['pos'],me,zone,now)]
                        if remembered:
                            self.mode,goal = 'loot',min(remembered,key=lambda c:math.dist(pos[:2],c['pos'][:2]))['pos'][:2]
                    if target is None and now-self.memory.last_mech >= 10:
                        drops = [d for d in obs.get('drops',[]) if math.dist(pos[:2],d['pos'][:2]) < 80 and self._detour_safe(d['pos'],me,zone,now)]
                        if drops:
                            drop = min(drops,key=lambda d:math.dist(pos[:2],d['pos'][:2]))
                            self.mode,goal = 'drop',drop['pos'][:2]
                            if math.dist(pos[:2],goal) < max(1,drop.get('radius',8)-3):
                                move = [0,0]
                if goal is None and move is None and target is None:
                    if now < 150 and not any(w and w['kind'] != 'gun' for w in me.get('weapons',[])):
                        self.mode,goal = 'explore',self._explore(pos,circle,now)
                    else:
                        self.mode = 'hide'
                        goal = self._cover(me,[],circle,now,deadline) or safe
                        speed = .4
            else:
                self.mode = 'hide'
                move = [0,0]
        if goal is not None and move is None and time.perf_counter() < deadline-.007:
            move = self.nav.move(pos,goal,now,budget_ms=min(8,(deadline-time.perf_counter())*1000-6),speed=speed)
            if self.nav_cell is not None and not self.geo.is_walkable(self.geo.cell(pos)):
                point = self.geo.point(self.nav_cell)
                move = unit(point[0]-pos[0],point[1]-pos[1])
        move = move or [0,0]
        evasion = dodge(me,visible_mechs,obs.get('projectiles',[]),self.geo,self.config,deadline-.004)
        if evasion is not None:
            move = evasion
            action['boost'] = me.get('energy',0) > 0
            self.mode = 'dodge'
        move = self._avoid_npcs(pos,move,target)
        actors = obs.get('mechs',[])+[{**c,'kind':'crate'} for c in obs.get('crates',[])]
        if time.perf_counter() < deadline-.003:
            aim,fire,reload = fire_control(me,target,target is not None and target['id'] in visible_ids,actors,self.geo,self.config,deadline-.001)
            if aim is not None:
                action['aim'] = aim
            action['fire'],action['reload'] = fire,reload
        if self.memory.damage_bearing is not None and now-self.memory.last_damage < .4 and self.memory.attacker not in visible_ids:
            action['aim'] = [self.memory.damage_bearing,0]
            action['fire'] = [False,False]
        elif 'aim' not in action:
            sounds = self.memory.sounds
            yaw = sounds[-1][0]['bearing'] if sounds else (int(now/1.5) % 6)*math.pi/3
            action['aim'] = [yaw,0]
        attempted = self.nav.search is None and (any(self.last_move) or (goal is not None and math.dist(pos[:2],goal) > 3))
        slow = math.hypot(*me.get('vel',[0,0,0])[:2]) < 1
        if attempted and slow:
            if self.stuck_since is None:
                self.stuck_since = now
            elif now-self.stuck_since >= 1:
                action['jump'],action['boost'] = True,me.get('energy',0) > 0
                if not any(move):
                    yaw = self.rng.uniform(-math.pi,math.pi)
                    move = [math.cos(yaw),math.sin(yaw)]
                self.nav.invalidate()
                self.stuck_since = now
        else:
            self.stuck_since = None
        action['move'] = move
        self.last_move = move
        self.target_id = target['id'] if target is not None else None
        return action


if __name__ == '__main__':
    mf.run(Bot)
