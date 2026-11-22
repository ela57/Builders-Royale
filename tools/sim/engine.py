"""20 Hz local match model. Approximation boundaries are listed in README.md."""
from copy import deepcopy
import math
import random
import time

from tools.sim.agents import BotAdapter, HouseBot
from tools.sim.entities import WorldEntities
from tools.sim.model import Fighter, Zone, angle, cylinder_hit, direction, distance, make_weapon
from tools.sim.telemetry import Telemetry
from tools.sim.timing import calls, collections, reset_calls


class Simulator:
    def __init__(self, hello, geo, seed=1, mode='practice', bot='ours', ffa_mix='ours', duration=None,
                 supply_drops=False, npcs=False):
        self.hello,self.config,self.geo = deepcopy(hello),deepcopy(hello['config']),geo
        self.seed,self.mode,self.bot_kind = seed,mode,bot
        self.rng = random.Random(seed)
        self.dt = 1/self.config['match']['tick_hz']
        self.tick,self.time = 0,0.
        self.duration = duration if duration is not None else self.config['match']['duration_s']
        self.zone = Zone(self.config['zone'],self.rng)
        self.projectiles,self.cubes,self.pickups,self.crates,self.beams = [],[],[],[],[]
        self.next_id = 30000
        self.timings = []
        self.slow_decisions = []
        self.telemetry = Telemetry()
        self.last_known = None
        self.last_known_at = -1e9
        self.dead = []
        self._open = list(zip(*geo.walkable.nonzero()))
        self.fighters = [Fighter(i,pos,self.config) for i,pos in enumerate(self.spawns())]
        self.hunter_order = [f.id for f in self.fighters[1:]]
        self.rng.shuffle(self.hunter_order)
        for i in range(int(self.config['crates']['count'])):
            self.crates.append({'id':1000+i,'pos':self.open_point(),'hp':self.config['crates']['hp']})
        for i in range(int(self.config['loot']['floor_count'])):
            self.pickups.append({'id':8000+i,'pos':self.open_point(),'kind':self.random_weapon()})
        self.entities = WorldEntities(self,supply_drops,npcs)
        self.agents = {}
        self.agents[0] = BotAdapter(bot,self.make_hello(0),geo)
        if mode == 'practice':
            tiers = ['easy']*4+['medium']*4+['hard']*4+['expert']*3
            self.rng.shuffle(tiers)
            for fighter,tier in zip(self.fighters[1:],tiers):
                self.agents[fighter.id] = HouseBot(fighter,tier,geo,random.Random(seed*1009+fighter.id))
        else:
            for fighter in self.fighters[1:]:
                kind = 'example' if ffa_mix == 'example' or (ffa_mix == 'mixed' and fighter.id % 2) else 'ours'
                self.agents[fighter.id] = BotAdapter(kind,self.make_hello(fighter.id),geo)

    def make_hello(self, slot):
        return {**deepcopy(self.hello),'slot':slot,'seed':self.seed*1009+slot,
                'map':{'dir':str(self.geo.map_dir),'files':['v2/world.json','v2/terrain.i16']}}

    def new_id(self):
        self.next_id += 1
        return self.next_id

    def ground(self, x, y):
        t = self.geo.terrain_info
        fx = min(t['nx']-1,max(0,(x-t['minX'])/t['step']))
        fy = min(t['ny']-1,max(0,(y-t['minY'])/t['step']))
        i,j = min(t['nx']-2,int(fx)),min(t['ny']-2,int(fy))
        u,v = fx-i,fy-j
        a = self.geo.terrain
        return float((a[j,i]*(1-u)+a[j,i+1]*u)*(1-v)+(a[j+1,i]*(1-u)+a[j+1,i+1]*u)*v)

    def open_point(self):
        j,i = self._open[self.rng.randrange(len(self._open))]
        x,y = self.geo.point((int(i),int(j)))
        return [x,y,self.ground(x,y)]

    def spawns(self):
        cfg = self.config['match']
        center = self.config['zone']['start_center']
        radius = self.config['zone']['start_radius']*cfg['spawn_zone_fraction']
        bounds = self.geo.world['bounds']
        margin = cfg['arena_margin']
        for _ in range(100):
            points = []
            for _ in range(30000):
                point = self.open_point()
                if distance(point,center) > radius:
                    continue
                if not (bounds['minX']+margin <= point[0] <= bounds['maxX']-margin and bounds['minY']+margin <= point[1] <= bounds['maxY']-margin):
                    continue
                if all(distance(point,p) >= cfg['spawn_min_separation'] for p in points):
                    points.append(point)
                    if len(points) == cfg['mechs']:
                        return points
        raise RuntimeError('Could not pack legal spawns; separation was not relaxed')

    def random_weapon(self):
        weights = self.config['loot']['weights']
        return self.rng.choices(list(weights),weights=list(weights.values()))[0]

    def visible(self, viewer, point, height=0):
        cfg = self.config['vision']
        dist = distance(viewer.pos,point)
        near = cfg['near_range']+viewer.stats.get('near_range_bonus',0)
        reach = cfg['range']+viewer.stats.get('vision_bonus',0)
        bearing = math.atan2(point[1]-viewer.pos[1],point[0]-viewer.pos[0])
        if dist > reach or (dist > near and abs(angle(bearing-viewer.aim[0])) > math.radians(cfg['fov_deg']/2)):
            return False
        eye = [viewer.pos[0],viewer.pos[1],viewer.pos[2]+viewer.stats['eye_height']*viewer.scale]
        return self.geo.los(eye,[point[0],point[1],point[2]+height])

    def observe(self, viewer):
        zone = self.zone.at(self.time)
        actors = []
        for other in self.fighters:
            if other.id == viewer.id or not other.alive:
                continue
            if self.visible(viewer,other.pos,other.stats['height']*other.scale/2):
                viewer.memory[other.id] = self.time
            if self.time-viewer.memory.get(other.id,-1e9) <= self.config['vision'].get('memory_s',5):
                actors.append(other.actor())
        def items(collection, height=0):
            return [{k:deepcopy(v) for k,v in item.items() if k in ('id','pos','kind','hp')} for item in collection if self.visible(viewer,item['pos'],height)]
        actors += self.entities.npc_observations(viewer)
        sounds = [{**s,'age':self.time-s['age']} for s in viewer.sounds]
        obs = {'type':'obs','tick':self.tick,'t':self.time,'alive':sum(f.alive for f in self.fighters),
               'self':viewer.observation(zone),'mechs':actors,
               'projectiles':[{'id':p['id'],'kind':p.get('kind','gun'),'pos':p['pos'][:],'vel':p['vel'][:]} for p in self.projectiles if p['owner'] != viewer.id and self.visible(viewer,p['pos'])],
               'crates':items(self.crates,self.config['crates']['height']/2),
               'cubes':items(self.cubes),'pickups':items(self.pickups),
               'sounds':sounds,'beams':[{**deepcopy(b),'age':self.time-b['age']} for b in self.beams if b['by'] != viewer.id and any(self.visible(viewer,p) for p in b['path'])],
               'events':deepcopy(viewer.events),'zone':zone,'offer':deepcopy(viewer.offer),
               'drops':self.entities.drop_observations()}
        viewer.events.clear()
        viewer.sounds.clear()
        return obs

    def emit_sound(self, source, kind):
        cfg = self.config['sound']
        for other in self.fighters:
            if not other.alive or other.id == source.id:
                continue
            dist = distance(source.pos,other.pos)
            if dist > cfg['range'].get(kind,0):
                continue
            bearing = math.atan2(source.pos[1]-other.pos[1],source.pos[0]-other.pos[0])
            bearing += self.rng.gauss(0,math.radians(cfg['bearing_noise_deg'])*other.stats.get('hearing',1))
            other.sounds.append({'kind':kind,'bearing':angle(bearing),'dist':'near' if dist < cfg['near'] else ('mid' if dist < cfg['mid'] else 'far'),'age':self.time})
            other.sounds = other.sounds[-64:]

    def decide(self):
        shared = None
        if self.mode == 'practice' and self.fighters[0].alive:
            player = self.fighters[0]
            if any(f.alive and self.visible(f,player.pos,player.stats['height']*player.scale/2) for f in self.fighters[1:]):
                shared = player.actor()
                self.last_known,self.last_known_at = player.pos[:2],self.time
            if self.time-self.last_known_at > 5:
                self.last_known = None
        houses = [f for f in self.fighters[1:] if f.alive]
        count = len(houses) if self.time > 300 or len(houses)+int(self.fighters[0].alive) <= 6 else min(len(houses),1+self.fighters[0].kills)
        alive_ids = {f.id for f in houses}
        hunters = set([slot for slot in self.hunter_order if slot in alive_ids][:count])
        actions = []
        for fighter in self.fighters:
            if not fighter.alive:
                continue
            agent = self.agents[fighter.id]
            if isinstance(agent,HouseBot):
                action = agent.act(self,fighter,shared,fighter.id in hunters)
                fighter.events.clear()
                fighter.sounds.clear()
            else:
                obs = self.observe(fighter)
                gc_before = collections()
                reset_calls()
                cpu_start = time.process_time()
                start = time.perf_counter()
                action = agent.act(obs)
                elapsed = (time.perf_counter()-start)*1000
                if fighter.id == 0:
                    self.timings.append(elapsed)
                    if elapsed >= 15:
                        gc_after = collections()
                        self.slow_decisions.append({'tick':self.tick,'wall_ms':elapsed,
                            'cpu_ms':(time.process_time()-cpu_start)*1000,
                            'gc_wall_ms':(gc_after[0]-gc_before[0])*1000,
                            'gc_cpu_ms':(gc_after[1]-gc_before[1])*1000,
                            'gc_generations_ms':[(gc_after[i]-gc_before[i])*1000 for i in range(2,5)],
                            'calls_ms':calls(),
                            'pos':fighter.pos[:],'mode':getattr(agent.bot,'mode',None)})
            actions.append((fighter,action or {}))
        def take_order(pair):
            fighter,action = pair
            item = next((p for p in self.pickups if p['id'] == action.get('take')),None)
            return (distance(fighter.pos,item['pos']) if item else math.inf,fighter.id)
        for fighter,action in sorted(actions,key=take_order):
            fighter.action = action
            self.one_shots(fighter)
        self.beams.clear()

    def one_shots(self, me):
        action = me.action
        if action.get('jump') and me.grounded:
            me.vel[2],me.grounded = me.stats['jump_speed'],False
        if action.get('reload'):
            for weapon in me.weapons:
                if weapon and weapon['kind'] == 'gun' and not weapon['reload'] and weapon['ammo'] < weapon['stats']['mag']:
                    weapon['reload'] = weapon['stats']['reload']
                    self.emit_sound(me,'reload')
        if me.offer and ('pick' in action or self.time >= me.offer['expires']):
            index = int(action.get('pick',0))
            if self.time >= me.offer['expires'] and not 0 <= index < len(me.offer['choices']):
                index = 0
            if 0 <= index < len(me.offer['choices']):
                chosen = me.offer['choices'][index]['id']
                self.upgrade(me,chosen)
                me.offer = None
        taking = next((p for p in self.pickups if p['id'] == action.get('take')),None)
        dropping = action.get('drop')
        if taking:
            reason = None
            if distance(me.pos,taking['pos']) > self.config['loot']['pickup_radius'] or abs(me.pos[2]-taking['pos'][2]) > 4:
                reason = 'range'
            elif taking['kind'] in {w['kind'] for w in me.weapons if w}:
                reason = 'have'
            slot = int(dropping) if dropping in (0,1) else (me.weapons.index(None) if None in me.weapons else -1)
            if reason is None and slot < 0:
                reason = 'full'
            if reason is None and me.weapons[slot] and me.weapons[slot]['kind'] == 'laser' and me.weapons[slot]['charge'] > 0:
                reason = 'busy'
            if reason:
                me.events.append({'type':'take_failed','id':taking['id'],'reason':reason})
            else:
                self.drop_weapon(me,slot)
                w = deepcopy(taking.get('weapon')) if 'weapon' in taking else make_weapon(taking['kind'],self.config)
                me.weapons[slot] = w
                self.weapon_upgrades(me,w)
                w['cooldown'] = max(w['cooldown'],self.config['loot']['equip_s'],w['stats']['cooldown'])
                me.events.append({'type':'took','id':taking['id'],'kind':taking['kind'],'slot':slot})
                self.pickups.remove(taking)
        elif 'take' in action:
            me.events.append({'type':'take_failed','id':action['take'],'reason':'gone'})
        elif dropping in (0,1):
            self.drop_weapon(me,int(dropping))

    def drop_weapon(self, me, slot):
        weapon = me.weapons[slot]
        if not weapon or (weapon['kind'] == 'laser' and weapon['charge'] > 0):
            return
        item = {'id':self.new_id(),'kind':weapon['kind'],'pos':me.pos[:],'weapon':deepcopy(weapon)}
        self.pickups.append(item)
        me.events.append({'type':'dropped','id':item['id'],'kind':item['kind'],'slot':slot})
        me.weapons[slot] = None

    def weapon_upgrades(self, me, weapon):
        weapon['stats'] = deepcopy(self.config['weapons'][weapon['kind']])
        for upgrade in me.upgrades:
            definition = next(u for u in self.config['upgrades'] if u['id'] == upgrade)
            for effect in definition['effects']:
                parts = effect['path'].split('.')
                if len(parts) == 3 and parts[:2] == ['weapons',weapon['kind']]:
                    old = weapon['stats'].get(parts[2],0)
                    weapon['stats'][parts[2]] = old*effect.get('mul',1)+effect.get('add',0)

    def upgrade(self, me, name):
        definition = next(u for u in self.config['upgrades'] if u['id'] == name)
        old_hp = me.max_hp
        me.upgrades.append(name)
        for effect in definition['effects']:
            parts = effect['path'].split('.')
            if parts[0] == 'mech':
                me.stats[parts[1]] = me.stats.get(parts[1],0)*effect.get('mul',1)+effect.get('add',0)
        for weapon in me.weapons:
            if weapon:
                self.weapon_upgrades(me,weapon)
        me.hp += me.max_hp-old_hp
        me.events.append({'type':'upgrade','id':name})

    def offer_upgrade(self, me):
        thresholds = self.config['cubes']['upgrade_thresholds']
        if me.offer or me.thresholds >= len(thresholds) or me.cubes < thresholds[me.thresholds]:
            return
        held = {w['kind'] for w in me.weapons if w}
        possible = [u for u in self.config['upgrades'] if me.upgrades.count(u['id']) < u['max'] and (not u.get('requires') or u['requires'] in held)]
        if possible:
            choices = self.rng.sample(possible,min(3,len(possible)))
            me.offer = {'id':self.new_id(),'choices':[{k:u[k] for k in ('id','name','desc')} for u in choices],
                        'expires':self.time+self.config['bot']['offer_timeout_s']}
        me.thresholds += 1

    def can_stand(self, x, y, z, step):
        i,j = self.geo.cell((x,y))
        if not (0 <= i < self.geo.width and 0 <= j < self.geo.height):
            return False
        if self.geo.raw_walkable[j,i]:
            return True
        top = float(self.geo.solid_top[j,i])
        return z+step >= top if math.isfinite(top) else z > self.ground(x,y)+step

    def move(self, me):
        dt,stats = self.dt,me.stats
        action = me.action
        move = action.get('move',[0,0])
        magnitude = math.hypot(*move)
        move = [v/max(1,magnitude) for v in move]
        boost = bool(action.get('boost')) and me.energy > 0
        speed = stats['boost_speed'] if boost and me.grounded else stats['walk_speed']
        if any(w and w['kind'] == 'laser' and w['charge'] > 0 for w in me.weapons):
            speed *= stats['charge_move_mult']
        accel = stats['ground_accel'] if me.grounded else stats['air_accel']
        dx,dy = move[0]*speed-me.vel[0],move[1]*speed-me.vel[1]
        change = math.hypot(dx,dy)
        fraction = min(1,accel*dt/max(change,1e-9))
        me.vel[0] += dx*fraction
        me.vel[1] += dy*fraction
        if boost and (not me.grounded or magnitude > 0):
            cost = stats['boost_drain'] if me.grounded else stats['jet_drain']
            me.energy = max(0,me.energy-cost*dt)
            me.last_energy = self.time
            if not me.grounded:
                me.vel[2] = min(stats['max_rise_speed'],me.vel[2]+stats['jet_accel']*dt)
        if self.time-me.last_energy >= stats['energy_regen_delay']:
            me.energy = min(stats['energy'],me.energy+stats['energy_regen']*dt)
        if not me.grounded:
            me.vel[2] -= stats['gravity']*dt
        z = me.pos[2]+me.vel[2]*dt
        for axis in (0,1):
            trial = me.pos[:]
            trial[axis] += me.vel[axis]*dt
            ground = self.ground(*trial[:2])
            if self.can_stand(trial[0],trial[1],max(z,me.pos[2]),stats['step_up']) and (not me.grounded or ground-me.pos[2] <= stats['step_up']):
                me.pos[axis] = trial[axis]
            else:
                me.vel[axis] = 0
        ground = self.ground(*me.pos[:2])
        i,j = self.geo.cell(me.pos)
        if 0 <= i < self.geo.width and 0 <= j < self.geo.height:
            top = float(self.geo.solid_top[j,i])
            if math.isfinite(top) and z >= top-stats['step_up']:
                ground = max(ground,top)
        if z <= ground or (me.grounded and z-ground < stats['step_up']):
            me.pos[2],me.vel[2],me.grounded = ground,0.,True
        else:
            me.pos[2],me.grounded = z,False
        desired = action.get('aim',me.aim)
        yaw_change = max(-stats['turn_rate']*dt,min(stats['turn_rate']*dt,angle(desired[0]-me.aim[0])))
        me.aim[0] = angle(me.aim[0]+yaw_change)
        pitch = min(stats['max_pitch'],max(stats['min_pitch'],desired[1]))
        me.aim[1] += max(-stats['pitch_rate']*dt,min(stats['pitch_rate']*dt,pitch-me.aim[1]))
        if math.hypot(*me.vel[:2]) >= self.config['sound']['step_speed'] and self.time-me.last_step >= self.config['sound']['step_every']:
            self.emit_sound(me,'boost' if boost else 'step')
            me.last_step = self.time

    def enemies(self, owner):
        squad = self.mode == 'practice' and owner is not None and 0 < owner < len(self.fighters)
        return [f for f in self.fighters if f.alive and f.id != owner and not (squad and f.id != 0)]

    def muzzle(self, me, yaw=None, pitch=None):
        yaw,pitch = me.aim if yaw is None else (yaw,pitch)
        origin = [me.pos[0],me.pos[1],me.pos[2]+me.stats['muzzle_height']*me.scale]
        ray = direction(yaw,pitch)
        offset = me.stats['radius']*me.scale+.5
        out = [origin[i]+ray[i]*offset for i in range(3)]
        return (out if self.geo.los(origin,out) else origin),ray

    def ray_target(self, owner, a, b, radius=0):
        hits = []
        for target in self.enemies(owner):
            hit = cylinder_hit(a,b,target.pos,target.stats['radius']*target.scale+radius,target.stats['height']*target.scale)
            if hit is not None:
                hits.append((hit,target.id,target))
        for crate in self.crates:
            hit = cylinder_hit(a,b,crate['pos'],self.config['crates']['radius']+radius,self.config['crates']['height'])
            if hit is not None:
                hits.append((hit,crate['id'],crate))
        if owner is not None and 0 <= owner < len(self.fighters):
            cfg = self.config['npc']
            for npc in self.entities.npcs:
                if npc['hp'] <= 0:
                    continue
                hit = cylinder_hit(a,b,npc['pos'],cfg['radius']+radius,cfg['height'])
                if hit is not None:
                    hits.append((hit,npc['id'],npc))
        if not hits:
            return None,None
        fraction,_,target = min(hits,key=lambda h:(h[0],h[1]))
        hitpoint = [a[i]+(b[i]-a[i])*fraction for i in range(3)]
        if self.geo.los(a,hitpoint):
            return target,hitpoint
        return None,None

    def damage(self, owner, target, amount, kind, source):
        attacker = self.fighters[owner] if owner is not None and 0 <= owner < len(self.fighters) else None
        if isinstance(target,dict):
            if target['hp'] <= 0:
                return
            target['hp'] -= amount
            npc = target.get('kind') == 'npc'
            if npc and attacker:
                target['target'],target['last_by'],target['returning'] = owner,owner,False
            if owner == 0:
                (self.telemetry.npc_hits if npc else self.telemetry.crate_hits)[kind] += 1
            if attacker:
                attacker.last_combat = self.time
                attacker.events.append({'type':'hit','target':target['id'],'amount':amount,'weapon':kind})
            return
        if not target.alive or target.hp <= 0:
            return
        if target.id == 0 and attacker is not None:
            self.telemetry.contact(self.time,target)
            self.telemetry.aggressors[owner] = self.time
        if owner == 0:
            self.telemetry.hits[kind] += 1
            self.telemetry.contact(self.time,self.fighters[0])
        rear,bearing = False,None
        if source is not None:
            bearing = math.atan2(source[1]-target.pos[1],source[0]-target.pos[0])
            rear = abs(angle(bearing-target.aim[0])) > math.radians(target.stats['front_arc_deg']/2)
            amount *= (target.stats['rear_mult'] if rear else 1)*(1-min(.95,target.stats.get('armor',0)))
        target.hp -= amount
        target.last_combat,target.last_by = self.time,owner
        target.cause = 'zone' if kind == 'zone' else ('npc' if kind == 'npc' else 'mech')
        target.events.append({'type':'damaged','amount':amount,'weapon':kind,'bearing':bearing,'by':owner,'rear':rear})
        if attacker:
            attacker.last_combat = self.time
            attacker.events.append({'type':'hit','target':target.id,'amount':amount,'weapon':kind})

    def weapons(self, me):
        firing = me.action.get('fire',[False,False])
        for slot,w in enumerate(me.weapons):
            if not w:
                continue
            stats,kind = w['stats'],w['kind']
            overdue = min(0,w['cooldown']-self.dt) if w['cooldown'] > 0 else 0
            w['cooldown'] = max(0,w['cooldown']-self.dt)
            held = bool(firing[slot]) if slot < len(firing) else False
            if kind == 'gun':
                if w['reload'] > 0:
                    w['reload'] = max(0,w['reload']-self.dt)
                    if w['reload'] <= 1e-8:
                        w['reload'] = 0.
                        w['ammo'] = int(stats['mag'])
                if held and w['cooldown'] <= 1e-8 and w['ammo'] > 0 and w['reload'] <= 1e-8:
                    spread = math.radians(stats['spread_deg'])
                    a,ray = self.muzzle(me,me.aim[0]+self.rng.gauss(0,spread),me.aim[1]+self.rng.gauss(0,spread))
                    damage = stats['damage']*self.damage_scale(me)
                    self.projectiles.append({'id':self.new_id(),'owner':me.id,'pos':a,'vel':[v*stats['speed'] for v in ray],
                                             'left':stats['range'],'damage':damage,'source':me.pos[:]})
                    w['ammo'] -= 1
                    self.telemetry.shot(self,me,'gun')
                    w['cooldown'] = stats['cooldown']+overdue
                    if w['ammo'] == 0:
                        w['reload'] = stats['reload']
                        self.emit_sound(me,'reload')
                    self.emit_sound(me,'gun')
            elif kind == 'laser' and w['cooldown'] <= 1e-8 and self.time >= me.short_until:
                if held or (w['charge'] > 0 and w['charge'] < stats['min_charge']):
                    if w['charge'] == 0:
                        self.emit_sound(me,'laser_charge')
                    w['charge'] += self.dt
                release = w['charge'] >= stats['min_charge'] and not held
                if release or w['charge'] >= stats['charge']+stats['hold_s']:
                    self.telemetry.shot(self,me,'laser')
                    a,ray = self.muzzle(me)
                    b = [a[i]+ray[i]*stats['range'] for i in range(3)]
                    target,point = self.ray_target(me.id,a,b)
                    if target is not None:
                        fraction = min(1,w['charge']/max(.001,stats['charge']))
                        damage = stats['damage']*(stats['min_frac']+(1-stats['min_frac'])*fraction)*self.damage_scale(me)
                        if isinstance(target,Fighter) and any(w and w['kind'] == 'mortar' for w in target.weapons):
                            damage *= 1+stats['cookoff']
                        self.damage(me.id,target,damage,'laser',me.pos)
                    self.beams.append({'by':me.id,'kind':'laser','path':[a,point or b],'age':self.time})
                    self.emit_sound(me,'laser')
                    w['charge'],w['cooldown'] = 0.,stats['cooldown']+overdue
            elif kind == 'lightning' and held and w['cooldown'] <= 1e-8 and me.energy >= stats['energy']:
                self.telemetry.shot(self,me,'lightning')
                me.energy -= stats['energy']
                me.last_energy = self.time
                w['cooldown'] = stats['cooldown']+overdue
                a,_ = self.muzzle(me)
                candidates = []
                targets = self.enemies(me.id)+self.crates+self.entities.npcs
                for target in targets:
                    p = target['pos'] if isinstance(target,dict) else target.pos
                    height = (self.config['npc']['height']/2 if target.get('kind') == 'npc' else 1.2) if isinstance(target,dict) else target.stats['height']*target.scale/2
                    point = [p[0],p[1],p[2]+height]
                    if math.dist(a,point) <= stats['range'] and abs(angle(math.atan2(p[1]-a[1],p[0]-a[0])-me.aim[0])) <= math.radians(stats['cone_deg']) and self.geo.los(a,point):
                        candidates.append((math.dist(a,point),target,point))
                if candidates:
                    _,target,point = min(candidates,key=lambda it:it[0])
                    self.damage(me.id,target,stats['damage']*self.damage_scale(me),'lightning',me.pos)
                    if isinstance(target,Fighter):
                        target.short_until = self.time+stats['short_s']
                        for tw in target.weapons:
                            if tw and tw['kind'] == 'laser':
                                tw['charge'] = 0.
                    self.beams.append({'by':me.id,'kind':'lightning','path':[a,point],'age':self.time})
                self.emit_sound(me,'lightning')

    def damage_scale(self, me):
        cfg = self.config['cubes']
        return 1+cfg['damage_per_cube']*min(me.cubes,cfg['stat_cap'])

    def advance_projectiles(self):
        remaining = []
        for bullet in self.projectiles:
            a = bullet['pos']
            length = min(bullet['left'],math.sqrt(sum(v*v for v in bullet['vel']))*self.dt)
            fraction = length/max(1e-9,math.sqrt(sum(v*v for v in bullet['vel'])))
            b = [a[i]+bullet['vel'][i]*fraction for i in range(3)]
            target,point = self.ray_target(bullet['owner'],a,b,self.config['projectile']['gun_radius'])
            if target is not None:
                source = [target.pos[i]-bullet['vel'][i] for i in range(3)] if isinstance(target,Fighter) else bullet['source']
                self.damage(bullet['owner'],target,bullet['damage'],bullet.get('kind','gun'),source)
            elif self.geo.los(a,b) and bullet['left'] > length:
                bullet['pos'],bullet['left'] = b,bullet['left']-length
                remaining.append(bullet)
        self.projectiles = remaining

    def add_cubes(self, pos, count):
        for _ in range(int(count)):
            self.cubes.append({'id':self.new_id(),'pos':pos[:]})

    def collect(self, me):
        cfg = self.config['cubes']
        for cube in self.cubes[:]:
            if distance(me.pos,cube['pos']) <= cfg['pickup_radius'] and abs(me.pos[2]-cube['pos'][2]) <= 4:
                eligible = [f for f in self.fighters if f.alive and f.hp > 0 and distance(f.pos,cube['pos']) <= cfg['pickup_radius'] and abs(f.pos[2]-cube['pos'][2]) <= 4]
                if min(eligible,key=lambda f:(distance(f.pos,cube['pos']),f.id)).id != me.id:
                    continue
                old = me.max_hp
                me.cubes += 1
                me.hp = min(me.max_hp,me.hp+me.max_hp-old)
                me.events.append({'type':'pickup','cubes':me.cubes})
                self.cubes.remove(cube)
        self.offer_upgrade(me)

    def resolve_deaths(self):
        dead = [f for f in self.fighters if f.alive and f.hp <= 0]
        place = sum(f.alive for f in self.fighters)-len(dead)+1
        for fighter in dead:
            fighter.alive,fighter.place,fighter.death_time = False,place,self.time+self.dt
            self.dead.append(fighter.id)
            if fighter.last_by is not None and 0 <= fighter.last_by < len(self.fighters):
                killer = self.fighters[fighter.last_by]
                killer.kills += 1
                if killer.id == 0 and self.telemetry.active:
                    self.telemetry.active['kills'] += 1
                killer.events.append({'type':'kill','target':fighter.id})
            self.add_cubes(fighter.pos,fighter.cubes+self.config['cubes']['death_drop_extra'])
            if self.config['loot']['death_drop']:
                for slot,w in enumerate(fighter.weapons):
                    if w and w['kind'] != 'gun':
                        w['charge'] = 0
                        self.drop_weapon(fighter,slot)
        for crate in self.crates[:]:
            if crate['hp'] <= 0:
                self.add_cubes(crate['pos'],self.config['crates']['cubes'])
                if self.rng.random() < self.config['loot']['crate_weapon_chance']:
                    self.pickups.append({'id':self.new_id(),'pos':crate['pos'][:],'kind':self.random_weapon()})
                self.crates.remove(crate)

    def step(self):
        self.entities.announce()
        if self.tick % self.config['match']['decide_every'] == 0:
            self.decide()
        if self.fighters[0].alive:
            self.telemetry.tick(self)
        zone = self.zone.at(self.time)
        for fighter in self.fighters:
            if fighter.alive:
                self.move(fighter)
        self.entities.move_npcs()
        for fighter in self.fighters:
            if fighter.alive:
                self.weapons(fighter)
        self.advance_projectiles()
        for fighter in self.fighters:
            if not fighter.alive:
                continue
            inside = distance(fighter.pos,zone['center']) <= zone['radius']
            if not inside:
                self.damage(None,fighter,zone['dps']*self.dt,'zone',None)
            elif self.time-fighter.last_combat >= fighter.stats['regen_delay'] and fighter.hp > 0:
                fighter.hp = min(fighter.max_hp,fighter.hp+fighter.stats['regen_per_s']*self.dt)
            if fighter.hp > 0:
                self.collect(fighter)
        self.resolve_deaths()
        self.entities.resolve_npcs()
        self.entities.capture()
        self.tick += 1
        self.time = self.tick*self.dt

    def finish(self):
        alive = sorted((f for f in self.fighters if f.alive),key=lambda f:(-f.hp,f.id))
        for rank,fighter in enumerate(alive,1):
            fighter.place,fighter.death_time,fighter.cause = rank,self.time,'timeout'
        player = self.fighters[0]
        return {'seed':self.seed,'place':player.place,'score':16-player.place,'cause':player.cause,
                'kills':player.kills,'cubes':player.cubes,'time_alive':round(player.death_time,2),
                'timings':self.timings,'telemetry':self.telemetry.result(self),
                'world':{'drop_captures':dict(self.entities.drop_captures),
                         'npc_kills':dict(self.entities.npc_kills)},
                'slow_decisions':self.slow_decisions}

    def run(self):
        while self.time < self.duration and sum(f.alive for f in self.fighters) > 1 and self.fighters[0].alive:
            self.step()
        return self.finish()
