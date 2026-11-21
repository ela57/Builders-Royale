"""In-process bot adapters and configurable practice-squad opponents."""
import ast
from collections import deque
import math
from pathlib import Path

from bot import Bot
from combat import aim_at, intercept, unit
import mechfight as mf
from navigation import Navigator
from tools.sim.model import angle, distance


def example_class():
    """Keep the kit class unchanged; omit only its unconditional stdin entrypoint."""
    path = Path(__file__).with_name('example_bot.py')
    tree = ast.parse(path.read_text(),filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name)
        and node.value.func.value.id == 'mf' and node.value.func.attr == 'run')]
    namespace = {'__name__':'sim_example','__file__':str(path)}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['Bot']


class BotAdapter:
    def __init__(self, kind, hello, geo):
        if kind == 'ours':
            self.bot = Bot(hello,geo=geo)
        else:
            self.bot = example_class()(hello)

    def act(self, obs):
        mf.draw.clear()
        return self.bot.act(obs)


# yaw/pitch noise (radians), reaction seconds, laser and lightning willingness.
TIERS = {
    'easy':(.30,1.2,.2,.35),
    'medium':(.16,.8,.55,.65),
    'hard':(.09,.5,.85,.9),
    'expert':(.04,.25,1.,1.),
}


class HouseBot:
    def __init__(self, fighter, tier, geo, rng):
        self.id,self.tier,self.rng = fighter.id,tier,rng
        self.nav = Navigator(geo)
        self.history = deque()
        self.next_noise = 0
        self.noise = [0,0]
        self.stuck_at = 0
        self.last_pos = fighter.pos[:]
        self.last_goal = None

    def act(self, sim, me, shared, hunting):
        now,zone = sim.time,sim.zone.at(sim.time)
        sigma,delay,laser_use,lightning_use = TIERS[self.tier]
        if shared is not None:
            self.history.append((now,shared))
        while len(self.history) > 1 and self.history[1][0] <= now-delay:
            self.history.popleft()
        target = self.history[0][1] if self.history and self.history[0][0] <= now-delay and now-self.history[0][0] < 5 else None
        if now >= self.next_noise:
            self.noise = [self.rng.gauss(0,sigma),self.rng.gauss(0,sigma*.4)]
            self.next_noise = now+.3
        action = {'fire':[False,False],'move':[0,0],'boost':False}
        urgent = distance(me.pos,zone['next_center']) > .8*zone['next_radius']
        travel = max(0,distance(me.pos,zone['next_center'])-.7*zone['next_radius'])/me.stats['walk_speed']
        rotating = distance(me.pos,zone['center']) > .87*zone['radius'] or (urgent and now > zone['shrink_start']-travel-12)
        goal = None
        if rotating:
            toward = unit(me.pos[0]-zone['next_center'][0],me.pos[1]-zone['next_center'][1])
            goal = [zone['next_center'][0]+toward[0]*zone['next_radius']*.5,zone['next_center'][1]+toward[1]*zone['next_radius']*.5]
            action['boost'] = me.energy > 15
        elif hunting:
            goal = (target['pos'] if target else sim.last_known or zone['next_center'])[:2]
            if target and distance(me.pos,target['pos']) < 100:
                toward = unit(target['pos'][0]-me.pos[0],target['pos'][1]-me.pos[1])
                desired = 30 if any(w and w['kind'] == 'lightning' for w in me.weapons) else 65
                radial = 1 if distance(me.pos,target['pos']) > desired+10 else (-1 if distance(me.pos,target['pos']) < desired-10 else 0)
                side = 1 if (int(now/2)+self.id) % 2 else -1
                action['move'] = unit(toward[0]*radial-toward[1]*side*.7,toward[1]*radial+toward[0]*side*.7)
                goal = None
        else:
            nearby = [p for p in sim.pickups if p['kind'] != 'mortar' and distance(p['pos'],me.pos) < 90 and distance(p['pos'],zone['center']) < zone['radius']*.9 and p['kind'] not in {w['kind'] for w in me.weapons if w}]
            if nearby:
                goal = min(nearby,key=lambda p:distance(p['pos'],me.pos))['pos'][:2]
            elif self.last_goal is None or distance(me.pos,self.last_goal) < 8 or int(now*10) % 150 == self.id:
                theta = self.rng.uniform(-math.pi,math.pi)
                radius = min(35,zone['radius']*.2)*self.rng.random()
                point = [me.pos[0]+math.cos(theta)*radius,me.pos[1]+math.sin(theta)*radius]
                if distance(point,zone['center']) > zone['radius']*.9:
                    point = zone['next_center'][:]
                cell = self.nav.geo.nearest(point,radius=3)
                self.last_goal = list(self.nav.geo.point(cell)) if cell is not None else point
                goal = self.last_goal
            else:
                goal = self.last_goal
        if goal is not None:
            if sim.geo.walk_clear(me.pos,goal):
                action['move'] = unit(goal[0]-me.pos[0],goal[1]-me.pos[1]) if distance(me.pos,goal) > 3 else [0,0]
            else:
                action['move'] = self.nav.move(me.pos,goal,now,budget_ms=.25)
        action['aim'] = [math.atan2(action['move'][1],action['move'][0]) if any(action['move']) else angle(now*.5+self.id),0]
        if target and distance(me.pos,target['pos']) < 280:
            muzzle = [me.pos[0],me.pos[1],me.pos[2]+me.stats['muzzle_height']*me.scale]
            point = [*target['pos'][:2],target['pos'][2]+4.5*(1+.05*target['cubes'])]
            selected = next((w for w in me.weapons if w and w['kind'] == 'laser'),None)
            if selected is None:
                gun = next((w for w in me.weapons if w and w['kind'] == 'gun'),None)
                if gun:
                    point = intercept(muzzle,point,target['vel'],gun['stats']['speed']) or point
            aim = aim_at(muzzle,point)
            action['aim'] = [aim[0]+self.noise[0],aim[1]+self.noise[1]]
            if sim.geo.los(muzzle,point):
                error = abs(angle(me.aim[0]-aim[0]))
                for slot,w in enumerate(me.weapons):
                    if not w or error > .18:
                        continue
                    kind = w['kind']
                    if kind == 'gun':
                        action['fire'][slot] = True
                    elif kind == 'laser' and laser_use >= self.rng.random():
                        action['fire'][slot] = w['charge'] < w['stats']['charge'] or error > .04
                    elif kind == 'lightning' and lightning_use >= self.rng.random():
                        action['fire'][slot] = distance(me.pos,target['pos']) < w['stats']['range']*.95
        for pickup in sim.pickups:
            if distance(me.pos,pickup['pos']) <= sim.config['loot']['pickup_radius'] and pickup['kind'] != 'mortar' and pickup['kind'] not in {w['kind'] for w in me.weapons if w}:
                if None in me.weapons:
                    action['take'] = pickup['id']
                elif me.weapons[0]['kind'] == 'gun':
                    action.update(take=pickup['id'],drop=0)
                break
        if any(w and w['kind'] == 'gun' and w['ammo'] == 0 for w in me.weapons):
            action['reload'] = True
        if me.offer:
            action['pick'] = 0
        if distance(me.pos,self.last_pos) > 1:
            self.stuck_at,self.last_pos = now,me.pos[:]
        elif now-self.stuck_at > 1 and any(action['move']):
            action.update(jump=True,boost=me.energy > 0)
            side = 1 if self.id % 2 else -1
            mx,my = action['move']
            action['move'] = unit(mx-my*side,my+mx*side)
        return action
