"""Stateful priorities and safe-circle planning, independent of protocol I/O."""
import math

from combat import unit


OFFER_PRIORITY = ('plating','armor','nanites','laser_damage','laser_charge',
                  'lightning_efficiency','gun_damage','gun_rate','servos')
DESIRED = {'laser','lightning'}


def pick_offer(offer):
    if not offer or not offer.get('choices'):
        return None
    return min(range(len(offer['choices'])),key=lambda i: OFFER_PRIORITY.index(offer['choices'][i]['id']) if offer['choices'][i]['id'] in OFFER_PRIORITY else len(OFFER_PRIORITY))


def take_weapon(me, pickups):
    weapons = me.get('weapons',[None,None])
    held = {w['kind'] for w in weapons if w}
    candidates = [p for p in pickups if p['kind'] in DESIRED-held and math.dist(me['pos'][:2],p['pos'][:2]) <= 4 and abs(me['pos'][2]-p['pos'][2]) <= 4]
    candidates.sort(key=lambda p:math.dist(me['pos'],p['pos']))
    for pickup in candidates:
        if len(weapons) < 2 or any(w is None for w in weapons):
            return {'take':pickup['id']}
        if weapons[0]['kind'] == 'gun' and not any(w and w['kind'] == 'laser' and w.get('charge',0) > 0 for w in weapons):
            return {'take':pickup['id'],'drop':0}
    return {}


def safe_point(me, zone, now):
    pos = me['pos'][:2]
    next_center = zone['next_center']
    distance = math.dist(pos,next_center)
    travel = max(0,distance-.6*zone['next_radius'])/max(1,me.get('stats',{}).get('walk_speed',11.7))
    use_next = now >= zone['shrink_start']-travel-10
    center = next_center if use_next else zone['center']
    radius = zone['next_radius'] if use_next else zone['radius']
    direction = unit(pos[0]-center[0],pos[1]-center[1])
    offset = min(math.dist(pos,center),.6*radius)
    point = [center[0]+direction[0]*offset,center[1]+direction[1]*offset]
    outside = not me.get('in_zone',True) or math.dist(pos,zone['center']) > zone['radius']
    urgent = outside or (use_next and math.dist(pos,point) > 2)
    return point,urgent,{'center':center,'radius':radius},travel


class Memory:
    def __init__(self):
        self.enemies = {}
        self.npcs = {}
        self.last_mech = -math.inf
        self.last_damage = -math.inf
        self.damage_bearing = None
        self.attacker = None
        self.sounds = []
        self.weapons = [None,None]

    def update(self, obs):
        now = obs['t']
        self.weapons = list(obs['self'].get('weapons',[None,None]))
        for actor in obs.get('mechs',[]):
            record = {**actor,'seen':now}
            if actor.get('kind') == 'mech':
                self.enemies[actor['id']] = record
                self.last_mech = now
            else:
                self.npcs[actor['id']] = record
        self.enemies = {i:e for i,e in self.enemies.items() if now-e['seen'] < 5}
        self.npcs = {i:e for i,e in self.npcs.items() if now-e['seen'] < 30}
        for event in obs.get('events',[]):
            if event['type'] == 'damaged':
                self.last_damage,self.attacker = now,event.get('by')
                self.damage_bearing = event.get('bearing')
            elif event['type'] == 'took':
                slot = int(event['slot'])
                if 0 <= slot < len(self.weapons):
                    self.weapons[slot] = {'kind':event['kind']}
            elif event['type'] == 'dropped':
                slot = int(event['slot'])
                if 0 <= slot < len(self.weapons):
                    self.weapons[slot] = None
        self.sounds = [(s,t) for s,t in self.sounds if now-t < 3]
        self.sounds.extend((s,now-s.get('age',0)) for s in obs.get('sounds',[]) if s['kind'] in ('gun','laser','laser_charge','step','lightning'))
        self.sounds = self.sounds[-32:]

    def primary(self, me, now):
        pos = me['pos']
        def score(enemy):
            damage = -40 if enemy['id'] == self.attacker and now-self.last_damage < 3 else 0
            return math.dist(pos[:2],enemy['pos'][:2])+damage+.025*enemy.get('hp',1400)
        return min(self.enemies.values(),key=score,default=None)
