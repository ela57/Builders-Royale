"""State and elementary geometry for the deliberately approximate local simulator."""
from copy import deepcopy
from dataclasses import dataclass, field
import math


def distance(a, b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def angle(a):
    return (a+math.pi) % (2*math.pi)-math.pi


def direction(yaw, pitch):
    c = math.cos(pitch)
    return [math.cos(yaw)*c,math.sin(yaw)*c,math.sin(pitch)]


def cylinder_hit(a, b, pos, radius, height):
    """Earliest segment parameter intersecting a vertical cylinder, including caps."""
    dx,dy,dz = [b[i]-a[i] for i in range(3)]
    x,y = a[0]-pos[0],a[1]-pos[1]
    aa,bb,cc = dx*dx+dy*dy,2*(x*dx+y*dy),x*x+y*y-radius*radius
    if aa < 1e-12:
        if cc > 0:
            return None
        lo,hi = 0.,1.
    else:
        disc = bb*bb-4*aa*cc
        if disc < 0:
            return None
        lo,hi = max(0.,(-bb-math.sqrt(disc))/(2*aa)),min(1.,(-bb+math.sqrt(disc))/(2*aa))
    if abs(dz) < 1e-12:
        if not pos[2] <= a[2] <= pos[2]+height:
            return None
    else:
        z0,z1 = (pos[2]-a[2])/dz,(pos[2]+height-a[2])/dz
        lo,hi = max(lo,min(z0,z1)),min(hi,max(z0,z1))
    return lo if lo <= hi else None


def make_weapon(kind, config):
    result = {'kind':kind,'stats':deepcopy(config['weapons'][kind]),'cooldown':0.}
    if kind == 'gun':
        result.update(ammo=int(result['stats']['mag']),reload=0.)
    if kind == 'laser':
        result['charge'] = 0.
    return result


@dataclass
class Fighter:
    id: int
    pos: list
    config: dict
    vel: list = field(default_factory=lambda:[0.,0.,0.])
    aim: list = field(default_factory=lambda:[0.,0.])
    action: dict = field(default_factory=dict)
    cubes: int = 0
    kills: int = 0
    alive: bool = True
    grounded: bool = True
    hp: float = 1400.
    energy: float = 100.
    upgrades: list = field(default_factory=list)
    events: list = field(default_factory=list)
    sounds: list = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    last_combat: float = -1e9
    last_energy: float = -1e9
    last_step: float = -1e9
    last_by: object = None
    cause: str = 'timeout'
    death_time: float = 0.
    place: int = 0
    thresholds: int = 0
    offer: object = None
    short_until: float = 0.

    def __post_init__(self):
        self.stats = deepcopy(self.config['mech'])
        self.hp = self.stats['hp']
        self.energy = self.stats['energy']
        self.weapons = [make_weapon('gun',self.config),None]

    @property
    def max_hp(self):
        return self.stats['hp']*(1+self.config['cubes']['hp_per_cube']*min(self.cubes,self.config['cubes']['stat_cap']))

    @property
    def scale(self):
        return 1+self.config['cubes']['size_per_cube']*self.cubes

    def actor(self):
        return {'id':self.id,'kind':'mech','pos':self.pos[:],'vel':self.vel[:],
                'aim':self.aim[:],'hp':max(0,self.hp),'max_hp':self.max_hp,'cubes':self.cubes,
                'weapons':[w['kind'] for w in self.weapons if w],
                'charging':any(w and w['kind'] == 'laser' and w['charge'] > 0 for w in self.weapons)}

    def observation(self, zone):
        stats = self.stats.copy()
        for key in ('radius','height','eye_height','muzzle_height'):
            stats[key] *= self.scale
        return {'id':self.id,'pos':self.pos[:],'vel':self.vel[:],'aim':self.aim[:],
                'hp':max(0,self.hp),'max_hp':self.max_hp,'energy':self.energy,'max_energy':self.stats['energy'],
                'grounded':self.grounded,'cubes':self.cubes,'upgrades':self.upgrades[:],
                'weapons':deepcopy(self.weapons),'in_zone':distance(self.pos,zone['center']) <= zone['radius'],
                'stats':stats}


class Zone:
    def __init__(self, config, rng):
        self.phases = []
        center,radius,now = config['start_center'][:],config['start_radius'],0
        for phase in config['phases']:
            theta = rng.uniform(-math.pi,math.pi)
            offset = math.sqrt(rng.random())*max(0,radius-phase['radius'])
            nxt = [center[0]+math.cos(theta)*offset,center[1]+math.sin(theta)*offset]
            start,end = now+phase['hold_s'],now+phase['hold_s']+phase['shrink_s']
            self.phases.append({'center':center,'radius':radius,'next_center':nxt,'next_radius':phase['radius'],
                                'shrink_start':start,'shrink_end':end,'dps':phase['dps']})
            center,radius,now = nxt,phase['radius'],end

    def at(self, now):
        phase = next((p for p in self.phases if now < p['shrink_end']),self.phases[-1])
        fraction = min(1,max(0,(now-phase['shrink_start'])/max(.001,phase['shrink_end']-phase['shrink_start'])))
        result = deepcopy(phase)
        result['center'] = [phase['center'][i]*(1-fraction)+phase['next_center'][i]*fraction for i in range(2)]
        result['radius'] = phase['radius']*(1-fraction)+phase['next_radius']*fraction
        return result
