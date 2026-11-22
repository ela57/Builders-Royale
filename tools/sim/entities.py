"""Optional protocol supply drops and simple leashed NPC drones."""
import math
import random
import weakref
from collections import Counter
from types import SimpleNamespace

from tools.sim.model import direction, distance


class WorldEntities:
    def __init__(self, sim, drops=False, npcs=False):
        self.sim = weakref.proxy(sim)
        self.rng = random.Random(sim.seed+849013)
        self.called = set()
        self.drops = []
        self.npcs = []
        self.drop_captures = Counter()
        self.npc_kills = Counter()
        self.drops_enabled = drops
        if npcs:
            for i in range(int(sim.config['npc']['count'])):
                pos = self.open_point()
                self.npcs.append({'id':2000+i,'kind':'npc','pos':pos,'home':pos[:],
                    'vel':[0.,0.,0.],'aim':[0.,0.],'hp':sim.config['npc']['hp'],
                    'max_hp':sim.config['npc']['hp'],'target':None,'cooldown':0.,
                    'last_by':None,
                    'returning':False,'wander':pos[:]})

    def open_point(self):
        j,i = self.sim._open[self.rng.randrange(len(self.sim._open))]
        x,y = self.sim.geo.point((int(i),int(j)))
        return [x,y,self.sim.ground(x,y)]

    def announce(self):
        sim = self.sim
        if not self.drops_enabled:
            return
        cfg = sim.config['drops']
        for called_at in cfg['times']:
            if sim.time < called_at or called_at in self.called:
                continue
            zone = sim.zone.at(sim.time)
            for _ in range(10000):
                point = self.open_point()
                if distance(point,zone['next_center']) <= zone['next_radius']*cfg['spread']:
                    break
            else:
                raise RuntimeError('No legal supply-drop landing site in the next circle')
            self.called.add(called_at)
            self.drops.append({'id':sim.new_id(),'pos':point,'radius':cfg['radius'],
                'lands_at':called_at+cfg['fall_s'],'capture_s':cfg['capture_s'],
                'progress':0.,'holder':None,'landed':False})

    def drop_observations(self):
        return [{k:v for k,v in drop.items() if k not in ('lands_at','landed')} |
                {'pos':drop['pos'][:],'lands_in':max(0,drop['lands_at']-self.sim.time)}
                for drop in self.drops]

    def capture(self):
        sim = self.sim
        for drop in self.drops[:]:
            if sim.time < drop['lands_at']:
                continue
            if not drop['landed']:
                sim.emit_sound(SimpleNamespace(id=None,pos=drop['pos']),'drop')
                drop['landed'] = True
            occupants = [f for f in sim.fighters if f.alive and f.hp > 0 and
                         distance(f.pos,drop['pos']) <= drop['radius'] and
                         abs(f.pos[2]-drop['pos'][2]) <= 4]
            if not occupants:
                drop['holder'],drop['progress'] = None,0.
            elif len(occupants) == 1:
                holder = occupants[0]
                if drop['holder'] != holder.id:
                    drop['holder'],drop['progress'] = holder.id,0.
                drop['progress'] += sim.dt
                if drop['progress']+1e-9 >= drop['capture_s']:
                    holder.cubes += int(sim.config['drops']['cubes'])
                    holder.hp,holder.energy = holder.max_hp,holder.stats['energy']
                    holder.events.append({'type':'drop','id':drop['id'],'cubes':holder.cubes})
                    self.drop_captures[holder.id] += 1
                    sim.offer_upgrade(holder)
                    self.drops.remove(drop)

    def npc_observations(self, viewer):
        sim = self.sim
        result = []
        for npc in self.npcs:
            if sim.visible(viewer,npc['pos'],sim.config['npc']['height']/2):
                viewer.memory[npc['id']] = sim.time
            if sim.time-viewer.memory.get(npc['id'],-1e9) <= sim.config['vision']['memory_s']:
                result.append({key:npc[key][:] if isinstance(npc[key],list) else npc[key]
                    for key in ('id','kind','pos','vel','aim','hp','max_hp')})
        return result

    def move_npcs(self):
        sim = self.sim
        cfg = sim.config['npc']
        for npc in self.npcs:
            if npc['hp'] <= 0:
                continue
            npc['cooldown'] = max(0,npc['cooldown']-sim.dt)
            target = next((f for f in sim.fighters if f.id == npc['target'] and f.alive and f.hp > 0),None)
            home_distance = distance(npc['pos'],npc['home'])
            if target is not None and (distance(npc['pos'],target.pos) > cfg['deaggro_range'] or home_distance >= cfg['chase_leash']):
                target,npc['target'],npc['returning'] = None,None,True
            if npc['returning'] and home_distance < 1:
                npc['returning'] = False
            if target is None and not npc['returning'] and home_distance <= cfg['leash']:
                nearby = [f for f in sim.fighters if f.alive and f.hp > 0 and
                    distance(npc['pos'],f.pos) <= cfg['aggro_range'] and
                    sim.geo.los([*npc['pos'][:2],npc['pos'][2]+cfg['height']*.8],
                                [*f.pos[:2],f.pos[2]+f.stats['height']*f.scale/2])]
                target = min(nearby,key=lambda f:(distance(npc['pos'],f.pos),f.id),default=None)
                npc['target'] = target.id if target else None
            if target is not None:
                goal = target.pos
            elif npc['returning']:
                goal = npc['home']
            else:
                if distance(npc['pos'],npc['wander']) < 1:
                    yaw = self.rng.uniform(-math.pi,math.pi)
                    radius = math.sqrt(self.rng.random())*cfg['leash']
                    npc['wander'] = [npc['home'][0]+math.cos(yaw)*radius,
                                     npc['home'][1]+math.sin(yaw)*radius]
                goal = npc['wander']
            old = npc['pos'][:]
            length = distance(npc['pos'],goal)
            step = min(length,cfg['speed']*sim.dt)
            if length > 1:
                x = old[0]+(goal[0]-old[0])*step/length
                y = old[1]+(goal[1]-old[1])*step/length
                if sim.can_stand(x,old[1],old[2],.5):
                    npc['pos'][0] = x
                if sim.can_stand(npc['pos'][0],y,old[2],.5):
                    npc['pos'][1] = y
                npc['pos'][2] = sim.ground(*npc['pos'][:2])
            npc['vel'] = [(npc['pos'][i]-old[i])/sim.dt for i in range(3)]
            if target is None:
                continue
            origin = [*npc['pos'][:2],npc['pos'][2]+cfg['height']*.8]
            point = [*target.pos[:2],target.pos[2]+target.stats['height']*target.scale/2]
            dx,dy,dz = [point[i]-origin[i] for i in range(3)]
            npc['aim'] = [math.atan2(dy,dx),math.atan2(dz,math.hypot(dx,dy))]
            if npc['cooldown'] > 1e-8 or not sim.geo.los(origin,point):
                continue
            spread = math.radians(cfg['spread_deg'])
            ray = direction(npc['aim'][0]+self.rng.gauss(0,spread),
                            npc['aim'][1]+self.rng.gauss(0,spread))
            sim.projectiles.append({'id':sim.new_id(),'kind':'npc','owner':npc['id'],
                'pos':origin,'vel':[v*cfg['projectile_speed'] for v in ray],
                'left':cfg['deaggro_range'],'damage':cfg['damage'],'source':npc['pos'][:]})
            npc['cooldown'] = cfg['cooldown']
            sim.emit_sound(SimpleNamespace(id=npc['id'],pos=npc['pos']),'npc_gun')

    def resolve_npcs(self):
        for npc in self.npcs[:]:
            if npc['hp'] <= 0:
                self.sim.add_cubes(npc['pos'],self.sim.config['npc']['cubes'])
                if npc['last_by'] is not None:
                    self.sim.fighters[npc['last_by']].events.append({'type':'kill','target':npc['id']})
                    self.npc_kills[npc['last_by']] += 1
                self.npcs.remove(npc)
