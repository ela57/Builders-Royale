"""Evaluator-only combat accounting; never exposed to bot observations."""
from collections import Counter
import math


class Telemetry:
    def __init__(self):
        self.shots = Counter()
        self.hits = Counter()
        self.crate_hits = Counter()
        self.first_held = {'laser':None,'lightning':None}
        self.modes = Counter()
        self.fights = []
        self.active = None
        self.aggressors = {}

    def contact(self, now, player):
        if self.active is not None and now-self.active['last_contact'] > 8:
            self.close(now,'won' if self.active['kills'] else 'disengaged')
        if self.active is None:
            self.active = {'start':now,'start_hp':player.hp,'start_max_hp':player.max_hp,
                           'last_contact':now,'kills':0}
        self.active['last_contact'] = now

    def close(self, now, outcome):
        if self.active is not None:
            self.fights.append({**self.active,'end':now,'outcome':outcome})
            self.active = None

    def shot(self, sim, shooter, kind):
        if shooter.id == 0:
            self.shots[kind] += 1
            return
        player = sim.fighters[0]
        dx,dy = player.pos[0]-shooter.pos[0],player.pos[1]-shooter.pos[1]
        bearing = math.atan2(dy,dx)
        error = abs((bearing-shooter.aim[0]+math.pi) % (2*math.pi)-math.pi)
        if player.alive and error < .2 and math.hypot(dx,dy) < sim.config['weapons'][kind]['range']+20:
            self.aggressors[shooter.id] = sim.time

    def tick(self, sim):
        player = sim.fighters[0]
        agent = sim.agents[0]
        mode = getattr(getattr(agent,'bot',None),'mode',sim.bot_kind)
        self.modes[mode] += sim.dt
        if mode in ('engage','dodge'):
            self.contact(sim.time,player)
        elif self.active and sim.time-self.active['last_contact'] > 8:
            self.close(sim.time,'won' if self.active['kills'] else 'disengaged')
        for weapon in player.weapons:
            if weapon and weapon['kind'] in self.first_held and self.first_held[weapon['kind']] is None:
                self.first_held[weapon['kind']] = sim.time

    def result(self, sim):
        player = sim.fighters[0]
        self.close(sim.time,'lost' if not player.alive else 'timeout')
        return {'shots':dict(self.shots),'hits':dict(self.hits),'crate_hits':dict(self.crate_hits),
                'first_held':self.first_held,'modes':dict(self.modes),'fights':self.fights,
                'shooters_at_death':sum(sim.time-at <= 2 for at in self.aggressors.values()) if not player.alive else None,
                'final_fight_start_hp':self.fights[-1]['start_hp'] if self.fights else None}
