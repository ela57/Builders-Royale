"""Rule-level simulator regressions; no real-time or match-score assertions."""
from copy import deepcopy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from check.check import Schema
from geo import Geo
from tools.sim.agents import BotAdapter, HouseBot, TIERS
from tools.sim.engine import Simulator
from tools.sim.entities import WorldEntities
from tools.sim.model import Fighter, Zone, cylinder_hit, distance, make_weapon


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT / 'check/stream.jsonl.gz','rt') as stream:
            cls.hello = json.loads(next(stream))
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / 'v2').mkdir()
        world = {'bounds':{'minX':0,'maxX':200,'minY':0,'maxY':200},'prisms':[],
                 'terrain':{'minX':0,'minY':0,'nx':201,'ny':201,'step':1,'file':'terrain.i16'}}
        (root / 'v2/world.json').write_text(json.dumps(world))
        np.zeros((201,201),dtype='<i2').tofile(root / 'v2/terrain.i16')
        cls.geo = Geo(root)
        cls.schema = Schema(ROOT / 'mechfight.schema.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        with patch.object(Simulator,'spawns',return_value=[[10.,10.,0.],[35.,10.,0.]]):
            self.sim = Simulator(self.hello,deepcopy(self.geo),seed=7,bot='example')
        self.sim.crates,self.sim.pickups = [],[]
        self.sim.decide = lambda:None
        self.me,self.enemy = self.sim.fighters
        self.enemy.aim = [math.pi,0]

    def wall(self):
        self.sim.geo.raw_walkable[:,20] = False
        self.sim.geo.solid_top[:,20] = 20
        self.sim.geo.occluder[:,20] = 20

    def test_config_and_legal_real_map_spawns(self):
        geo = Geo(ROOT / 'arena')
        sim = Simulator(self.hello,geo,seed=19,bot='example')
        self.assertEqual(sim.dt,.05)
        self.assertEqual(len(sim.fighters),16)
        self.assertEqual(len(sim.crates),36)
        self.assertEqual(len(sim.pickups),40)
        self.assertEqual(sorted(a.tier for a in sim.agents.values() if isinstance(a,HouseBot)),sorted(['easy']*4+['medium']*4+['hard']*4+['expert']*3))
        for fighter in sim.fighters:
            self.assertTrue(geo.is_walkable(geo.cell(fighter.pos)))
            self.assertLessEqual(distance(fighter.pos,sim.config['zone']['start_center']),448)
            for other in sim.fighters:
                if other.id != fighter.id:
                    self.assertGreaterEqual(distance(fighter.pos,other.pos),130)
        for point in sim.crates+sim.pickups:
            self.assertTrue(geo.is_walkable(geo.cell(point['pos'])))
        self.assertTrue(all(a[0] > b[0] and a[1] > b[1] for a,b in zip(list(TIERS.values()),list(TIERS.values())[1:])))

    def test_twenty_hz_decisions_every_two_ticks_and_schema(self):
        calls = []
        class Agent:
            def act(_, obs):
                self.assertEqual(self.schema.errors(obs,'Observation'),[])
                self.assertEqual(json.loads(json.dumps(obs,allow_nan=False)),obs)
                calls.append((obs['self']['id'],obs['tick']))
                return {}
        self.sim.agents = {0:Agent(),1:Agent()}
        self.sim.decide = lambda:Simulator.decide(self.sim)
        for _ in range(6):
            self.sim.step()
        self.assertEqual(calls,[(0,0),(1,0),(0,2),(1,2),(0,4),(1,4)])
        self.assertAlmostEqual(self.sim.time,.3)

    def test_accel_walk_boost_energy_and_slew(self):
        self.me.action = {'move':[1,0],'aim':[math.pi,1.]}
        self.sim.move(self.me)
        self.assertAlmostEqual(self.me.vel[0],58.5*.05)
        self.assertAlmostEqual(abs(self.me.aim[0]),4*.05)
        self.assertAlmostEqual(self.me.aim[1],3*.05)
        for _ in range(20):
            self.sim.move(self.me)
        self.assertAlmostEqual(self.me.vel[0],11.7)
        self.me.action['boost'] = True
        for _ in range(20):
            self.sim.move(self.me)
        self.assertAlmostEqual(self.me.vel[0],19.5)
        self.assertAlmostEqual(self.me.energy,83)
        self.sim.time = .7
        self.me.action = {}
        self.sim.move(self.me)
        self.assertAlmostEqual(self.me.energy,83+.9)

    def test_jump_jet_ground_and_sliding(self):
        self.me.action = {'jump':True,'boost':True,'move':[0,0]}
        self.sim.one_shots(self.me)
        self.assertFalse(self.me.grounded)
        self.sim.move(self.me)
        self.assertGreater(self.me.pos[2],0)
        self.assertAlmostEqual(self.me.energy,100-15*.05)
        self.me.action = {}
        for _ in range(70):
            self.sim.move(self.me)
        self.assertTrue(self.me.grounded)
        self.assertEqual(self.me.pos[2],0)
        self.wall()
        self.me.pos = [39.9,10,0]
        self.me.vel = [11.7,11.7,0]
        self.me.action = {'move':[1,1]}
        self.sim.move(self.me)
        self.assertEqual(self.me.pos[0],39.9)
        self.assertGreater(self.me.pos[1],10)
        self.assertEqual(self.me.vel[0],0)
        self.assertFalse(self.sim.can_stand(-1,10,0,1.2))
        self.assertTrue(self.sim.can_stand(41,10,25,1.2))

    def test_fast_ground_agrees_with_geo(self):
        self.sim.geo.terrain[:] = np.indices((201,201))[1]*.5
        for x,y in [(10.25,22.5),(-1,500),(200,200)]:
            self.assertAlmostEqual(self.sim.ground(x,y),float(self.sim.geo.ground(x,y)))

    def test_swept_cylinder_including_vertical_and_miss(self):
        self.assertAlmostEqual(cylinder_hit([0,0,4],[100,0,4],[50,0,0],2.5,9),.475)
        self.assertEqual(cylinder_hit([50,0,20],[50,0,0],[50,0,0],2.5,9),.55)
        self.assertIsNone(cylinder_hit([0,5,4],[100,5,4],[50,0,0],2.5,9))
        self.assertIsNone(cylinder_hit([0,0,10],[100,0,10],[50,0,0],2.5,9))

    def test_projectile_speed_damage_mag_autoreload_and_cooldown(self):
        gun = self.me.weapons[0]
        gun['stats']['spread_deg'] = 0
        self.me.action = {'fire':[True,False]}
        self.sim.weapons(self.me)
        self.assertEqual(gun['ammo'],11)
        self.assertEqual(len(self.sim.projectiles),1)
        start = self.sim.projectiles[0]['pos'][0]
        self.sim.advance_projectiles()
        self.assertAlmostEqual(self.sim.projectiles[0]['pos'][0]-start,252*.05)
        self.sim.advance_projectiles()
        self.assertEqual(self.sim.projectiles,[])
        self.assertEqual(self.enemy.hp,1400-45)
        gun['ammo'],gun['cooldown'] = 1,0
        self.sim.weapons(self.me)
        self.assertEqual(gun['ammo'],0)
        self.assertEqual(gun['reload'],2.4)
        self.me.action = {}
        for _ in range(48):
            self.sim.weapons(self.me)
        self.assertEqual(gun['ammo'],12)
        self.assertAlmostEqual(gun['reload'],0)
        self.me.action = {'fire':[True,False]}
        gun['ammo'] = 1000
        for _ in range(200):
            self.sim.weapons(self.me)
        self.assertEqual(1000-gun['ammo'],42)

    def test_gun_wall_and_practice_friendly_fire(self):
        self.wall()
        self.enemy.pos = [70,10,0]
        self.me.action = {'fire':[True,False]}
        self.me.weapons[0]['stats']['spread_deg'] = 0
        self.sim.weapons(self.me)
        for _ in range(6):
            self.sim.advance_projectiles()
        self.assertEqual(self.enemy.hp,1400)
        self.assertEqual(self.sim.projectiles,[])
        ally = Fighter(2,[20,20,0],self.sim.config)
        self.sim.fighters.append(ally)
        self.assertEqual([f.id for f in self.sim.enemies(1)],[0])
        self.sim.mode = 'ffa'
        self.assertEqual([f.id for f in self.sim.enemies(1)],[0,2])

    def test_laser_charge_release_half_speed_and_short(self):
        laser = make_weapon('laser',self.sim.config)
        self.me.weapons[1] = laser
        self.me.action = {'fire':[False,True],'move':[1,0]}
        for _ in range(15):
            self.sim.weapons(self.me)
        self.assertAlmostEqual(laser['charge'],.75)
        self.me.vel = [11.7*.5,0,0]
        self.sim.move(self.me)
        self.assertAlmostEqual(self.me.vel[0],11.7*.5)
        self.me.action['fire'] = [False,False]
        self.sim.weapons(self.me)
        self.assertAlmostEqual(self.enemy.hp,1400-315)
        self.assertEqual(laser['charge'],0)
        self.assertAlmostEqual(laser['cooldown'],1.65)
        self.assertEqual(self.sim.beams[-1]['kind'],'laser')
        self.assertEqual(self.schema.errors(self.sim.observe(self.enemy),'Observation'),[])
        self.enemy.weapons[1] = make_weapon('lightning',self.sim.config)
        laser['cooldown'],laser['charge'] = 0,.4
        self.enemy.action = {'fire':[False,True]}
        self.sim.weapons(self.enemy)
        self.assertEqual(laser['charge'],0)
        self.me.action['fire'] = [False,True]
        self.sim.weapons(self.me)
        self.assertEqual(laser['charge'],0)

    def test_laser_min_release_cookoff_and_blocked_los(self):
        laser = make_weapon('laser',self.sim.config)
        self.me.weapons[1] = laser
        self.enemy.weapons[1] = make_weapon('mortar',self.sim.config)
        self.me.action = {'fire':[False,True]}
        self.sim.weapons(self.me)
        self.me.action['fire'] = [False,False]
        for _ in range(2):
            self.sim.weapons(self.me)
        self.assertAlmostEqual(self.enemy.hp,1400-315*(.35+.65*.15/.75)*1.6)
        self.enemy.pos = [70,10,0]
        self.wall()
        laser['charge'],laser['cooldown'] = .75,0
        hp = self.enemy.hp
        self.sim.weapons(self.me)
        self.assertEqual(self.enemy.hp,hp)

    def test_lightning_range_cone_energy_and_empty_shot(self):
        lightning = make_weapon('lightning',self.sim.config)
        self.me.weapons[1] = lightning
        self.me.action = {'fire':[False,True]}
        self.sim.weapons(self.me)
        self.assertEqual(self.enemy.hp,1340)
        self.assertEqual(self.me.energy,97)
        lightning['cooldown'] = 0
        self.enemy.pos = [10,50,0]
        self.sim.weapons(self.me)
        self.assertEqual(self.enemy.hp,1340)
        self.assertEqual(self.me.energy,94)
        lightning['cooldown'] = 0
        self.enemy.pos = [60,10,0]
        self.sim.weapons(self.me)
        self.assertEqual(self.enemy.hp,1340)
        lightning['cooldown'] = 0
        self.enemy.pos = [25,10,70]
        self.sim.weapons(self.me)
        self.assertEqual(self.enemy.hp,1340)

    def test_rear_armor_zone_damage_and_regen(self):
        self.enemy.aim = [0,0]
        self.sim.damage(0,self.enemy,100,'gun',self.me.pos)
        self.assertEqual(self.enemy.hp,1250)
        self.enemy.stats['armor'] = .1
        self.sim.damage(0,self.enemy,100,'laser',self.me.pos)
        self.assertEqual(self.enemy.hp,1115)
        self.sim.damage(None,self.enemy,100,'zone',None)
        self.assertEqual(self.enemy.hp,1015)
        self.sim.time,self.sim.tick = 4,80
        self.sim.step()
        self.assertEqual(self.enemy.hp,1015)
        self.sim.time,self.sim.tick = 6,120
        self.sim.step()
        self.assertAlmostEqual(self.enemy.hp,1015+45*.05)
        self.enemy.pos = [199,199,0]
        self.sim.time,self.sim.tick = 200,4000
        hp = self.enemy.hp
        self.sim.step()
        self.assertLess(self.enemy.hp,hp)
        self.assertEqual(self.enemy.last_combat,200)

    def test_vision_fov_memory_los_and_sounds(self):
        self.enemy.pos = [10,100,0]
        self.assertEqual(self.sim.observe(self.me)['mechs'],[])
        self.me.aim[0] = math.pi/2
        self.assertEqual(len(self.sim.observe(self.me)['mechs']),1)
        self.me.aim[0] = -math.pi/2
        self.sim.time = 4.9
        self.assertEqual(len(self.sim.observe(self.me)['mechs']),1)
        self.sim.time = 5.1
        self.assertEqual(self.sim.observe(self.me)['mechs'],[])
        self.enemy.pos = [10,40,0]
        self.assertTrue(self.sim.visible(self.me,self.enemy.pos,4.5))
        self.enemy.pos = [70,10,0]
        self.me.aim[0] = 0
        self.wall()
        self.assertFalse(self.sim.visible(self.me,self.enemy.pos,4.5))
        self.sim.emit_sound(self.enemy,'gun')
        self.sim.time += .1
        obs = self.sim.observe(self.me)
        self.assertEqual(obs['sounds'][0]['dist'],'mid')
        self.assertAlmostEqual(obs['sounds'][0]['age'],.1)
        self.assertEqual(self.sim.observe(self.me)['sounds'],[])

    def test_take_drop_failure_and_cooldown_preservation(self):
        self.sim.pickups = [{'id':8000,'kind':'laser','pos':self.me.pos[:]}]
        self.me.action = {'take':8000}
        self.sim.one_shots(self.me)
        self.assertEqual(self.me.weapons[1]['kind'],'laser')
        self.assertEqual(self.me.weapons[1]['cooldown'],1.65)
        self.assertEqual(self.me.events[-1]['type'],'took')
        self.sim.pickups = [{'id':8001,'kind':'laser','pos':self.me.pos[:]}]
        self.me.action = {'take':8001,'drop':0}
        self.sim.one_shots(self.me)
        self.assertEqual(self.me.events[-1]['reason'],'have')
        self.me.weapons[1]['charge'] = .5
        self.me.action = {'drop':1}
        self.sim.one_shots(self.me)
        self.assertIsNotNone(self.me.weapons[1])
        self.me.action = {'take':99999,'drop':0}
        self.sim.one_shots(self.me)
        self.assertEqual(self.me.events[-1]['reason'],'gone')
        self.assertIsNotNone(self.me.weapons[0])
        self.me.weapons[0]['ammo'] = 4
        self.me.action = {'drop':0}
        self.sim.one_shots(self.me)
        dropped = next(p for p in self.sim.pickups if p['kind'] == 'gun')
        self.me.action = {'take':dropped['id']}
        self.sim.one_shots(self.me)
        self.assertEqual(self.me.weapons[0]['ammo'],4)

    def test_crate_cube_offers_upgrade_and_death(self):
        crate = {'id':1000,'pos':self.me.pos[:],'hp':300}
        self.sim.crates.append(crate)
        self.sim.damage(0,crate,300,'laser',self.me.pos)
        self.sim.resolve_deaths()
        self.assertEqual(len(self.sim.cubes),1)
        self.sim.collect(self.me)
        self.assertEqual(self.me.cubes,1)
        self.assertEqual(self.me.max_hp,1456)
        self.assertEqual(len(self.me.offer['choices']),3)
        self.sim.upgrade(self.me,'plating')
        self.assertAlmostEqual(self.me.max_hp,1400*1.04*1.15)
        self.sim.damage(0,self.enemy,2000,'laser',self.me.pos)
        self.sim.resolve_deaths()
        self.assertFalse(self.enemy.alive)
        self.assertEqual(self.enemy.place,2)
        self.assertEqual(len(self.sim.cubes),2)
        self.assertEqual(self.me.kills,1)
        self.assertEqual(self.me.events[-1],{'type':'kill','target':1})

    def test_carried_cubes_drop_with_two_extra_and_tie_goes_to_lower_id(self):
        self.enemy.cubes = 4
        self.sim.damage(0,self.enemy,10000,'gun',self.me.pos)
        self.sim.resolve_deaths()
        self.assertEqual(len(self.sim.cubes),6)
        self.assertTrue(all(c['pos'] == self.enemy.pos for c in self.sim.cubes))
        self.sim.cubes = []
        self.enemy.alive,self.enemy.hp = True,1400
        self.me.pos,self.enemy.pos = [10,10,0],[12,10,0]
        self.sim.add_cubes([11,10,0],1)
        self.sim.collect(self.enemy)
        self.assertEqual(self.enemy.cubes,4)
        self.sim.collect(self.me)
        self.assertEqual(self.me.cubes,1)
        self.assertEqual(self.sim.cubes,[])
        self.sim.collect(self.me)
        self.assertEqual(self.me.cubes,1)

    def test_closest_cube_collector_and_vertical_reach(self):
        self.enemy.pos = [12,10,0]
        self.sim.add_cubes([11.5,10,0],1)
        self.sim.collect(self.me)
        self.assertEqual(self.me.cubes,0)
        self.sim.collect(self.enemy)
        self.assertEqual(self.enemy.cubes,1)
        self.sim.add_cubes([11,10,8],1)
        self.sim.collect(self.me)
        self.assertEqual(self.me.cubes,0)

    def test_offer_expiry_explicit_swap_and_event_ages(self):
        self.me.cubes = 1
        self.sim.offer_upgrade(self.me)
        first = self.me.offer['choices'][0]['id']
        self.sim.time = self.me.offer['expires']+.1
        self.me.action = {'pick':99}
        self.sim.one_shots(self.me)
        self.assertIsNone(self.me.offer)
        self.assertIn(first,self.me.upgrades)
        self.sim.pickups = [{'id':8000,'kind':'laser','pos':self.me.pos[:]}]
        self.me.action = {'take':8000,'drop':0}
        self.sim.one_shots(self.me)
        self.assertEqual(self.me.weapons[0]['kind'],'laser')
        self.assertIsNone(self.me.weapons[1])
        self.sim.beams = [{'by':1,'kind':'laser','path':[[10,10,7],[35,10,7]],'age':self.sim.time}]
        self.sim.time += .1
        self.assertAlmostEqual(self.sim.observe(self.me)['beams'][0]['age'],.1)

    def test_zone_containment_timing_and_final_radius(self):
        zone = Zone(self.hello['config']['zone'],random.Random(1))
        for p in zone.phases:
            self.assertLessEqual(distance(p['center'],p['next_center'])+p['next_radius'],p['radius']+1e-9)
        self.assertEqual(zone.at(0)['radius'],560)
        self.assertEqual(zone.at(60)['radius'],560)
        self.assertEqual(zone.at(85)['radius'],470)
        self.assertEqual(zone.at(110)['radius'],380)
        self.assertEqual(zone.at(480)['radius'],0)

    def test_same_tick_ties_timeout_and_empty_world(self):
        self.me.hp,self.enemy.hp = 0,0
        self.sim.resolve_deaths()
        self.assertEqual(self.me.place,self.enemy.place)
        self.assertEqual(self.sim.enemies(0),[])
        self.assertEqual(self.sim.finish()['place'],1)
        self.me.alive,self.enemy.alive = True,True
        self.me.hp,self.enemy.hp = 100,500
        self.sim.time = 480
        result = self.sim.finish()
        self.assertEqual((result['place'],result['score'],result['cause']),(2,14,'timeout'))
        self.sim.projectiles = []
        self.sim.advance_projectiles()

    def test_house_shared_vision_and_hunter_escalation(self):
        self.sim.fighters = [Fighter(i,[10.+(i%4)*40,10.+(i//4)*40,0],self.sim.config) for i in range(16)]
        self.sim.hunter_order = list(range(15,0,-1))
        calls = []
        for f in self.sim.fighters[1:]:
            agent = HouseBot(f,'easy',self.sim.geo,random.Random(f.id))
            agent.act = lambda sim,me,shared,hunt:calls.append((me.id,shared,hunt)) or {}
            self.sim.agents[f.id] = agent
        self.sim.agents[0] = type('Idle',(),{'act':lambda _,obs:{}})()
        self.sim.visible = lambda viewer,point,height=0:viewer.id == 2
        Simulator.decide(self.sim)
        self.assertTrue(all(call[1]['id'] == 0 for call in calls))
        self.assertEqual([slot for slot,_,hunt in calls if hunt],[15])
        self.sim.fighters[0].kills = 2
        calls.clear()
        Simulator.decide(self.sim)
        self.assertEqual([slot for slot,_,hunt in calls if hunt],[13,14,15])
        self.sim.time = 301
        calls.clear()
        self.sim.visible = lambda *args:False
        Simulator.decide(self.sim)
        self.assertTrue(all(hunt and shared is None for _,shared,hunt in calls))
        self.assertIsNone(self.sim.last_known)

    def test_exceptions_fail_loudly_and_empty_observation(self):
        self.enemy.alive = False
        obs = self.sim.observe(self.me)
        for name in ('mechs','projectiles','crates','cubes','pickups','sounds','events','beams'):
            self.assertEqual(obs[name],[])
        self.assertEqual(self.schema.errors(obs,'Observation'),[])
        class Broken:
            def act(self, obs):
                raise RuntimeError('bad bot')
        self.sim.agents = {0:Broken()}
        with self.assertRaisesRegex(RuntimeError,'bad bot'):
            Simulator.decide(self.sim)

    def test_example_wrapper_matches_unmodified_class(self):
        path = Path(__file__).with_name('example_bot.py')
        spec = importlib.util.spec_from_file_location('original_example',path)
        module = importlib.util.module_from_spec(spec)
        with patch('mechfight.run') as run:
            spec.loader.exec_module(module)
            run.assert_called_once_with(module.Bot)
        reference = module.Bot(self.sim.make_hello(0))
        adapter = BotAdapter('example',self.sim.make_hello(0),self.sim.geo)
        obs = self.sim.observe(self.me)
        self.assertEqual(adapter.act(obs),reference.act(obs))

    def test_telemetry_records_real_shots_modes_fights_and_loot(self):
        self.sim.agents[0].bot.mode = 'engage'
        self.me.weapons[1] = make_weapon('laser',self.sim.config)
        self.sim.telemetry.tick(self.sim)
        self.me.action = {'fire':[True,False]}
        self.me.weapons[0]['stats']['spread_deg'] = 0
        self.sim.weapons(self.me)
        self.sim.advance_projectiles()
        self.sim.advance_projectiles()
        self.sim.time = 1
        self.sim.damage(1,self.me,2000,'laser',self.enemy.pos)
        self.sim.resolve_deaths()
        data = self.sim.finish()['telemetry']
        self.assertEqual(data['shots'],{'gun':1})
        self.assertEqual(data['hits'],{'gun':1})
        self.assertEqual(data['first_held'],{'laser':0,'lightning':None})
        self.assertEqual(data['shooters_at_death'],1)
        self.assertEqual(data['final_fight_start_hp'],1400)
        self.assertEqual(data['fights'][0]['outcome'],'lost')
        self.assertAlmostEqual(data['modes']['engage'],.05)

    def test_telemetry_empty_and_disengaged_episode(self):
        data = self.sim.finish()['telemetry']
        self.assertEqual(data['fights'],[])
        self.assertEqual(data['shots'],{})
        self.assertIsNone(data['final_fight_start_hp'])
        self.sim.telemetry.contact(0,self.me)
        self.sim.telemetry.contact(10,self.me)
        self.assertEqual(self.sim.telemetry.fights[0]['outcome'],'disengaged')

    def landed_drop(self):
        self.sim.entities = WorldEntities(self.sim,drops=True)
        self.sim.entities.drops = [{'id':90000,'pos':[10.,10.,0.],'radius':8,
            'lands_at':0.,'capture_s':4.,'progress':0.,'holder':None,'landed':False}]
        return self.sim.entities.drops[0]

    def drone(self):
        self.sim.entities = WorldEntities(self.sim,npcs=True)
        npc = self.sim.entities.npcs[0]
        npc.update(pos=[80.,80.,0.],home=[80.,80.,0.],wander=[80.,80.,0.])
        self.sim.entities.npcs = [npc]
        return npc

    def test_optional_entities_are_empty_by_default(self):
        self.assertEqual(self.sim.observe(self.me)['drops'],[])
        self.assertEqual(self.sim.entities.npcs,[])

    def test_entities_do_not_retain_finished_simulator(self):
        entities = self.sim.entities
        sim = weakref.ref(self.sim)
        self.sim = None
        self.assertIsNone(sim())
        with self.assertRaises(ReferenceError):
            entities.open_point()

    def test_drop_announcement_landing_and_protocol_shape(self):
        self.sim.entities = WorldEntities(self.sim,drops=True)
        self.sim.zone.at = lambda now:{'next_center':[100,100],'next_radius':80,
            'center':[100,100],'radius':100,'shrink_start':120,'shrink_end':150,'dps':1}
        self.sim.time = 69.95
        self.sim.entities.announce()
        self.assertEqual(self.sim.entities.drops,[])
        self.sim.time = 70
        self.sim.entities.announce()
        self.sim.entities.announce()
        self.assertEqual(len(self.sim.entities.drops),1)
        drop = self.sim.entities.drops[0]
        obs = self.sim.observe(self.me)
        self.assertEqual(self.schema.errors(obs,'Observation'),[])
        self.assertEqual(obs['drops'][0]['lands_in'],25)
        self.assertLessEqual(distance(drop['pos'],[100,100]),80)
        self.me.pos = drop['pos'][:]
        self.sim.entities.capture()
        self.assertEqual(drop['progress'],0)
        self.sim.time = 95
        self.sim.entities.capture()
        self.assertAlmostEqual(drop['progress'],.05)
        self.assertTrue(any(sound['kind'] == 'drop' for sound in self.me.sounds))

    def test_drop_capture_repairs_rewards_and_offers(self):
        drop = self.landed_drop()
        self.me.hp,self.me.energy = 50,1
        for _ in range(79):
            self.sim.entities.capture()
        self.assertEqual(self.me.cubes,0)
        self.sim.entities.capture()
        self.assertEqual(self.me.cubes,3)
        self.assertEqual(self.me.hp,self.me.max_hp)
        self.assertEqual(self.me.energy,self.me.stats['energy'])
        self.assertIsNotNone(self.me.offer)
        self.assertEqual(self.sim.entities.drops,[])
        self.assertIn({'type':'drop','id':drop['id'],'cubes':3},self.me.events)
        self.assertEqual(self.sim.entities.drop_captures[0],1)
        self.sim.entities.capture()
        self.assertEqual(self.me.cubes,3)

    def test_drop_contest_freezes_empty_and_new_holder_reset(self):
        drop = self.landed_drop()
        self.sim.entities.capture()
        self.enemy.pos = self.me.pos[:]
        self.sim.entities.capture()
        self.assertEqual(drop['progress'],.05)
        self.me.pos = [60.,10.,0.]
        self.sim.entities.capture()
        self.assertEqual(drop['holder'],1)
        self.assertEqual(drop['progress'],.05)
        self.enemy.pos = [60.,20.,0.]
        self.sim.entities.capture()
        self.assertIsNone(drop['holder'])
        self.assertEqual(drop['progress'],0)
        self.me.pos = [10.,10.,20.]
        self.sim.entities.capture()
        self.assertEqual(drop['progress'],0)

    def test_drop_no_legal_site_fails_explicitly(self):
        self.sim.entities = WorldEntities(self.sim,drops=True)
        self.sim.time = 70
        self.sim.zone.at = lambda now:{'next_center':[1000,1000],'next_radius':1}
        with self.assertRaisesRegex(RuntimeError,'No legal supply-drop'):
            self.sim.entities.announce()

    def test_npc_protocol_vision_memory_and_zone_immunity(self):
        npc = self.drone()
        self.assertEqual(len(WorldEntities(self.sim,npcs=True).npcs),12)
        self.me.pos = [60.,80.,0.]
        obs = self.sim.observe(self.me)
        self.assertEqual(self.schema.errors(obs,'Observation'),[])
        self.assertEqual([a['id'] for a in obs['mechs'] if a['kind'] == 'npc'],[npc['id']])
        self.me.pos = [10.,10.,0.]
        self.me.aim = [-math.pi,0]
        self.sim.time = 4.9
        self.assertTrue(self.sim.entities.npc_observations(self.me))
        self.sim.time = 5.1
        self.assertEqual(self.sim.entities.npc_observations(self.me),[])
        self.me.alive = self.enemy.alive = False
        self.sim.zone.at = lambda now:{'center':[-1000,-1000],'radius':1,'dps':1000}
        self.sim.step()
        self.assertEqual(npc['hp'],350)

    def test_npc_aggro_bullets_retaliation_and_rewards(self):
        npc = self.drone()
        self.me.pos = [60.,80.,0.]
        self.me.aim = [0,0]
        self.sim.config['npc']['spread_deg'] = 0
        self.sim.entities.move_npcs()
        self.assertEqual(npc['target'],0)
        self.assertEqual(len(self.sim.projectiles),1)
        self.assertEqual(self.sim.observe(self.me)['projectiles'][0]['kind'],'npc')
        hp = self.me.hp
        for _ in range(6):
            self.sim.advance_projectiles()
        self.assertLess(self.me.hp,hp)
        self.assertEqual(self.me.events[-1]['weapon'],'npc')
        self.sim.damage(1,npc,100,'gun',self.enemy.pos)
        self.assertEqual(npc['target'],1)
        self.sim.damage(0,npc,250,'laser',self.me.pos)
        self.sim.entities.resolve_npcs()
        self.assertEqual(len(self.sim.cubes),2)
        self.assertEqual(self.sim.entities.npcs,[])
        self.assertEqual(self.me.kills,0)
        self.assertEqual(self.sim.entities.npc_kills[0],1)
        self.assertIn({'type':'kill','target':npc['id']},self.me.events)
        self.assertEqual(self.sim.telemetry.npc_hits['laser'],1)
        self.sim.entities.resolve_npcs()
        self.assertEqual(len(self.sim.cubes),2)

    def test_npc_home_leash_deaggro_and_wall_sliding(self):
        npc = self.drone()
        self.sim.entities.move_npcs()
        self.assertIsNone(npc['target'])
        self.assertLessEqual(distance(npc['wander'],npc['home']),30)
        npc.update(pos=[140.,80.,0.],target=0)
        self.me.pos = [150.,80.,0.]
        self.sim.entities.move_npcs()
        self.assertIsNone(npc['target'])
        self.assertTrue(npc['returning'])
        self.assertLess(npc['pos'][0],140)
        npc.update(pos=[80.,80.,0.],target=0,returning=False)
        self.me.pos = [180.,80.,0.]
        self.sim.entities.move_npcs()
        self.assertIsNone(npc['target'])
        npc.update(pos=[39.,40.,0.],home=[39.,40.,0.],target=None,returning=False,wander=[60,60])
        self.me.alive = self.enemy.alive = False
        self.wall()
        for _ in range(20):
            self.sim.entities.move_npcs()
        self.assertLess(npc['pos'][0],40)
        self.assertGreater(npc['pos'][1],40)

    def test_npc_hitscan_lightning_and_npc_death_attribution(self):
        npc = self.drone()
        self.me.pos,self.me.aim = [60.,80.,0.],[0,0]
        target,_ = self.sim.ray_target(0,[60,80,1.75],[100,80,1.75])
        self.assertIs(target,npc)
        self.me.weapons = [make_weapon('lightning',self.sim.config),None]
        self.me.action = {'fire':[True,False]}
        self.me.aim = [0,-.27]
        self.sim.weapons(self.me)
        self.assertEqual(npc['hp'],290)
        self.sim.damage(npc['id'],self.me,100000,'npc',npc['pos'])
        self.sim.resolve_deaths()
        self.assertEqual(self.me.cause,'npc')
        self.assertEqual(self.enemy.kills,0)

    def test_house_defends_against_nearby_drone(self):
        npc = self.drone()
        self.enemy.pos,self.enemy.aim = [60.,80.,0.],[0,0]
        agent = self.sim.agents[1]
        agent.noise,agent.next_noise = [0,0],100
        action = agent.act(self.sim,self.enemy,None,False)
        self.assertTrue(action['fire'][0])
        self.assertLess(action['aim'][1],0)

    def test_idle_house_competes_for_landed_supply_drop(self):
        self.landed_drop()
        self.enemy.pos = [35.,10.,0.]
        self.sim.zone.at = lambda now:{'center':[100,100],'radius':500,'next_center':[100,100],
            'next_radius':500,'shrink_start':100,'shrink_end':150,'dps':1}
        agent = self.sim.agents[1]
        action = agent.act(self.sim,self.enemy,None,False)
        self.assertLess(action['move'][0],0)
        self.enemy.pos = [10.,10.,0.]
        self.assertEqual(agent.act(self.sim,self.enemy,None,False)['move'],[0,0])


if __name__ == '__main__':
    unittest.main()
