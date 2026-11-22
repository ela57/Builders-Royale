"""Synthetic observations isolate strategic decisions and weapon sequencing."""
import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bot import Bot
from brain import Memory, pick_offer, safe_point, take_weapon
from combat import aim_at, centre, dodge, fire_control, intercept


class OpenGeo:
    def __init__(self):
        self.clear = True
        self.cover = (10,10)

    def los(self, a, b):
        return self.clear

    def walk_clear(self, a, b):
        return True

    def ground(self, x, y):
        return 0

    def nearest(self, point, radius=3):
        return tuple(point[:2])

    def point(self, cell):
        return cell

    def find_cover(self, *args, **kwargs):
        return self.cover

    def search(self, start, goal):
        class Search:
            path = [tuple(goal[:2])]

            def advance(self, budget_ms):
                return 'complete'
        return Search()


def weapon(kind, **updates):
    stats = {'gun':{'speed':252,'range':230},
             'laser':{'charge':.75,'hold_s':2,'range':280},
             'lightning':{'range':45,'cone_deg':30,'energy':3}}[kind]
    return {'kind':kind,'cooldown':0,'stats':stats,'ammo':12,'charge':0,**updates}


def enemy(x=80, y=0, id=1, **updates):
    return {'id':id,'kind':'mech','pos':[x,y,0],'vel':[0,0,0],
            'aim':[math.pi,0],'hp':1400,'max_hp':1400,'weapons':['gun'],**updates}


def observation():
    return {'type':'obs','tick':0,'t':0,'self':{
        'id':0,'pos':[0,0,0],'vel':[2,0,0],'aim':[0,0],
        'hp':1400,'max_hp':1400,'energy':100,'cubes':0,'in_zone':True,
        'weapons':[weapon('gun'),None],
        'stats':{'muzzle_height':7,'walk_speed':10,'regen_delay':5}},
        'zone':{'center':[0,0],'radius':500,'next_center':[0,0],'next_radius':300,
                'shrink_start':100,'shrink_end':150},
        'mechs':[],'projectiles':[],'cubes':[],'pickups':[],'crates':[],
        'drops':[],'events':[],'sounds':[],'offer':None}


class BotTests(unittest.TestCase):
    def setUp(self):
        self.geo = OpenGeo()
        self.bot = Bot({'seed':7},geo=self.geo)
        self.obs = observation()

    def test_startup_collects_then_freezes_loaded_geometry_without_disabling_gc(self):
        for injected in (False,True):
            with self.subTest(injected=injected):
                calls = []
                def load(*args):
                    calls.append('load')
                    return self.geo
                with patch('bot.Geo',side_effect=load), \
                     patch('bot.gc.collect',side_effect=lambda:calls.append('collect')), \
                     patch('bot.gc.freeze',side_effect=lambda:calls.append('freeze')), \
                     patch('bot.gc.disable') as disable:
                    bot = Bot({'seed':7},geo=self.geo if injected else None)
                self.assertIs(bot.geo,self.geo)
                self.assertEqual(calls,(['load'] if not injected else [])+['collect','freeze'])
                disable.assert_not_called()

    def aligned(self, target):
        self.obs['self']['aim'] = aim_at([0,0,7],centre(target))

    def test_intercept_hits_moving_target(self):
        origin,target,velocity,speed = [0,0,7],[100,20,4.5],[0,15,1],252
        hit = intercept(origin,target,velocity,speed)
        t = math.dist(origin,hit)/speed
        for i in range(3):
            self.assertAlmostEqual(hit[i],target[i]+velocity[i]*t)
        self.assertIsNone(intercept([0,0,0],[10,0,0],[300,0,0],100))
        self.assertIsNone(intercept(origin,target,velocity,0))
        self.assertEqual(intercept([0,0,0],[0,0,0],[0,0,0],10),(0,0,0))
        self.assertEqual(intercept([0,0,0],[10,0,0],[-10,0,0],10),(5,0,0))

    def test_gun_uses_lead_and_actual_aim(self):
        target = enemy(100,vel=[0,20,0])
        self.obs['mechs'] = [target]
        me = self.obs['self']
        lead = intercept([0,0,7],centre(target),target['vel'],252)
        me['aim'] = aim_at([0,0,7],lead)
        action = self.bot.act(self.obs)
        self.assertTrue(action['fire'][0])
        self.assertAlmostEqual(action['aim'][0],me['aim'][0])
        me['aim'] = [math.pi,0]
        self.assertFalse(self.bot.act(self.obs)['fire'][0])

    def test_laser_charge_release_and_hold_limit(self):
        target = enemy()
        self.obs['mechs'] = [target]
        self.aligned(target)
        laser = weapon('laser')
        self.obs['self']['weapons'] = [laser,None]
        for charge,expected in [(0,True),(.4,True),(.75,False)]:
            laser['charge'] = charge
            self.assertEqual(self.bot.act(self.obs)['fire'][0],expected)
        self.obs['self']['aim'] = [1,0]
        laser['charge'] = 1
        self.assertTrue(self.bot.act(self.obs)['fire'][0])
        laser['charge'] = 2.7
        self.assertFalse(self.bot.act(self.obs)['fire'][0])
        laser['charge'],laser['cooldown'] = 0,1.5
        self.assertFalse(self.bot.act(self.obs)['fire'][0])

    def test_take_drop_duplicates_range_and_charging(self):
        me = self.obs['self']
        pickup = {'id':8000,'kind':'laser','pos':[2,0,0]}
        self.assertEqual(take_weapon(me,[pickup]),{'take':8000})
        me['weapons'][1] = weapon('lightning')
        self.assertEqual(take_weapon(me,[pickup]),{'take':8000,'drop':0})
        me['weapons'][1] = weapon('laser',charge=.3)
        pickup['kind'] = 'lightning'
        self.assertEqual(take_weapon(me,[pickup]),{})
        pickup['kind'] = 'laser'
        self.assertEqual(take_weapon(me,[pickup]),{})
        me['weapons'][1] = None
        pickup['pos'] = [5,0,0]
        self.assertEqual(take_weapon(me,[pickup]),{})
        pickup['pos'] = [0,0,5]
        self.assertEqual(take_weapon(me,[pickup]),{})
        self.assertEqual(take_weapon(me,[]),{})
        pickup['kind'] = 'mortar'
        pickup['pos'] = [0,0,0]
        self.assertEqual(take_weapon(me,[pickup]),{})

    def test_damaging_mech_memory_ignores_npcs_zone_self_and_expires(self):
        memory = Memory()
        self.obs['mechs'] = [enemy(150),enemy(200,id=2)]
        self.obs['events'] = [{'type':'damaged','by':1,'weapon':'gun','amount':30,'bearing':0},
                              {'type':'damaged','by':99,'weapon':'npc','bearing':1},
                              {'type':'damaged','by':None,'weapon':'zone'},
                              {'type':'damaged','by':0,'weapon':'mortar','bearing':0},
                              {'type':'damaged','by':3,'weapon':'laser','amount':0,'bearing':1}]
        memory.update(self.obs)
        self.assertEqual(set(memory.attackers),{1})
        self.obs['events'] = [{'type':'damaged','by':2,'weapon':'laser','bearing':0}]
        memory.update(self.obs)
        self.assertEqual(set(memory.attackers),{1,2})
        self.obs['events'] = [{'type':'kill','target':2}]
        memory.update(self.obs)
        self.assertEqual(set(memory.attackers),{1})
        self.obs.update(t=6,events=[],mechs=[])
        memory.update(self.obs)
        self.assertEqual(memory.attackers,{})

    def test_long_range_crossfire_precedes_loot_and_does_not_chase_finishes(self):
        self.geo.cover = None
        self.obs['mechs'] = [enemy(150,hp=100),enemy(220,id=2)]
        self.obs['cubes'] = [{'id':8000,'pos':[20,0,0]}]
        self.obs['events'] = [{'type':'damaged','by':i,'weapon':'gun','bearing':0} for i in (1,2)]
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'disengage')
        self.assertLess(action['move'][0],0)
        self.assertTrue(action['boost'])
        self.assertEqual(self.bot.target_id,1)

    def test_crowd_fallback_uses_centroid_and_cover_considers_all_known_enemies(self):
        self.geo.cover = None
        self.obs['mechs'] = [enemy(80,40),enemy(80,-40,id=2),enemy(250,id=3)]
        self.obs['events'] = [{'type':'damaged','by':i,'weapon':'gun','bearing':0} for i in (1,2)]
        with patch.object(self.geo,'find_cover',return_value=None) as cover:
            self.bot.act(self.obs)
        self.assertEqual(self.bot.nav.goal,(-30.,0.))
        self.assertEqual(len(cover.call_args.args[1]),3)

    def test_unseen_crossfire_uses_reported_bearings_not_hidden_positions(self):
        self.geo.cover = None
        self.obs['events'] = [{'type':'damaged','by':i,'weapon':'laser','bearing':0} for i in (1,2)]
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'disengage')
        self.assertLess(action['move'][0],0)
        self.assertEqual(self.bot.memory.attackers[1]['pos'],[100,0,0])
        self.assertFalse(any(action['fire']))

    def test_crossfire_escape_cannot_be_reversed_by_dodge_or_npc_steering(self):
        self.geo.cover = None
        self.obs['mechs'] = [enemy(150),enemy(200,id=2)]
        self.obs['events'] = [{'type':'damaged','by':i,'weapon':'gun','bearing':0} for i in (1,2)]
        with patch('bot.dodge',return_value=[1,0]), patch.object(self.bot,'_avoid_npcs',return_value=[1,0]):
            action = self.bot.act(self.obs)
        self.assertLess(action['move'][0],0)
        self.assertEqual(self.bot.mode,'disengage')

    def test_crossfire_keeps_urgent_zone_escape_and_single_attacker_behavior(self):
        self.obs['mechs'] = [enemy(80)]
        self.obs['events'] = [{'type':'damaged','by':1,'weapon':'gun','bearing':math.pi}]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'engage')
        self.obs['self']['pos'] = [300,0,0]
        self.obs['zone']['next_radius'] = 100
        self.obs.update(t=99,mechs=[enemy(80),enemy(120,id=2)])
        self.obs['events'].append({'type':'damaged','by':2,'weapon':'laser','bearing':math.pi})
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'zone')
        self.assertLess(action['move'][0],0)

    def test_offer_priority_and_empty(self):
        self.assertIsNone(pick_offer(None))
        self.assertIsNone(pick_offer({'choices':[]}))
        self.obs['offer'] = {'choices':[{'id':'servos'},{'id':'armor'},{'id':'plating'}]}
        self.assertEqual(self.bot.act(self.obs)['pick'],2)
        self.assertEqual(pick_offer({'choices':[{'id':'unknown'}]}),0)

    def test_zone_safe_point_timing_and_boost_reserve(self):
        me = self.obs['self']
        me['pos'] = [300,0,0]
        zone = self.obs['zone']
        zone['next_center'],zone['next_radius'] = [0,0],100
        point,urgent,_,travel = safe_point(me,zone,65.9)
        self.assertEqual(travel,24)
        self.assertFalse(urgent)
        self.assertEqual(point,[300,0])
        point,urgent,circle,_ = safe_point(me,zone,66)
        self.assertTrue(urgent)
        self.assertEqual(point,[60,0])
        self.assertEqual(circle['radius'],100)
        self.obs['t'] = 66
        self.assertTrue(self.bot.act(self.obs)['boost'])
        self.assertEqual(self.bot.mode,'zone')
        me['energy'] = 20
        self.assertFalse(self.bot.act(self.obs)['boost'])
        me['in_zone'] = False
        self.assertTrue(safe_point(me,zone,0)[1])

    def test_dodge_charging_laser(self):
        target = enemy(charging=True)
        self.obs['mechs'] = [target]
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'dodge')
        self.assertTrue(action['boost'])
        self.assertAlmostEqual(action['move'][0],0)
        self.assertEqual(abs(action['move'][1]),1)
        target['aim'] = [0,0]
        self.assertNotEqual(self.bot.act(self.obs)['move'],action['move'])

    def test_mortar_landing_dodge(self):
        shell = {'kind':'mortar','pos':[5,0,11],'vel':[0,0,0]}
        move = dodge(self.obs['self'],[],[shell],self.geo,{})
        self.assertLess(move[0],0)
        shell['pos'][0] = 50
        self.assertIsNone(dodge(self.obs['self'],[],[shell],self.geo,{}))

    def test_unseen_attacker_bearing(self):
        self.obs['events'] = [{'type':'damaged','by':9,'bearing':1.2}]
        action = self.bot.act(self.obs)
        self.assertEqual(action['aim'],[1.2,0])
        self.assertEqual(action['fire'],[False,False])

    def test_never_fire_when_blocked_or_not_visible(self):
        target = enemy(30)
        self.aligned(target)
        for kind in ('gun','laser','lightning'):
            self.obs['self']['weapons'] = [weapon(kind),None]
            self.obs['mechs'] = [target]
            self.geo.clear = False
            self.assertEqual(self.bot.act(self.obs)['fire'],[False,False])
            self.geo.clear = True
            self.assertEqual(fire_control(self.obs['self'],target,False,[target],self.geo,{})[1],[False,False])

    def test_lightning_nearest_cone_energy_and_dual_fire(self):
        target = enemy(30)
        me = self.obs['self']
        self.aligned(target)
        me['weapons'] = [weapon('gun'),weapon('lightning')]
        self.assertEqual(fire_control(me,target,True,[target],self.geo,{})[1],[True,True])
        closer = enemy(10,id=2)
        self.assertFalse(fire_control(me,target,True,[closer,target],self.geo,{})[1][1])
        me['energy'] = 2
        self.assertFalse(fire_control(me,target,True,[target],self.geo,{})[1][1])
        me['energy'],me['aim'] = 100,[math.pi,0]
        self.assertFalse(fire_control(me,target,True,[target],self.geo,{})[1][1])

    def test_memory_expiry_took_sounds_and_reload(self):
        self.obs['mechs'] = [enemy()]
        self.obs['events'] = [{'type':'took','id':8000,'slot':1,'kind':'laser'}]
        self.obs['sounds'] = [{'kind':'gun','bearing':1,'age':0}]
        memory = Memory()
        memory.update(self.obs)
        self.assertEqual(memory.weapons[1]['kind'],'laser')
        self.assertEqual(len(memory.enemies),1)
        self.obs.update(t=5,mechs=[],events=[],sounds=[])
        memory.update(self.obs)
        self.assertEqual(memory.enemies,{})
        self.assertEqual(memory.sounds,[])
        self.obs['self']['weapons'][0]['ammo'] = 3
        self.assertTrue(self.bot.act(self.obs)['reload'])

    def test_outnumbered_and_low_hp_disengage_then_regen(self):
        self.obs['mechs'] = [enemy(),enemy(70,10,id=2)]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'disengage')
        self.obs['self']['hp'] = 300
        self.obs['mechs'] = [enemy()]
        self.bot.act(self.obs)
        self.assertTrue(self.bot.regenerating)
        self.obs.update(t=6,mechs=[])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'regen')
        self.obs['self']['hp'] = 1400
        self.bot.act(self.obs)
        self.assertFalse(self.bot.regenerating)

    def test_loot_crate_npc_drop_and_threat_priority(self):
        self.obs['cubes'] = [{'id':5000,'pos':[20,0,0]}]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'loot')
        self.obs['cubes'] = []
        self.obs['self']['pos'] = [20,0,0]
        self.obs['crates'] = [{'id':1000,'pos':[40,0,0]}]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'crate')
        self.obs['mechs'] = [enemy()]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.target_id,1)
        self.obs.update(t=6,mechs=[enemy(60,id=100,kind='npc')],crates=[],events=[{'type':'damaged','by':100,'bearing':0}])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'npc')
        self.obs.update(t=20,events=[],mechs=[],drops=[{'id':20000,'pos':[30,0,0],'radius':8}])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'drop')

    def test_idle_sound_scan_and_stuck_recovery(self):
        self.obs['self']['weapons'][1] = weapon('laser')
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'hide')
        self.assertLessEqual(math.hypot(*action['move']),.4)
        self.obs['sounds'] = [{'kind':'laser','bearing':2,'age':0}]
        self.assertEqual(self.bot.act(self.obs)['aim'],[2,0])
        self.obs['cubes'] = [{'id':5000,'pos':[40,0,0]}]
        self.obs['self']['vel'] = [0,0,0]
        self.bot.act(self.obs)
        self.obs['t'] = 1.1
        action = self.bot.act(self.obs)
        self.assertTrue(action['jump'])
        self.assertTrue(action['boost'])

    def test_remembered_loot_expiry_gone_and_kill_cleanup(self):
        self.obs['pickups'] = [{'id':8000,'kind':'laser','pos':[60,0,0]}]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'equip')
        self.obs.update(t=1,pickups=[],mechs=[enemy(180)])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'equip')
        self.assertEqual(self.bot.nav.goal,(60,0))
        self.obs['events'] = [{'type':'take_failed','id':8000,'reason':'gone'},{'type':'kill','target':1}]
        self.obs['mechs'] = []
        self.bot.act(self.obs)
        self.assertEqual(self.bot.memory.pickups,{})
        self.assertEqual(self.bot.memory.enemies,{})
        self.obs.update(events=[],t=100)
        self.bot.memory.pickups = {8001:{'id':8001,'kind':'laser','pos':[80,0,0],'seen':1}}
        self.bot.act(self.obs)
        self.assertEqual(self.bot.memory.pickups,{})

    def test_loot_detour_rejects_danger_and_unreachable_roof(self):
        self.obs['pickups'] = [{'id':8000,'kind':'laser','pos':[60,0,20]}]
        self.bot.act(self.obs)
        self.assertNotEqual(self.bot.mode,'equip')
        self.obs['pickups'][0]['pos'][2] = 0
        self.obs['mechs'] = [enemy(70)]
        self.bot.act(self.obs)
        self.assertNotEqual(self.bot.mode,'equip')

    def test_kill_site_survives_missing_cubes_and_dead_drone_avoidance(self):
        self.obs['mechs'] = [enemy(55,id=100,kind='npc')]
        self.bot.memory.update(self.obs)
        self.obs.update(t=1,mechs=[],events=[{'type':'kill','target':100}])
        action = self.bot.act(self.obs)
        self.assertNotIn(100,self.bot.memory.npcs)
        self.assertEqual(self.bot.mode,'loot')
        self.assertEqual(self.bot.nav.goal,(55,0))
        self.assertGreater(action['move'][0],0)
        self.obs.update(t=25,events=[])
        self.bot.act(self.obs)
        self.assertIn(100,self.bot.memory.harvests)
        self.obs['self']['pos'] = [54,0,0]
        self.bot.act(self.obs)
        self.assertNotIn(100,self.bot.memory.harvests)

    def test_cubes_precede_weapon_detours_but_not_nearby_threats_or_zone(self):
        self.obs['cubes'] = [{'id':5000,'pos':[20,0,0]}]
        self.obs['pickups'] = [{'id':8000,'kind':'laser','pos':[60,0,0]}]
        self.obs['mechs'] = [enemy(190)]
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'loot')
        self.assertEqual(self.bot.nav.goal,(20,0))
        self.obs['mechs'] = [enemy(70)]
        self.bot.act(self.obs)
        self.assertNotEqual(self.bot.mode,'loot')
        self.obs['self']['pos'] = [490,0,0]
        self.obs['cubes'][0]['pos'] = [470,0,0]
        self.obs['mechs'] = []
        self.obs['t'] = 99
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'zone')

    def test_cube_memory_outlasts_detour_and_only_clears_at_collection_distance(self):
        self.obs['cubes'] = [{'id':5000,'pos':[30,0,0]}]
        self.bot.memory.update(self.obs)
        self.obs.update(t=25,cubes=[])
        self.obs['self']['pos'] = [26,0,0]
        self.bot.memory.update(self.obs)
        self.assertIn(5000,self.bot.memory.cubes)
        self.obs['self']['pos'] = [29,0,0]
        self.bot.memory.update(self.obs)
        self.assertNotIn(5000,self.bot.memory.cubes)
        self.bot.memory.harvests[1] = {'id':1,'pos':[40,0,0],'seen':0}
        self.obs.update(t=91,events=[{'type':'kill','target':999}])
        self.bot.memory.update(self.obs)
        self.assertEqual(self.bot.memory.harvests,{})

    def test_safe_farming_approaches_crates_and_drones_but_rejects_roofs(self):
        self.obs['crates'] = [{'id':1000,'pos':[40,0,0],'hp':300}]
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'crate')
        self.assertGreater(action['move'][0],0)
        self.obs.update(t=.1,events=[{'type':'hit','target':1000,'amount':45}])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.memory.harvests,{})
        self.obs.update(t=.2,crates=[])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'loot')
        self.assertIn(1000,self.bot.memory.harvests)
        self.bot.memory.harvests.clear()
        self.bot.memory.crates.clear()
        self.obs.update(t=1,events=[],mechs=[enemy(50,id=100,kind='npc')])
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'npc')
        self.assertGreater(action['move'][0],0)
        self.obs.update(t=7,mechs=[],crates=[{'id':1001,'pos':[40,0,30],'hp':300}])
        self.bot.act(self.obs)
        self.assertNotEqual(self.bot.mode,'crate')
        self.assertNotEqual(self.bot.nav.goal,(40,0))
        self.assertEqual(self.bot.memory.npcs,{})

    def test_empty_observation_and_time_guards(self):
        action = self.bot.act(self.obs)
        self.assertEqual(action['fire'],[False,False])
        target = enemy()
        me = self.obs['self']
        self.assertEqual(fire_control(me,target,True,[target],self.geo,{},time.perf_counter()-1)[1],[False,False])

    def test_npc_avoidance_preserves_silent_speed(self):
        self.bot.memory.npcs = {100:enemy(30,id=100,kind='npc')}
        move = self.bot._avoid_npcs([0,0,0],[.4,0],None)
        self.assertLessEqual(math.hypot(*move),.400001)
        self.assertLess(move[0],0)
        self.assertEqual(self.bot._avoid_npcs([0,0,0],[.4,0],self.bot.memory.npcs[100]),[.4,0])

    def test_cover_cache_invalidates_when_new_threat_arrives(self):
        self.obs['self']['weapons'][1] = weapon('laser')
        calls = []
        def cover(*args, **kwargs):
            calls.append(args[1])
            return (10,10)
        self.geo.find_cover = cover
        self.bot.act(self.obs)
        self.obs['mechs'] = [enemy(),enemy(70,id=2)]
        self.obs['t'] = .1
        self.bot.act(self.obs)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(calls[-1]),2)

    def test_early_exploration_and_no_wall_freeze(self):
        action = self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'explore')
        self.assertGreater(math.hypot(*action['move']),0)
        self.assertLess(math.dist(self.bot.explore_goal,[0,0]),500)
        self.geo.walk_clear = lambda a,b:False
        self.obs['mechs'] = [enemy()]
        self.assertGreater(math.hypot(*self.bot.act(self.obs)['move']),0)
        self.obs.update(t=151,mechs=[])
        self.bot.act(self.obs)
        self.assertEqual(self.bot.mode,'hide')


if __name__ == '__main__':
    unittest.main()
