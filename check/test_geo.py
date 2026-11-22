"""Small deterministic maps exercise geometry without the practice recording."""
import json
import gc
import math
import random
import signal
from pathlib import Path
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo import BudgetExceeded, DecisionTimeout, Geo
from navigation import Navigator
from bot import Bot


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'v2').mkdir()
        self.world = {'bounds':{'minX':0,'minY':0,'maxX':100,'maxY':100},
                      'terrain':{'minX':0,'minY':0,'step':1,'nx':101,'ny':101,'file':'terrain.i16'},
                      'prisms':[]}
        self.terrain = np.zeros((101,101),dtype='<i2')

    def tearDown(self):
        self.tmp.cleanup()

    def make_geo(self, cache=None):
        (self.root / 'v2/world.json').write_text(json.dumps(self.world))
        self.terrain.tofile(self.root / 'v2/terrain.i16')
        return Geo(self.root,cache)

    def prism(self, base=0, top=20, holes=None):
        return {'base':base,'top':top,'ring':[40,20,60,20,60,80,40,80],'holes':holes or []}

    def test_bilinear(self):
        self.terrain = (np.indices((101,101))[1]*100).astype('<i2')
        geo = self.make_geo()
        self.assertAlmostEqual(float(geo.ground(10.25,20.5)),10.25)
        self.assertAlmostEqual(float(geo.ground(-10,500)),0)

    def test_scalar_geometry_matches_vector_reference(self):
        self.terrain = (np.indices((101,101))[1]*31+np.indices((101,101))[0]*7).astype('<i2')
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        rng = random.Random(8)
        for _ in range(300):
            a,b = [np.array([rng.uniform(-5,105),rng.uniform(-5,105),rng.uniform(-1,30)]) for _ in range(2)]
            count = max(2,math.ceil(float(np.linalg.norm(b-a))/2)+1)
            points = a+(b-a)*np.linspace(0,1,count)[:,None]
            cells = np.floor((points[:,:2]-geo.origin)/2).astype(int)
            inside = np.all(cells >= 0) and np.all(cells[:,0] < geo.width) and np.all(cells[:,1] < geo.height)
            expected = bool(inside and np.all(points[:,2] >= geo.occluder[cells[:,1],cells[:,0]]))
            self.assertEqual(geo.los(a,b),expected)
            self.assertAlmostEqual(geo.ground(*a[:2]),geo.ground(a[:1],a[1:2])[0],places=12)

    def test_expired_geometry_budget_fails_closed(self):
        geo = self.make_geo()
        geo.deadline = 0
        for call in (lambda:geo.los((10,10,5),(90,90,5)),lambda:geo.nearest((10,10)),
                     lambda:geo.walk_clear((10,10),(90,90)),lambda:geo.ground(10,10)):
            with self.assertRaises(BudgetExceeded):
                call()
        self.assertEqual(geo.search((10,10),(90,90)).advance(8),'pending')
        geo.deadline = math.inf
        self.assertIsNone(geo.find_cover((10,10,0),[],{'center':[50,50],'radius':80},time_budget_ms=0))

    def test_cover_returns_only_completed_candidates_and_restores_deadline(self):
        geo = self.make_geo()
        clock = [0.]
        visited = []
        def score(cell, threats):
            visited.append(geo.point(cell))
            if len(visited) == 2:
                clock[0] = .2
            return len(threats)
        with patch('geo.time.perf_counter',side_effect=lambda:clock[0]),patch.object(geo,'cover_score',side_effect=score):
            result = geo.find_cover((50,50,0),[[0,0,0]],{'center':[50,50],'radius':80},time_budget_ms=100)
            self.assertEqual(result,visited[0])
            self.assertEqual(geo.deadline,math.inf)
            clock[0],visited[:] = 0.,[]
            geo.deadline = .15
            with self.assertRaises(BudgetExceeded):
                geo.find_cover((50,50,0),[[0,0,0]],{'center':[50,50],'radius':80},time_budget_ms=100)
            self.assertEqual(geo.deadline,.15)

    def test_search_slices_resume_through_expansion_and_smoothing(self):
        geo = self.make_geo()
        clock = [0.]
        walkable = geo.is_walkable
        def work(cell):
            clock[0] += .00002
            return walkable(cell)
        search = geo.search((10,10),(90,90))
        with patch('geo.time.perf_counter',side_effect=lambda:clock[0]),patch.object(geo,'is_walkable',side_effect=work):
            for _ in range(10000):
                start = clock[0]
                search.advance(.1)
                self.assertLessEqual(clock[0]-start,.000161)
                if search.status != 'pending':
                    break
        self.assertEqual(search.status,'complete')
        self.assertTrue(all(geo.walk_clear(a,b) for a,b in zip(search.path,search.path[1:])))

    def test_cancelled_search_retires_heavy_storage_without_synchronous_destruction(self):
        geo = self.make_geo()
        freed = []
        class Item:
            def __del__(self):
                freed.append(1)
        search = geo.search((10,10),(90,90))
        search._containers.append([Item() for _ in range(1000)])
        search.close()
        self.assertEqual(freed,[])
        clock = [0.]
        def tick():
            clock[0] += .00001
            return clock[0]
        with patch('geo.time.perf_counter',side_effect=tick):
            geo.collect_retired(.0001)
        self.assertGreater(len(freed),0)
        self.assertLess(len(freed),1000)
        geo.collect_retired(math.inf)
        self.assertEqual(len(freed),1000)
        self.assertFalse(geo._retired)

    def test_decision_budget_coasts_without_firing_and_does_not_hide_errors(self):
        geo = self.make_geo()
        bot = Bot({},geo=geo)
        bot.last_move = [.2,-.5]
        obs = {'tick':9,'self':{'aim':[1,.1]}}
        with patch.object(bot,'_act',side_effect=BudgetExceeded):
            action = bot.act(obs)
        self.assertEqual(action,{'tick':9,'move':[.2,-.5],'aim':[1,.1],'fire':[False,False]})
        self.assertEqual(geo.deadline,math.inf)
        with patch.object(bot,'_act',side_effect=ValueError),self.assertRaises(ValueError):
            bot.act(obs)
        self.assertEqual(geo.deadline,math.inf)

    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX wall timers required')
    def test_wall_alarm_interrupts_uncooperative_work_and_restores_signal_state(self):
        if signal.getitimer(signal.ITIMER_REAL) != (0.,0.):
            self.skipTest('External wall timer is active')
        geo = self.make_geo()
        bot = Bot({},geo=geo)
        before = signal.getsignal(signal.SIGALRM)
        def busy(obs):
            while True:
                pass
        with patch.object(bot,'_act',side_effect=busy):
            action = bot.act({'tick':9,'self':{}})
        self.assertEqual(action['fire'],[False,False])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertIs(signal.getsignal(signal.SIGALRM),before)
        self.assertEqual(geo.deadline,math.inf)

    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX wall timers required')
    def test_wall_alarm_preserves_existing_timer_and_cancels_interrupted_search(self):
        geo = self.make_geo()
        bot = Bot({},geo=geo)
        obs = {'tick':9,'self':{}}
        try:
            signal.setitimer(signal.ITIMER_REAL,2)
            with patch.object(bot,'_act',return_value={'move':[0,0]}):
                bot.act(obs)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],1)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        bot.nav.move([10,10,0],[90,90],0,budget_ms=0)
        self.assertIsNotNone(bot.nav.search)
        with patch.object(bot,'_act',side_effect=DecisionTimeout):
            self.assertEqual(bot.act(obs)['fire'],[False,False])
        self.assertIsNone(bot.nav.search)
        with patch.object(geo,'_find_cover',side_effect=DecisionTimeout),self.assertRaises(DecisionTimeout):
            geo.find_cover([10,10,0],[],{'center':[50,50],'radius':80})
        self.assertEqual(geo.deadline,math.inf)

    def test_prisms_dilation_holes_and_bridges(self):
        self.world['prisms'] = [self.prism(holes=[[46,30,54,30,54,70,46,70]])]
        geo = self.make_geo()
        self.assertFalse(geo.is_walkable(geo.cell((39,41))))
        self.assertTrue(geo.raw_walkable[geo.cell((39,41))[1],geo.cell((39,41))[0]])
        self.assertTrue(geo.is_walkable(geo.cell((51,51))))
        self.assertEqual(geo.solid_top[20,20],20)
        self.world['prisms'] = [self.prism(base=10)]
        geo = self.make_geo()
        self.assertTrue(geo.is_walkable(geo.cell((45,45))))
        self.assertTrue(geo.los((30,45,5),(70,45,5)))
        self.world['prisms'] = [self.prism(base=5)]
        geo = self.make_geo()
        self.assertFalse(geo.is_walkable(geo.cell((45,45))))
        self.assertTrue(np.isneginf(geo.solid_top[20,20]))

    def test_slope_and_border(self):
        self.terrain[:,50:] = 2000
        geo = self.make_geo()
        self.assertFalse(geo.is_walkable(geo.cell((49,49))))
        self.assertFalse(geo.is_walkable((0,0)))
        self.assertFalse(geo.walk_clear((-1,20),(20,20)))

    def test_path_and_budget(self):
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        path = geo.astar((15,51),(85,51),time_budget_ms=1000)
        self.assertEqual(path[-1],geo.point(geo.cell((85,51))))
        self.assertGreater(len(path),2)
        for a,b in zip(path,path[1:]):
            self.assertTrue(geo.walk_clear(a,b))
        search = geo.search((15,51),(85,51))
        self.assertEqual(search.advance(0),'pending')
        self.assertEqual(search.path,[])
        while search.status == 'pending':
            search.advance(.1)
        self.assertEqual(search.path[-1],path[-1])
        geo.walkable[:] = False
        geo._walk = geo.walkable.tolist()
        self.assertEqual(geo.astar((15,51),(85,51)),[])

    def test_corner_crossing(self):
        geo = self.make_geo()
        geo._walk[11][10] = False
        self.assertFalse(geo.walk_clear(geo.point((10,10)),geo.point((11,11))))

    def test_abandoned_search_releases_its_generator_without_collection(self):
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        nav = Navigator(geo)
        enabled = gc.isenabled()
        gc.disable()
        try:
            nav.move((15,51),(85,51),0,budget_ms=.01)
            self.assertIsNotNone(nav.search)
            old = weakref.ref(nav.search)
            nav.invalidate()
            self.assertIsNone(old())
            nav.move((15,51),(85,51),1,budget_ms=.01)
            old = weakref.ref(nav.search)
            nav.move((15,51),(85,85),1.1,budget_ms=0)
            self.assertIsNone(old())
            nav.invalidate()
            search = geo.search((15,51),(85,51))
            search.close()
            search.close()
            self.assertEqual(search.advance(1),'unreachable')
            self.assertEqual(search.path,[])
            search = geo.search((15,51),(85,51))
            search.advance(.01)
            old = weakref.ref(search)
            del search
            self.assertIsNone(old())
        finally:
            if enabled:
                gc.enable()

    def test_los_cover_empty_and_no_cover(self):
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        self.assertFalse(geo.los((15,51,8),(85,51,4.5)))
        self.assertTrue(geo.los((15,51,25),(85,51,25)))
        self.assertFalse(geo.los((-1,0,5),(20,20,5)))
        self.assertEqual(geo.cover_score(geo.cell((85,51)),[(15,51,0)]),1)
        zone = {'center':[50,50],'radius':60}
        cover = geo.find_cover((85,51,0),[(15,51,0)],zone,time_budget_ms=1000)
        self.assertIsNotNone(cover)
        self.assertEqual(geo.cover_score(geo.cell(cover),[(15,51,0)]),1)
        self.assertIsNotNone(geo.find_cover((85,51,0),[],zone,time_budget_ms=1000))
        self.assertIsNone(geo.find_cover((85,51,0),[],{'center':[500,500],'radius':1}))

    def test_cache_and_invalid_cache_fallback(self):
        geo = self.make_geo()
        cache = self.root / 'navgrid.npz'
        geo.save(cache)
        loaded = self.make_geo(cache)
        np.testing.assert_array_equal(geo.walkable,loaded.walkable)
        np.testing.assert_array_equal(geo.raw_walkable,loaded.raw_walkable)
        cache.write_bytes(b'not a cache')
        np.testing.assert_array_equal(geo.walkable,self.make_geo(cache).walkable)
        geo.save(cache)
        self.world['prisms'] = [self.prism()]
        self.assertFalse(self.make_geo(cache).is_walkable((25,25)))

    def test_bot_moves_on_path(self):
        self.make_geo()
        nav = Navigator(self.make_geo())
        move = nav.move([15,15,0],[85,85],0)
        self.assertGreater(sum(v*v for v in move),0)
        self.assertLessEqual(sum(v*v for v in move),1.000001)
        self.assertGreater(sum(v*v for v in nav.move([-10,-10,0],[85,85],.1)),0)

    def test_bot_follows_path_around_wall(self):
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        nav = Navigator(geo)
        pos = np.array([15.,51.])
        goal = [85,51]
        for tick in range(500):
            move = nav.move([*pos,0],goal,tick*.1)
            nxt = pos+np.array(move)*1.17
            self.assertTrue(geo.walk_clear(pos,nxt))
            pos = nxt
            if np.linalg.norm(pos-goal) < 2:
                break
        self.assertLess(np.linalg.norm(pos-goal),2)

    def test_unreachable_is_exhaustive(self):
        self.world['prisms'] = [{'base':0,'top':20,'ring':[40,0,60,0,60,100,40,100]}]
        geo = self.make_geo()
        search = geo.search((15,51),(85,51))
        while search.status == 'pending':
            search.advance(.2)
            self.assertEqual(search.path,[])
        self.assertEqual(search.status,'unreachable')

    def test_concave_route_keeps_pending_work(self):
        self.world['prisms'] = [self.prism(), {'base':0,'top':20,'ring':[10,20,60,20,60,30,10,30]}]
        geo = self.make_geo()
        nav = Navigator(geo)
        move = nav.move([31,51,0],[85,51],0,budget_ms=0)
        self.assertGreater(sum(v*v for v in move),0)
        self.assertLessEqual(sum(v*v for v in move),1.000001)
        search = nav.search
        self.assertEqual(nav.path,[])
        nav.move([31,51,0],[85,51],2,budget_ms=0)
        self.assertIs(nav.search,search)
        while search.status == 'pending':
            search.advance(.2)
        self.assertEqual(search.status,'complete')
        for a,b in zip(search.path,search.path[1:]):
            self.assertTrue(geo.walk_clear(a,b))

    def test_bot_backs_out_of_dilated_margin_before_following_route(self):
        self.world['prisms'] = [self.prism()]
        geo = self.make_geo()
        bot = Bot({'seed':1},geo=geo)
        obs = {'t':0,'self':{'id':0,'pos':[39,51,0],'vel':[0,0,0],
               'hp':1400,'max_hp':1400,'energy':100,'weapons':[], 'in_zone':True},
               'zone':{'center':[50,50],'radius':100,'next_center':[85,51],
                       'next_radius':20,'shrink_start':0,'shrink_end':50}}
        move = bot.act(obs)['move']
        self.assertLess(move[0],0)
        self.assertTrue(bot._reachable([85,51]))

    def test_zone_goal_avoids_disconnected_courtyard(self):
        self.world['prisms'] = [self.prism(holes=[[46,30,54,30,54,70,46,70]])]
        geo = self.make_geo()
        bot = Bot({'seed':1},geo=geo)
        bot.nav_cell = geo.cell([85,51])
        bot.nav_label = geo.components[bot.nav_cell[1],bot.nav_cell[0]]
        self.assertFalse(bot._reachable([51,51]))
        goal = bot._zone_goal([51,51],{'center':[51,51],'radius':60})
        self.assertTrue(bot._reachable(goal))
        self.assertLess(sum((goal[i]-51)**2 for i in (0,1)),60**2)


if __name__ == '__main__':
    unittest.main()
