"""Small deterministic maps exercise geometry without the practice recording."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo import Geo
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
