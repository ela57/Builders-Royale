"""Terrain, conservative ground navigation and height-field visibility."""
from pathlib import Path
import heapq
import hashlib
import json
import math
import time

import numpy as np


def in_ring(x, y, ring):
    vertices = np.asarray(ring).reshape(-1, 2)
    inside = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
    ax, ay = vertices[-1]
    for bx, by in vertices:
        if by != ay:
            inside ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
        ax, ay = bx, by
    return inside


class Geo:
    cell_size = 2.0

    def __init__(self, map_dir, cache=None):
        self.map_dir = Path(map_dir)
        self.world = json.loads((self.map_dir / 'v2/world.json').read_text())
        self.terrain_info = self.world['terrain']
        t = self.terrain_info
        self.terrain = np.fromfile(self.map_dir / 'v2' / t['file'], dtype='<i2').reshape(t['ny'], t['nx']) * .01
        self.signature = hashlib.sha256((self.map_dir / 'v2/world.json').read_bytes() + self.terrain.tobytes()).hexdigest()
        b = self.world['bounds']
        self.origin = np.array([b['minX'], b['minY']])
        self.width = int((b['maxX']-b['minX']) // 2)
        self.height = int((b['maxY']-b['minY']) // 2)
        self.components = None
        if cache is not None:
            try:
                with np.load(cache, allow_pickle=False) as data:
                    if str(data['signature']) != self.signature:
                        raise ValueError('navgrid map mismatch')
                    if not np.array_equal(data['origin'], self.origin):
                        raise ValueError('navgrid origin mismatch')
                    self.ground_grid = data['ground']
                    self.solid_top = data['solid_top']
                    self.walkable = data['walkable']
                    self.raw_walkable = data['raw_walkable'] if 'raw_walkable' in data else self.walkable.copy()
                    if 'components' in data:
                        self.components = data['components']
                    if self.walkable.shape != (self.height, self.width):
                        raise ValueError('navgrid shape mismatch')
            except (OSError, ValueError, KeyError):
                self.build()
        else:
            self.build()
        self.occluder = np.maximum(self.ground_grid, self.solid_top)
        self._walk = self.walkable.tolist()
        if self.components is None:
            self._label_components()

    def _label_components(self):
        # With corner-cutting forbidden, diagonal edges cannot join 4-components.
        width,height = self.width,self.height
        walk = self.walkable.ravel().tolist()
        labels = [0]*len(walk)
        component = 0
        for start,free in enumerate(walk):
            if not free or labels[start]:
                continue
            component += 1
            labels[start] = component
            stack = [start]
            while stack:
                current = stack.pop()
                x,y = current % width,current // width
                for nxt,valid in ((current-1,x>0),(current+1,x<width-1),(current-width,y>0),(current+width,y<height-1)):
                    if valid and walk[nxt] and not labels[nxt]:
                        labels[nxt] = component
                        stack.append(nxt)
        self.components = np.array(labels,dtype='i4').reshape(height,width)

    def ground(self, x, y):
        t = self.terrain_info
        fx = np.clip((np.asarray(x)-t['minX'])/t['step'], 0, t['nx']-1)
        fy = np.clip((np.asarray(y)-t['minY'])/t['step'], 0, t['ny']-1)
        i = np.minimum(fx.astype(int), t['nx']-2)
        j = np.minimum(fy.astype(int), t['ny']-2)
        u, v = fx-i, fy-j
        a = self.terrain
        return (a[j,i]*(1-u)+a[j,i+1]*u)*(1-v)+(a[j+1,i]*(1-u)+a[j+1,i+1]*u)*v

    def build(self):
        self.components = None
        x, y = np.meshgrid(self.origin[0]+1+2*np.arange(self.width), self.origin[1]+1+2*np.arange(self.height))
        self.ground_grid = np.mean([self.ground(x+dx, y+dy) for dx,dy in [(-.5,-.5),(.5,-.5),(-.5,.5),(.5,.5)]], axis=0).astype('f4')
        self.solid_top = np.full(x.shape, -np.inf, dtype='f4')
        gy, gx = np.gradient(self.terrain, self.terrain_info['step'])
        slope = np.hypot(gx, gy)
        ti = self.terrain_info
        ii = np.clip(np.rint((x-ti['minX'])/ti['step']).astype(int), 0, ti['nx']-1)
        jj = np.clip(np.rint((y-ti['minY'])/ti['step']).astype(int), 0, ti['ny']-1)
        blocked = slope[jj,ii] > math.tan(math.radians(50))
        for prism in self.world['prisms']:
            ring = np.array(prism['ring']).reshape(-1,2)
            lo = np.maximum(np.floor((ring.min(axis=0)-self.origin)/2).astype(int), 0)
            hi = np.minimum(np.ceil((ring.max(axis=0)-self.origin)/2).astype(int), [self.width,self.height])
            sx, sy = slice(lo[0],hi[0]), slice(lo[1],hi[1])
            mask = in_ring(x[sy,sx], y[sy,sx], prism['ring'])
            for hole in prism.get('holes', []):
                mask &= ~in_ring(x[sy,sx], y[sy,sx], hole)
            g = self.ground_grid[sy,sx]
            solid = mask & (prism['base'] <= g+1.2)
            top = self.solid_top[sy,sx]
            top[solid] = np.maximum(top[solid], prism['top'])
            blocked[sy,sx] |= mask & (prism['base'] < g+9) & (prism['top'] > g+1.2)
        # Include the cell half-diagonal so the radius clears rasterized edges.
        padded = np.pad(blocked, 2, constant_values=True)
        dilated = blocked.copy()
        for dy in range(-2,3):
            for dx in range(-2,3):
                if math.hypot(dx*2,dy*2) <= 2.5+math.sqrt(2):
                    dilated |= padded[2+dy:2+dy+self.height,2+dx:2+dx+self.width]
        self.walkable = ~dilated
        self.raw_walkable = ~blocked

    def save(self, path):
        np.savez_compressed(path, signature=self.signature, origin=self.origin, ground=self.ground_grid, solid_top=self.solid_top, walkable=self.walkable, components=self.components, raw_walkable=self.raw_walkable)

    def cell(self, point):
        return (math.floor((point[0]-self.origin[0])/2), math.floor((point[1]-self.origin[1])/2))

    def point(self, cell):
        return (float(self.origin[0]+2*cell[0]+1), float(self.origin[1]+2*cell[1]+1))

    def is_walkable(self, cell):
        x,y = cell
        return 0 <= x < self.width and 0 <= y < self.height and self._walk[y][x]

    def nearest(self, point, radius=20):
        c = self.cell(point)
        if self.is_walkable(c):
            return c
        for r in range(1, radius+1):
            candidates = [(c[0]+dx,c[1]+dy) for dy in range(-r,r+1) for dx in range(-r,r+1) if max(abs(dx),abs(dy)) == r]
            candidates = [p for p in candidates if self.is_walkable(p)]
            if candidates:
                return min(candidates, key=lambda p: math.dist(self.point(p),point[:2]))
        return None

    def walk_clear(self, a, b):
        """Supercover traversal, including both sides of corner crossings."""
        x,y = self.cell(a)
        ex,ey = self.cell(b)
        dx,dy = b[0]-a[0], b[1]-a[1]
        sx,sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        tx = ((self.origin[0]+2*(x+(sx>0))-a[0])/dx) if dx else math.inf
        ty = ((self.origin[1]+2*(y+(sy>0))-a[1])/dy) if dy else math.inf
        while True:
            if not self.is_walkable((x,y)):
                return False
            if (x,y) == (ex,ey):
                return True
            if abs(tx-ty) < 1e-10:
                if not self.is_walkable((x+sx,y)) or not self.is_walkable((x,y+sy)):
                    return False
                x,y = x+sx,y+sy
                tx,ty = tx+abs(2/dx),ty+abs(2/dy)
            elif tx < ty:
                x,tx = x+sx,tx+abs(2/dx)
            else:
                y,ty = y+sy,ty+abs(2/dy)

    def astar(self, start, goal, time_budget_ms=12):
        """One-shot convenience; use search().advance() to retain unfinished work."""
        search = self.search(start,goal)
        search.advance(time_budget_ms)
        return search.path

    def los(self, a, b):
        a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        count = max(2,math.ceil(float(np.linalg.norm(b-a))/2)+1)
        points = a+(b-a)*np.linspace(0,1,count)[:,None]
        cells = np.floor((points[:,:2]-self.origin)/2).astype(int)
        if np.any(cells < 0) or np.any(cells[:,0] >= self.width) or np.any(cells[:,1] >= self.height):
            return False
        return bool(np.all(points[:,2] >= self.occluder[cells[:,1],cells[:,0]]))

    def search(self, start, goal):
        return PathSearch(self, start, goal)

    def cover_score(self, cell, threat_positions):
        x,y = self.point(cell)
        target = (x,y,float(self.ground(x,y))+4.5)
        return sum(not self.los((t[0],t[1],t[2]+8),target) for t in threat_positions)

    def find_cover(self, me, threats, zone, time_budget_ms=12):
        deadline = time.perf_counter()+time_budget_ms/1000
        center = zone.get('next_center',zone['center'])
        radius = zone.get('next_radius',zone['radius'])
        cx,cy = self.cell(me)
        candidates = [(cx+dx,cy+dy) for dy in range(-30,31,3) for dx in range(-30,31,3) if dx*dx+dy*dy <= 900]
        candidates.sort(key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2)
        best,score = None,-math.inf
        for cell in candidates:
            if time.perf_counter() >= deadline:
                break
            if not self.is_walkable(cell):
                continue
            point = self.point(cell)
            if math.dist(point,center)+2.5 > radius or self.cover_score(cell,threats) != len(threats):
                continue
            x,y = cell
            wall = any(not self.is_walkable((x+dx,y+dy)) for dx,dy in ((3,0),(-3,0),(0,3),(0,-3)))
            nearby_top = float(np.max(self.occluder[max(0,y-4):y+5,max(0,x-4):x+5]))
            height = max(0,nearby_top-float(self.ground(*point)))
            value = 8*wall+.2*min(40,height)-math.dist(point,me[:2])
            if value > score:
                best,score = point,value
        return best


class PathSearch:
    """Resumable A*: publish only a complete, smoothed route, never a pocket path."""

    def __init__(self, geo, start, goal):
        self.geo = geo
        self.status = 'pending'
        self.path = []
        self.expansions = 0
        self._work = self._run(start, goal)

    def advance(self, budget_ms=8):
        deadline = time.perf_counter()+budget_ms/1000
        while self.status == 'pending' and time.perf_counter() < deadline:
            try:
                next(self._work)
            except StopIteration as result:
                self.path = result.value or []
                self.status = 'complete' if self.path else 'unreachable'
        return self.status

    def _run(self, start, goal):
        geo = self.geo
        source, target = geo.nearest(start), geo.nearest(goal)
        if source is None or target is None:
            return []
        if geo.components[source[1],source[0]] != geo.components[target[1],target[0]]:
            return []
        h = lambda p: max(abs(p[0]-target[0]),abs(p[1]-target[1])) + (math.sqrt(2)-1)*min(abs(p[0]-target[0]),abs(p[1]-target[1]))
        queue = [(h(source),0,source)]
        costs, parents = {source:0}, {}
        while queue:
            if self.expansions % 32 == 0:
                yield
            _,cost,current = heapq.heappop(queue)
            if cost != costs[current]:
                # Stale heap entries must also yield under the budget.
                yield
                continue
            self.expansions += 1
            if current == target:
                break
            x,y = current
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt = (x+dx,y+dy)
                if not geo.is_walkable(nxt):
                    continue
                if dx and dy and (not geo.is_walkable((x+dx,y)) or not geo.is_walkable((x,y+dy))):
                    continue
                g = cost+(math.sqrt(2) if dx and dy else 1)
                if g < costs.get(nxt,math.inf):
                    costs[nxt],parents[nxt] = g,current
                    heapq.heappush(queue,(g+h(nxt),g,nxt))
        else:
            return []
        path = [target]
        while path[-1] in parents:
            path.append(parents[path[-1]])
        points = [geo.point(p) for p in reversed(path)]
        result, anchor = points[:1],0
        while anchor < len(points)-1:
            # Exponential lookahead avoids quadratic tracing on long open routes.
            stride, last = 1,anchor+1
            while anchor+stride < len(points):
                yield
                if not geo.walk_clear(points[anchor],points[anchor+stride]):
                    break
                last = anchor+stride
                stride *= 2
            upper = min(anchor+stride,len(points)-1)
            while upper-last > 1:
                mid = (last+upper)//2
                yield
                if geo.walk_clear(points[anchor],points[mid]):
                    last = mid
                else:
                    upper = mid
            yield
            if geo.walk_clear(points[anchor],points[upper]):
                last = upper
            result.append(points[last])
            anchor = last
        return result
