"""Terrain, conservative ground navigation and height-field visibility."""
from pathlib import Path
from collections import deque
import heapq
import hashlib
import json
import math
import time

import numpy as np


class BudgetExceeded(Exception):
    pass


class DecisionTimeout(BudgetExceeded):
    pass


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
        self.deadline = math.inf
        self._retired = deque()
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
        self.check_budget()
        t = self.terrain_info
        if np.isscalar(x) and np.isscalar(y):
            fx = min(max((float(x)-t['minX'])/t['step'],0),t['nx']-1)
            fy = min(max((float(y)-t['minY'])/t['step'],0),t['ny']-1)
            i,j = min(int(fx),t['nx']-2),min(int(fy),t['ny']-2)
            u,v = fx-i,fy-j
            a = self.terrain
            return (float(a[j,i])*(1-u)+float(a[j,i+1])*u)*(1-v)+(float(a[j+1,i])*(1-u)+float(a[j+1,i+1])*u)*v
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
        return (math.floor((point[0]-float(self.origin[0]))/2), math.floor((point[1]-float(self.origin[1]))/2))

    def point(self, cell):
        return (float(self.origin[0])+2*cell[0]+1, float(self.origin[1])+2*cell[1]+1)

    def check_budget(self):
        if time.perf_counter() >= self.deadline:
            raise BudgetExceeded

    def collect_retired(self, deadline):
        while self._retired and time.perf_counter() < min(deadline,self.deadline):
            containers = self._retired[0]
            if not containers:
                self._retired.popleft()
            elif not containers[-1]:
                containers.pop()
            elif isinstance(containers[-1],dict):
                containers[-1].popitem()
            else:
                containers[-1].pop()

    def is_walkable(self, cell):
        x,y = cell
        return 0 <= x < self.width and 0 <= y < self.height and self._walk[y][x]

    def nearest(self, point, radius=20):
        work = self._nearest(point,radius)
        while True:
            self.check_budget()
            try:
                next(work)
            except StopIteration as result:
                return result.value

    def _nearest(self, point, radius=20):
        c = self.cell(point)
        if self.is_walkable(c):
            return c
        for r in range(1, radius+1):
            best,distance = None,math.inf
            for dy in range(-r,r+1):
                for dx in (range(-r,r+1) if abs(dy) == r else (-r,r)):
                    yield
                    p = (c[0]+dx,c[1]+dy)
                    if self.is_walkable(p):
                        d = math.dist(self.point(p),point[:2])
                        if d < distance:
                            best,distance = p,d
            if best is not None:
                return best
        return None

    def walk_clear(self, a, b):
        """Supercover traversal, including both sides of corner crossings."""
        work = self._walk_clear(a,b)
        while True:
            self.check_budget()
            try:
                next(work)
            except StopIteration as result:
                return result.value

    def _walk_clear(self, a, b):
        x,y = self.cell(a)
        ex,ey = self.cell(b)
        dx,dy = b[0]-a[0], b[1]-a[1]
        sx,sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        tx = ((float(self.origin[0])+2*(x+(sx>0))-a[0])/dx) if dx else math.inf
        ty = ((float(self.origin[1])+2*(y+(sy>0))-a[1])/dy) if dy else math.inf
        while True:
            yield
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
        ax,ay,az = map(float,a)
        bx,by,bz = map(float,b)
        dx,dy,dz = bx-ax,by-ay,bz-az
        count = max(1,math.ceil(math.sqrt(dx*dx+dy*dy+dz*dz)/2))
        ox,oy = map(float,self.origin)
        for i in range(count+1):
            self.check_budget()
            t = i/count
            x,y,z = (bx,by,bz) if i == count else (ax+dx*t,ay+dy*t,az+dz*t)
            cx,cy = math.floor((x-ox)/2),math.floor((y-oy)/2)
            if not (0 <= cx < self.width and 0 <= cy < self.height) or z < float(self.occluder[cy,cx]):
                return False
        return True

    def search(self, start, goal):
        return PathSearch(self, start, goal)

    def cover_score(self, cell, threat_positions):
        x,y = self.point(cell)
        target = (x,y,float(self.ground(x,y))+4.5)
        return sum(not self.los((t[0],t[1],t[2]+8),target) for t in threat_positions)

    def find_cover(self, me, threats, zone, time_budget_ms=12):
        previous = self.deadline
        self.deadline = min(previous,time.perf_counter()+time_budget_ms/1000)
        best = [None]
        try:
            self._find_cover(me,threats,zone,best)
        except DecisionTimeout:
            raise
        except BudgetExceeded:
            if time.perf_counter() >= previous:
                raise
        finally:
            self.deadline = previous
        return best[0]

    def _find_cover(self, me, threats, zone, best):
        self.check_budget()
        center = zone.get('next_center',zone['center'])
        radius = zone.get('next_radius',zone['radius'])
        cx,cy = self.cell(me)
        candidates = [(cx+dx,cy+dy) for dy in range(-30,31,3) for dx in range(-30,31,3) if dx*dx+dy*dy <= 900]
        candidates.sort(key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2)
        score = -math.inf
        for cell in candidates:
            self.check_budget()
            if not self.is_walkable(cell):
                continue
            point = self.point(cell)
            if math.dist(point,center)+2.5 > radius or self.cover_score(cell,threats) != len(threats):
                continue
            x,y = cell
            wall = any(not self.is_walkable((x+dx,y+dy)) for dx,dy in ((3,0),(-3,0),(0,3),(0,-3)))
            nearby_top = -math.inf
            for yy in range(max(0,y-4),min(self.height,y+5)):
                self.check_budget()
                for xx in range(max(0,x-4),min(self.width,x+5)):
                    nearby_top = max(nearby_top,float(self.occluder[yy,xx]))
            height = max(0,nearby_top-float(self.ground(*point)))
            value = 8*wall+.2*min(40,height)-math.dist(point,me[:2])
            if value > score:
                best[0],score = point,value


class PathSearch:
    """Resumable A*: publish only a complete, smoothed route, never a pocket path."""

    def __init__(self, geo, start, goal):
        self.geo = geo
        self.status = 'pending'
        self.path = []
        self._expansions = [0]
        self._containers = []
        self._work = self._run(geo,start,goal,self._expansions,self._containers)

    @property
    def expansions(self):
        return self._expansions[0]

    def advance(self, budget_ms=8):
        deadline = min(self.geo.deadline,time.perf_counter()+budget_ms/1000)
        self.geo.collect_retired(min(deadline,time.perf_counter()+.0002))
        while self.status == 'pending' and time.perf_counter() < deadline:
            try:
                next(self._work)
            except StopIteration as result:
                self.path = result.value or []
                self.status = 'complete' if self.path else 'unreachable'
        return self.status

    def close(self):
        self._work.close()
        if any(self._containers):
            self.geo._retired.append(self._containers)
        self._containers = []
        self.path = []
        self.status = 'unreachable'

    @staticmethod
    def _run(geo, start, goal, expansions, containers):
        source = yield from geo._nearest(start)
        target = yield from geo._nearest(goal)
        if source is None or target is None:
            return []
        if geo.components[source[1],source[0]] != geo.components[target[1],target[0]]:
            return []
        h = lambda p: max(abs(p[0]-target[0]),abs(p[1]-target[1])) + (math.sqrt(2)-1)*min(abs(p[0]-target[0]),abs(p[1]-target[1]))
        queue = [(h(source),0,source)]
        costs, parents = {source:0}, {}
        containers.extend((queue,costs,parents))
        while queue:
            yield
            _,cost,current = heapq.heappop(queue)
            if cost != costs[current]:
                # Stale heap entries must also yield under the budget.
                yield
                continue
            expansions[0] += 1
            if current == target:
                break
            x,y = current
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                yield
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
            for container in containers:
                while container:
                    yield
                    if isinstance(container,dict):
                        container.popitem()
                    else:
                        container.pop()
            return []
        path = [target]
        containers.append(path)
        while path[-1] in parents:
            yield
            path.append(parents[path[-1]])
        points = []
        containers.append(points)
        for p in reversed(path):
            yield
            points.append(geo.point(p))
        while queue:
            yield
            queue.pop()
        while parents:
            yield
            parents.popitem()
        while costs:
            yield
            costs.popitem()
        while path:
            yield
            path.pop()
        result, anchor = points[:1],0
        while anchor < len(points)-1:
            # Exponential lookahead avoids quadratic tracing on long open routes.
            stride, last = 1,anchor+1
            while anchor+stride < len(points):
                yield
                if not (yield from geo._walk_clear(points[anchor],points[anchor+stride])):
                    break
                last = anchor+stride
                stride *= 2
            upper = min(anchor+stride,len(points)-1)
            while upper-last > 1:
                mid = (last+upper)//2
                yield
                if (yield from geo._walk_clear(points[anchor],points[mid])):
                    last = mid
                else:
                    upper = mid
            yield
            if (yield from geo._walk_clear(points[anchor],points[upper])):
                last = upper
            result.append(points[last])
            anchor = last
        while points:
            yield
            points.pop()
        return result
