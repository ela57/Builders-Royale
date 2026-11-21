"""Reproducible host benchmark and independent dense path safety checks."""
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geo import Geo


def main():
    root = Path(__file__).resolve().parents[1]
    start = time.perf_counter()
    geo = Geo(root / 'arena')
    print(f'Navgrid build time: {time.perf_counter()-start:.6f} s')
    print(f'Walkable: {100*geo.walkable.mean():.3f}% ({geo.width} x {geo.height})')
    geo.save(root / 'navgrid.npz')
    start = time.perf_counter()
    with np.load(root / 'navgrid.npz') as data:
        arrays = {k:data[k] for k in data.files}
    print(f'navgrid.npz load time: {(time.perf_counter()-start)*1000:.3f} ms')
    start = time.perf_counter()
    geo = Geo(root / 'arena',root / 'navgrid.npz')
    print(f'Full Geo cached load time: {(time.perf_counter()-start)*1000:.3f} ms')
    cells = np.argwhere(geo.walkable)
    rng = random.Random(2026)
    pairs = []
    disconnected = []
    while len(pairs) < 20:
        a,b = [geo.point(tuple(reversed(cells[rng.randrange(len(cells))]))) for _ in range(2)]
        if 100 <= np.linalg.norm(np.subtract(a,b)) <= 700:
            ac,bc = geo.cell(a),geo.cell(b)
            if geo.components[ac[1],ac[0]] != geo.components[bc[1],bc[0]]:
                disconnected.append((a,b))
            else:
                pairs.append((a,b))
    for a,b in disconnected:
        search = geo.search(a,b)
        assert search.advance(8) == 'unreachable'
        print(f'Disconnected pair: {a} -> {b}; separate flood-fill components, rejected without expansion')
    complete = 0
    for i,(a,b) in enumerate(pairs,1):
        search = geo.search(a,b)
        slices = []
        while search.status == 'pending':
            start = time.perf_counter()
            search.advance(6)
            slices.append((time.perf_counter()-start)*1000)
        path = search.path
        complete += search.status == 'complete'
        for p,q in zip(path,path[1:]):
            samples = np.linspace(p,q,max(2,int(np.linalg.norm(np.subtract(q,p))/.1)+2))
            indices = np.floor((samples-geo.origin)/2).astype(int)
            assert np.all(geo.walkable[indices[:,1],indices[:,0]])
            assert geo.walk_clear(p,q)
        if path:
            assert path[-1] == b
        print(f'A* {i:02}: distance={np.linalg.norm(np.subtract(a,b)):.1f} m total={sum(slices):.3f} ms max_tick={max(slices):.3f} ms ticks={len(slices)} waypoints={len(path)} {search.status}')
    print(f'Path safety: all returned paths avoid blocked cells; {complete}/20 complete, {20-complete}/20 unreachable (exhaustive search)')
    start = time.perf_counter()
    for i in range(1000):
        a,b = pairs[i % 20]
        geo.los((*a,float(geo.ground(*a))+8),(*b,float(geo.ground(*b))+4.5))
    print(f'1000 LOS calls total: {(time.perf_counter()-start)*1000:.3f} ms')


if __name__ == '__main__':
    main()
