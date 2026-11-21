"""Build the navigation cache; runtime never needs to write to disk."""
from pathlib import Path
import time

from geo import Geo


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    start = time.perf_counter()
    geo = Geo(root / 'arena')
    geo.save(root / 'navgrid.npz')
    print(f'Built navgrid in {time.perf_counter()-start:.3f}s; {geo.walkable.mean():.2%} walkable')
