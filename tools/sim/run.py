#!/usr/bin/env python3
"""Run reproducible local practice/FFA comparisons, optionally across processes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from geo import Geo
from tools.sim.engine import Simulator


def initialize():
    global HELLO, GEO
    with gzip.open(ROOT / 'check/stream.jsonl.gz','rt') as stream:
        HELLO = json.loads(next(stream))
    GEO = Geo(ROOT / 'arena')


def match(arguments):
    seed,mode,bot,mix,duration = arguments
    return Simulator(HELLO,GEO,seed,mode,bot,mix,duration).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--mode',choices=('practice','ffa'),default='practice')
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--bot',choices=('ours','example'),default='ours',help='evaluated bot in slot 0')
    parser.add_argument('--ffa-mix',choices=('ours','mixed'),default='ours',help='other FFA slots: ours, or alternate ours/example')
    parser.add_argument('--duration',type=float,default=None,help='short smoke tests only; default uses hello duration (480 s)')
    args = parser.parse_args()
    if args.matches < 1 or args.jobs < 1 or (args.duration is not None and args.duration <= 0):
        parser.error('matches, jobs and duration must be positive')
    print(f'Local approximate simulator: mode={args.mode} bot={args.bot} matches={args.matches} seed={args.seed} jobs={min(args.jobs,args.matches)}',flush=True)
    arguments = [(args.seed+i,args.mode,args.bot,args.ffa_mix,args.duration) for i in range(args.matches)]
    results = []
    if args.jobs == 1:
        initialize()
        iterator = map(match,arguments)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(args.jobs,args.matches),initializer=initialize,
                                   mp_context=multiprocessing.get_context('spawn'))
        iterator = pool.map(match,arguments)
    try:
        for result in iterator:
            results.append(result)
            timings = result['timings']
            p99 = sorted(timings)[int(.99*(len(timings)-1))] if timings else 0
            print(f"seed={result['seed']:4} place={result['place']:2} score={result['score']:2} cause={result['cause']:7} kills={result['kills']:2} cubes={result['cubes']:2} alive={result['time_alive']:6.1f}s act_p99={p99:.3f}ms act_max={max(timings,default=0):.3f}ms",flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    scores = [r['score'] for r in results]
    timings = sorted(t for r in results for t in r['timings'])
    mean = statistics.mean(scores)
    stderr = statistics.stdev(scores)/(len(scores)**.5) if len(scores) > 1 else 0
    print(f'Mean (16-place): {mean:.3f}; standard error: {stderr:.3f}')
    print(f'Death causes: {dict(sorted(Counter(r["cause"] for r in results).items()))}')
    print(f'Placements: {dict(sorted(Counter(r["place"] for r in results).items()))}')
    print(f'act() samples={len(timings)} p99={timings[int(.99*(len(timings)-1))]:.3f} ms max={max(timings):.3f} ms (wall time under {min(args.jobs,args.matches)} workers)')
    if args.bot == 'example' and args.mode == 'practice':
        print(f'Example reference=1.150; simulator delta={mean-1.15:+.3f}; calibration target=1–2 (not validation of real-match fidelity)')


if __name__ == '__main__':
    main()
