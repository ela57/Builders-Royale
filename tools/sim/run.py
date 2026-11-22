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
from tools.sim.timing import trace_calls


def initialize(trace=False):
    global HELLO, GEO
    with gzip.open(ROOT / 'check/stream.jsonl.gz','rt') as stream:
        HELLO = json.loads(next(stream))
    GEO = Geo(ROOT / 'arena')
    if trace:
        trace_calls()


def match(arguments):
    seed,mode,bot,mix,duration,drops,npcs = arguments
    return Simulator(HELLO,GEO,seed,mode,bot,mix,duration,drops,npcs).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--mode',choices=('practice','ffa'),default='practice')
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--bot',choices=('ours','example'),default='ours',help='evaluated bot in slot 0')
    parser.add_argument('--ffa-mix',choices=('ours','mixed','example'),default='ours',help='other FFA slots: ours, mixed, or all kit examples')
    parser.add_argument('--output',type=Path,help='save per-match metrics and timing samples as JSON')
    parser.add_argument('--duration',type=float,default=None,help='short smoke tests only; default uses hello duration (480 s)')
    parser.add_argument('--supply-drops',action='store_true',help='enable protocol supply-drop calls, landing and capture')
    parser.add_argument('--npcs',action='store_true',help='enable protocol leashed NPC drones')
    parser.add_argument('--trace-slow',action='store_true',help='profile geometry/navigation methods for slow-decision diagnosis')
    args = parser.parse_args()
    if args.matches < 1 or args.jobs < 1 or (args.duration is not None and args.duration <= 0):
        parser.error('matches, jobs and duration must be positive')
    print(f'Local approximate simulator: mode={args.mode} bot={args.bot} matches={args.matches} seed={args.seed} jobs={min(args.jobs,args.matches)}',flush=True)
    arguments = [(args.seed+i,args.mode,args.bot,args.ffa_mix,args.duration,args.supply_drops,args.npcs) for i in range(args.matches)]
    results = []
    if args.jobs == 1:
        initialize(args.trace_slow)
        iterator = map(match,arguments)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(args.jobs,args.matches),initializer=initialize,initargs=(args.trace_slow,),
                                   mp_context=multiprocessing.get_context('spawn'))
        iterator = pool.map(match,arguments)
    try:
        for result in iterator:
            results.append(result)
            timings = result['timings']
            p99 = sorted(timings)[int(.99*(len(timings)-1))] if timings else 0
            print(f"seed={result['seed']:4} place={result['place']:2} score={result['score']:2} cause={result['cause']:7} kills={result['kills']:2} cubes={result['cubes']:2} alive={result['time_alive']:6.1f}s act_p99={p99:.3f}ms act_max={max(timings,default=0):.3f}ms",flush=True)
            metrics = result['telemetry']
            print(f"  shots={metrics['shots']} hits={metrics['hits']} first_held={metrics['first_held']} fights={len(metrics['fights'])} shooters_at_death={metrics['shooters_at_death']} final_fight_hp={metrics['final_fight_start_hp']}",flush=True)
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
    for kind in ('gun','laser','lightning'):
        shots = sum(r['telemetry']['shots'].get(kind,0) for r in results)
        hits = sum(r['telemetry']['hits'].get(kind,0) for r in results)
        acquired = [r['telemetry']['first_held'][kind] for r in results if kind in r['telemetry']['first_held'] and r['telemetry']['first_held'][kind] is not None]
        print(f'{kind}: hits/shots={hits}/{shots} ({100*hits/max(1,shots):.1f}%); first held={statistics.mean(acquired):.1f}s in {len(acquired)}/{len(results)}' if acquired else f'{kind}: hits/shots={hits}/{shots} ({100*hits/max(1,shots):.1f}%); never held as loot')
    modes,outcomes,shooters = Counter(),Counter(),Counter()
    for result in results:
        metrics = result['telemetry']
        modes.update(metrics['modes'])
        outcomes.update(f['outcome'] for f in metrics['fights'])
        shooters[metrics['shooters_at_death']] += 1
    print(f'Fight outcomes: {dict(outcomes)}; shooters within 2s of death: {dict(shooters)}')
    print('Mode seconds: '+', '.join(f'{mode}={seconds:.1f} ({100*seconds/max(1,sum(modes.values())):.1f}%)' for mode,seconds in modes.most_common()))
    final_hp = [r['telemetry']['final_fight_start_hp'] for r in results if r['telemetry']['final_fight_start_hp'] is not None]
    print(f'Final fight starting HP mean={statistics.mean(final_hp):.1f}' if final_hp else 'No final fights')
    slow = [{'seed':r['seed'],**decision} for r in results for decision in r['slow_decisions']]
    print(f'Slow decisions (>=30 ms): {len(slow)}; '+json.dumps(slow[:20]))
    print(f"Our supply-drop captures: {sum(r['world']['drop_captures'].get(0,0) for r in results)}; NPC kills: {sum(r['world']['npc_kills'].get(0,0) for r in results)}")
    npc_hits = Counter()
    for result in results:
        npc_hits.update(result['telemetry']['npc_hits'])
    print(f'Our NPC hits by weapon (separate from mech hits): {dict(npc_hits)}')
    if args.output:
        args.output.write_text(json.dumps({'arguments':vars(args)|{'output':str(args.output)},'results':results},indent=2))
    if args.bot == 'example' and args.mode == 'practice':
        print(f'Example reference=1.150; simulator delta={mean-1.15:+.3f}; calibration target=1–2 (not validation of real-match fidelity)')


if __name__ == '__main__':
    main()
