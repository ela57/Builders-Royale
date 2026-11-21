"""Inspect decisions against the immutable recording, not a simulation."""
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bot import Bot


def main():
    root = Path(__file__).resolve().parents[1]
    elapsed = []
    with gzip.open(root / 'check/stream.jsonl.gz','rt') as stream:
        hello = json.loads(next(stream))
        hello['map']['dir'] = str(root / 'arena')
        bot = Bot(hello)
        for message in map(json.loads,stream):
            if message['type'] != 'obs':
                continue
            start = time.perf_counter()
            action = bot.act(message)
            elapsed.append((time.perf_counter()-start)*1000)
            if (len(elapsed)-1) % 20 == 0:
                print(f"obs={len(elapsed):03} t={message['t']:5.1f} mode={bot.mode:10} target={str(bot.target_id):>4} fire={action['fire']} boost={action['boost']} move={[round(v,2) for v in action['move']]}")
    elapsed.sort()
    print(f'Replay decisions: {len(elapsed)}; p99={elapsed[int(.99*(len(elapsed)-1))]:.3f} ms max={max(elapsed):.3f} ms')


if __name__ == '__main__':
    main()
