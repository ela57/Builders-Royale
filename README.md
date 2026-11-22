# Builders Royale

Survival pilot with independent legs/torso control: upgrade priorities, desired
weapon pickup/replacement, timed zone rotation, threat memory, range/strafe
combat, low-HP disengagement/regen, laser/mortar dodges, NPC avoidance/retaliation,
safe loot/drop detours, cover hiding/sound scans and stuck jump/jet recovery.
Mortars are intentionally not collected or fired (the design's v1 option).
The conformance recording is not a simulation or evidence of match strength.

## Run

Python 3.12+ with `numpy==2.1.3` (the image uses Python 3.13):

```sh
python3 -m pip install numpy==2.1.3
python3 precompute.py
python3 check/check.py -- python3 -u bot.py
docker build -t mech-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" mech-bot
python3 -m unittest discover -s check -p 'test_*.py'
python3 check/bench_geo.py
python3 check/replay_bot.py
```

`mechfight.py`, the schema, map and conformance recording/check are copied from
the kit unchanged. Docker builds `navgrid.npz`; local runs can precompute it or
compute in memory during hello. Runtime never writes the cache. The pilot prefers
`hello.map.dir`, falling back to the bundled arena. A map fingerprint prevents
using a cache from another map.

## Geometry API

`Geo(map_dir, cache=None)` loads bilinear centimetre terrain and a 2 m grid.
`ground_grid` stores mean sampled height; `solid_top` stores ground-attached prism
tops; `occluder` is their maximum. Bridges clearing the mech are walkable;
low overhangs block navigation. Height-field LOS intentionally ignores raised
prisms and trees, per the design's approximation (not exact prism raycasting).
Prisms include buildings and stadium sectors, with polygon holes excluded.
Slope blocks and prism blocks are dilated by 2.5 m plus a cell half-diagonal,
and the map boundary is blocked conservatively.

`search(start_xy, goal_xy)` returns a resumable `PathSearch`. Call
`advance(budget_ms=8)` each tick until status is `complete` or `unreachable`;
only complete paths are published in `.path`. A* retains its frontier across
ticks, even after the normal replan interval. Flood-filled connectivity labels
reject disconnected endpoints immediately. Smoothing uses supercover traversal
with exponential lookahead, also yielding to the per-tick budget.
Pending searches do not retain their owners through generator cycles. `close()`
releases a search immediately; navigation closes superseded or invalidated work
instead of leaving large frontiers for a later garbage-collection pause.
Startup collects garbage and freezes the long-lived geometry after loading;
automatic garbage collection remains enabled for new runtime objects.
`astar(start_xy, goal_xy, time_budget_ms=12)` is a one-shot convenience returning
only a complete route (or `[]`); the bot uses resumable searches instead.
`walk_clear`, `los(a_xyz, b_xyz)`,
`cover_score(cell, threat_foot_positions)` and
`find_cover(me_xyz, threat_foot_positions, zone)` support later behaviours.
Cover returns a safe hidden point or `None`; it prefers proximity and walls,
using the next circle if supplied. A* and cover have cooperative time budgets.
`navigation.py` reuses the last complete safe path while a new one is pending;
without a previous path it moves directly toward the goal, allowing physical
wall sliding while the complete route is pending. No partial route is published.
The benchmark checks 20 connected random 100–700 m routes,
reports total latency and individual 6 ms work slices separately, and validates
all segments against blocked cells. Disconnected samples are reported separately.

`brain.py` owns memory, offer/pickup choices and safe-point timing; `combat.py`
owns intercept/aim/fire and dodge geometry. `bot.py` applies priorities and uses
25 ms cooperative decision deadlines, 8 ms navigation slices and 4 ms cover
slices. Enemy memories last 5 s, crowd estimates 3 s. The actual observed aim
and weapon charge are authoritative; no mech shots are issued through blocked
LOS or at remembered-only targets. Trigger false releases a charging laser;
losing LOS therefore cancels the hold but cannot prevent a shot already charged
from discharging into intervening cover. Exact bridge raycasting, larger-mech
navigation after cubes, and real-match fidelity remain outside this geometry model.

## Local tuning

`tools/sim/` imports the actual bot and the unchanged kit example class in-process
for approximate 20 Hz practice-squad or FFA matches. It reads game numbers from
the recorded hello, supports eight worker processes, and reports placement,
death causes, kills, cubes, survival time and act timings. See
[`tools/sim/README.md`](tools/sim/README.md) for modeled rules and limitations.
The simulator is excluded from the Docker context.

```sh
python3 tools/sim/run.py --matches 24 --seed 1 --jobs 8
python3 tools/sim/run.py --matches 24 --seed 1 --jobs 8 --bot example
python3 -m unittest discover -s tools/sim -p 'test_*.py'
```

Before 150 s, a gun-only pilot with no visible safe loot explores nearby open
areas inside the safe circle instead of standing still. Navigation fallback and
combat movement are not zeroed simply because a straight segment touches a wall.
Pickups/crates are remembered for 90 s and cubes for 20 s, with stale nearby
items and confirmed kills removed. Early gun-only pilots take safe remembered
weapon detours even while watching a distant enemy, rather than forgetting loot
as soon as it leaves the view wedge.
When collision/inertia leaves a pilot outside the conservative planning grid,
it first backs out to a valid cell rather than pushing toward an unreachable
waypoint. Exploration/loot goals stay in its connected component, and zone
aim points in enclosed courtyards are replaced by nearby reachable safe points.
