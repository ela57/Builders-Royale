# Local tuning simulator

This is an approximate regression/comparison tool, **not the game server**.
Every numeric game rule is initialized from the recorded hello in
`check/stream.jsonl.gz`. It imports our `Bot` in-process, shares immutable map
geometry between agents, and constructs normal `mechfight/1` observation dicts.
The kit example class is vendored unchanged; its unconditional `mf.run(Bot)`
entrypoint alone is omitted by the in-process AST loader. No stdin monkeypatching
or edits to the example's decisions are used.

```sh
python3 tools/sim/run.py --matches 24 --seed 1 --jobs 8
python3 tools/sim/run.py --matches 24 --seed 1 --jobs 8 --bot example
python3 tools/sim/run.py --matches 1 --seed 1 --mode ffa --ffa-mix mixed
python3 tools/sim/run.py --matches 1 --seed 1 --duration 2 --jobs 1
python3 -m unittest discover -s tools/sim -p 'test_*.py'
```

The seed is incremented per match. Use the same seed range for paired comparisons.
Processes use independent seeded RNGs; act wall-clock budgets and CPU contention
can change path completion/decisions slightly, so this is not bit-reproducible
across machines or job counts. Reported act times are wall times, not isolated
1-core conformance measurements. Defaults use eight spawned worker processes.
`--duration` exists for smoke tests; calibration uses the full 480 s limit.

## Implemented

- 20 Hz movement, 10 Hz decisions; acceleration, walk/boost/energy, approximate
  jumps/jets, ground, map boundaries, blocked-cell walls with independent-axis
  sliding, approximate roof landing; torso yaw/pitch slew.
- Config gun projectiles with swept cylinder hits, terrain/wall LOS checks,
  spread, cooldown, ammo and automatic/early reload; charged laser hitscan/release,
  hold timeout, cooldown, charging half-speed; lightning cone/range/energy and
  laser shorting. Rear armor, armor upgrades, cube scaling, combat/zone regen
  suppression, damage/hit/kill/took/pickup events.
- Config zone phase durations/radii/DPS and uniform-area next centers contained
  inside their parent circle. Config-separated random spawns (no relaxation).
- Config crate/pickup counts on random walkable ground cells, crate and death
  cubes (+2), weapon drops, offer thresholds/expiry and upgrade effects.
- Near/FOV/range/LOS vision with five-second real-position tracking, local sound
  bearing noise/range/buckets, visible projectiles/beams and protocol dicts.
- Practice squad never damages itself, shares sightings, starts with one hunter,
  adds a hunter per player kill, all hunt after 300 s or at six alive. Four easy,
  four medium, four hard, three expert opponents have different aim noise,
  reaction delay and weapon willingness. They obey the same physical weapons,
  health, loot and zone rules; there are no baseline-specific physics or bonuses.
  Hunters are selected in a seeded fixed order, not using hidden player position;
  non-hunters wander locally until zone rotation is needed.
- FFA uses independent copies of our bot, optionally alternating with the kit
  example. No squad vision or friendly-fire exemption in FFA.
- Placement: same-tick deaths share `survivors_after_tick + 1`; timeout survivors
  rank by HP (ID breaks ties). Evaluation can stop after slot 0's death because
  its placement and metrics can no longer change. Otherwise ends at one alive
  or the configured time limit. `timeout` also labels an undefeated survivor.

## Known approximations / omissions

Walls use `Geo.raw_walkable` before radius dilation, so collision tests the mech
centre against 2 m cells, not exact prism-vs-cylinder geometry. The bot still
plans on its normal radius-dilated grid. Air movement, roofs, bridges and slopes
are approximate; LOS uses our height field, including its raised-prism limitation.
There is no exact canopy/trunk vision, mech push-apart, shell/mortar simulation,
lightning chaining, laser piercing, NPCs or supply-drop capture. Mortar pickups
can spawn but cannot fire. All crates/pickups are on the ground, not roof-biased.
Targets are swept at their tick-end positions, not continuous moving-cylinder
trajectories.
The local loop fails loudly on bot exceptions instead of silently coasting, and
measures (but does not enforce) wall-clock answer deadlines. Debug drawings are
discarded between calls. House loot selection is simplified and can see nearby
pickups through cover; combat target information still requires shared vision.

Calibrating the example to its known score is a useful scale check but can
overfit one policy; it does not prove equal difficulty for other strategies.
