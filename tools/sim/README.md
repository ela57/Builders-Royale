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
python3 tools/sim/run.py --matches 32 --seed 1 --mode ffa --ffa-mix example --jobs 8 --output results.json
python3 tools/sim/run.py --matches 192 --seed 1 --jobs 8 --supply-drops --output drops.json
python3 tools/sim/run.py --matches 192 --seed 1 --jobs 8 --npcs --output npcs.json
python3 -m unittest discover -s tools/sim -p 'test_*.py'
```

The seed is incremented per match. Use the same seed range for paired comparisons.
Processes use independent seeded RNGs; act wall-clock budgets and CPU contention
can change path completion/decisions slightly, so this is not bit-reproducible
across machines or job counts. Reported act times are wall times, not isolated
1-core conformance measurements. Defaults use eight spawned worker processes.
`--duration` exists for smoke tests; calibration uses the full 480 s limit.
Supply drops and NPCs are opt-in, preserving historical calibration by default.
Compare policies with identical feature flags and seed ranges; do not compare a
new-feature run with a baseline lacking that feature. `--trace-slow` adds method
timings to decisions exceeding 30 ms. Slow-decision records always include CPU,
wall and per-generation GC time, so collection pauses and descheduling can be
distinguished from bot computation. Profiling itself adds some overhead.

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
- Optional supply drops: configured calls/fall delays, globally visible protocol
  observations, open landing sites in the next circle, landing sounds, exclusive
  capture with contest freeze/empty-or-new-holder reset, cubes, full repair/energy
  and upgrade offers. Idle nearby house bots compete for drops. Captures are
  automatic for all agents standing in a ring, including kit examples.
- Optional NPCs: configured count/stats, wandering and aggro/chase/home leashes,
  retaliation, spread/projectile speed/cooldown/damage, ordinary vision and memory,
  weapon collisions, two-cube rewards, kill events and no zone damage. Houses
  defend against nearby drones. NPC kills do not increase mech kill counts or
  the practice hunter count. NPC hits/captures/kills have separate telemetry.
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
lightning chaining or laser piercing. Mortar pickups
can spawn but cannot fire. All crates/pickups are on the ground, not roof-biased.
Targets are swept at their tick-end positions, not continuous moving-cylinder
trajectories.
NPC walking uses simple independent-axis sliding without path planning or
acceleration. NPC bullets do not hit other drones. Supply-ring capture uses a
4 m foot-height tolerance; it does not model a falling crate's physical volume.
Roof camping is not a credible comparison: planning is ground-only, movement
uses cell-height approximations, and roof/crate accessibility and bridge LOS
are not modeled faithfully enough to support that policy.
The local loop fails loudly on bot exceptions instead of silently coasting, and
measures (but does not enforce) wall-clock answer deadlines. Debug drawings are
discarded between calls. House loot selection is simplified and can see nearby
pickups through cover; combat target information still requires shared vision.

Calibrating the example to its known score is a useful scale check but can
overfit one policy; it does not prove equal difficulty for other strategies.

## Combat telemetry

`--output results.json` retains per-match actual shots, mech/crate hits by weapon,
first laser/lightning acquisition times, fight episodes and starting HP, recent
shooters at death, mode seconds and timing samples. The CLI prints aggregate
hit rates, acquisition rates/times, fight outcomes, final-fight starting HP,
shooters and mode percentages. `--ffa-mix example` evaluates slot 0 against 15
unchanged kit examples.

Fights merge engage/dodge and actual dealt/received mech damage within eight
seconds. Ending contact after a kill counts as won, otherwise disengaged; death
in an active episode is lost. Recent shooters means unique attackers damaging
us or firing within 0.2 rad of us, in range, during the final two seconds. This
is an evaluator-only intent estimate, never information given to the bot.
