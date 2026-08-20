# Low-Simulation Policy Distillation

This branch improves 48/96-simulation play without modifying the frozen
v1.0.0 mainline or its public assets.

## Frozen inputs

- Student bootstrap / gate baseline: `runtime/final_iter440.net`
  - SHA-256: `66aa74b70cd2a73b1c61616df13aaa4a61073d3c5cdf10c1979084212827b2c4`
- Frozen teacher: the same iter440 network, searched with 600 simulations and
  no root noise.
- Public low-budget reference: `runtime/iter330.net`
  - SHA-256: `c4f41fcba6acd9f0956a5f64c0a80762fc21b9871b5ef429d3335ecfa47275dc`

None of these files is overwritten. Fast runs use `runtime_fast/`, a fresh
AdamW optimizer, and an empty replay buffer.

## Target semantics

Each self-play position first runs the student at 96 simulations. The action
is always selected from this behavior policy. Independently, 70% of positions
run the frozen teacher at 600 simulations; only `Sample.policy` is replaced by
the teacher visit distribution. `Sample.value` remains the real terminal
outcome of the student trajectory. Replay serialization is unchanged.

## Pilot recipe

```bash
taskset -c 0-55 ./bin/alphazero train \
  --run-dir runtime_fast/iter440_teacher96 \
  --iterations 5 --resume 0 --cache 1 \
  --init-model runtime/final_iter440.net \
  --init-best-model runtime/final_iter440.net \
  --teacher-model runtime/final_iter440.net \
  --teacher-sims 600 --teacher-target-prob .70 \
  --workers 48 --games-per-iter 40 --sims 96 \
  --train-steps 200 --batch 128 --lr .0003 --wd .0001 \
  --value-weight 1.2 --hard-fraction .45 --buffer 200000 \
  --max-moves 200 --temp-moves 6 --seed-hard-prob .3 \
  --cpuct .8 --dir-eps .25 --dir-alpha .3 --fpu 0 \
  --gate-every 5 --gate-games 40 --gate-threshold .55 \
  --save-buffer-every 5 --trunk 32 --blocks 4 --seed 4242
```

The pilot uses 40 games per iteration to obtain an early signal. A surviving
candidate continues with 80 games per iteration.

## Pilot 1 result: full-network update rejected

The five-iteration pilot completed, but the candidate lost its 96-simulation
gate to frozen iter440, 15:25. Fixed-seed L6/L7 gauntlets showed a useful but
unsafe split: L6 improved at both 48 and 96 simulations, while L7 as white at
96 fell from iter440's 20/20 to 0/20. Updating the shared trunk/value path was
therefore rejected rather than published.

The second pilot adds `--policy-head-only 1`. It freezes the trunk and value
head, keeps every BatchNorm running statistic fixed, and updates only
`policy.conv`, `policy.norm`, and `policy.fc`. The optimizer checkpoint is
mode-specific and a mode mismatch fails closed. File-level verification showed
zero changed bytes in the header/trunk/value regions; the policy region alone
changed. This follows the conservative first experiment in the research plan:
distill the deep-search policy without damaging the value estimate on which
600-simulation search depends.

## Pilot 2 result: policy-only fast model accepted

The first policy-only run used learning rate `3e-4`. Its gate improved to
21:19 but the final policy became budget-sensitive: L7 was perfect at 96 sims
and 0% at 48 sims. A second run changed only the learning rate to `1e-4` and
saved every iteration. Iteration 4 was the first robust cross-budget point and
is frozen as `runtime_fast/champion_fast_iter4.net`:

- SHA-256: `1bbd86347ee4942f8b99c9732f8afbd20c896bd5ef703eac87f11f45dedb26ad`;
- byte audit versus iter440: header/trunk/value unchanged, only policy head
  changed;
- L6/L7 at 48 sims with bounded subtree reuse, 25 games/color: L6 88%/56%,
  L7 100%/100%; iter330 scored L6 56%/64% and L7 100%/100%;
- L6/L7 at 96 fresh-tree sims: L6 96%/76%, L7 100%/100%; iter330 scored
  84%/28% and 100%/0%, iter440 scored 92%/68% and 100%/100%;
- direct 600-sim arena versus iter440: 40:0, winning both colors;
- 600-sim L6/L7 confirmation, 25 games/color: L6 100%/96%, L7 100%/100%;
  the maximum color-specific regression versus iter440 was 4 percentage
  points, within the pre-registered 5-point guard.

The accepted model improves the browser-oriented low-simulation path without
reducing the value/trunk representation used by high-budget search. Frozen
iter440 remains the mainline training artifact and v1.0.0 regression reference.

## Throughput engineering

Instantaneous `pidstat` during self-play measured about 36 fully busy cores;
the earlier 15-core number was lifetime-average `ps` output plus the end-of-
iteration straggler tail. Forty pilot games can activate at most 40 of 48
workers. Production runs should use at least 48 games (normally 80) to
amortize that tail.

An unrelated but material host leak was removed: 1,155 orphaned Node opponent
engines from aborted gauntlets held about 61 GiB summed RSS and 4.3 CPU cores.
`Subprocess` now directly execs Node with a Linux parent-death signal instead
of using `/bin/sh -c`; killing a real gauntlet was verified to kill its Node
child. Training logs now include self-play and optimizer phase durations.

## Release gates

A fast model is publishable only when all hold against fixed openings/seeds:

1. L6 and L7 at 48 simulations improve over the public iter330 model.
2. L6 and L7 at 96 simulations improve over both iter330 and iter440.
3. At 600 simulations the candidate is non-inferior to frozen iter440, with a
   maximum allowed score regression of 3 percentage points.
4. No color-specific L6/L7 regression exceeds 5 percentage points at 600.

Until then, v1.0.0, iter440, iter330, Cloudflare assets, and book assets remain
unchanged.
