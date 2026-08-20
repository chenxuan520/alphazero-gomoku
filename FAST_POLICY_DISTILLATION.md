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

## Release gates

A fast model is publishable only when all hold against fixed openings/seeds:

1. L6 and L7 at 48 simulations improve over the public iter330 model.
2. L6 and L7 at 96 simulations improve over both iter330 and iter440.
3. At 600 simulations the candidate is non-inferior to frozen iter440, with a
   maximum allowed score regression of 3 percentage points.
4. No color-specific L6/L7 regression exceeds 5 percentage points at 600.

Until then, v1.0.0, iter440, iter330, Cloudflare assets, and book assets remain
unchanged.
