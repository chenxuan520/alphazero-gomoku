# Kimi handoff: iter440-evolved-64x8

Updated: 2026-08-21

This document is the authoritative handoff for the unfinished attempt to
clearly surpass the frozen iter440 AlphaZero Gomoku model.

## 1. User requirement

- Standard 15x15 freestyle Gomoku, no forbidden moves.
- Continue with a **pure AlphaZero self-play loop**:
  - behavior moves come from the current 64x8 model's own MCTS;
  - policy target `pi` comes from that same current model's own MCTS visits;
  - value target `z` comes from that self-play game's final result.
- Do **not** use a frozen teacher to replace policy targets.
- Allowed resources: CPUs 0-55 and roughly 200GB RAM.
- Frozen iter440 and all v1.x production assets must remain unchanged until a
  candidate clearly passes the multi-budget gates.
- Target is a material improvement over iter440, not merely a lower loss or a
  noisy 20-game promotion.

## 2. Repository state

Repository:

```text
/data00/home/lingchen.judy/self/alphazero-gomoku
https://github.com/chenxuan520/alphazero-gomoku
```

Current branch:

```text
iter440-evolved-64x8
```

Committed foundation:

```text
4792f8a feat: add AlphaZero v2 batching and Net2Net expansion foundation
```

The commit contains:

- opt-in cross-game dynamic batching;
- exact-preserving 32x4 -> 64x8 model expansion;
- configurable trainer thread count;
- serialized evaluator snapshots before concurrent self-play/duel/random/
  gauntlet workers;
- expansion CLI;
- concurrency, parity, checkpoint and learnability tests.

Current uncommitted work:

```text
D  .agents/plans/v2-breakthrough.md
?? .agents/plans/iter440-evolved-64x8.md
M  src/main.cpp
M  src/train/trainer.cpp
M  src/train/trainer.h
M  test/test_main.cpp
?? KIMI_HANDOFF.md
```

The source changes add `--init-buffer FILE`, allowing a fresh optimizer and a
new expanded model to import the old pure-self-play replay without resuming the
old optimizer or iteration number. The plan-file change is a rename plus an
updated pure-self-play direction.

Latest verification after these uncommitted changes:

```text
./bin/test_az
8266 checks, 0 failed
```

Earlier foundation also passed ThreadSanitizer with 8256 checks and received an
independent code-reviewer `APPROVED` result. The new init-buffer test is covered
by the normal suite, but the exact current diff has not yet been committed or
pushed.

## 3. Frozen baseline artifacts

Never overwrite these:

```text
runtime/final_iter440.net
SHA256 66aa74b70cd2a73b1c61616df13aaa4a61073d3c5cdf10c1979084212827b2c4

runtime/checkpoint.buffer.440.bin
SHA256 3ec7ddd78de9b884c06001b45441419049255c87d203cffe69281fb75d9ebfae
size   ~860 MiB / 200,000 pure-self-play samples
```

Expanded initial model:

```text
runtime_v2/iter440-expanded-64x8.net
SHA256 82995aa7258a64655cabe5bb82ba1ce95f87512eb413ce0d0b92cd959f22ed6b
size   2,856,140 bytes
config trunk=64, residual_blocks=8, parameters=711,597
```

Expansion guarantees already verified:

- five real positions had exactly zero policy-logit and value difference from
  iter440;
- old channels/blocks retain the iter440 function;
- added output-channel features are independently randomized;
- links from new inputs into old outputs start at zero;
- new residual blocks use random conv1 + zero conv2, so they start as exact
  identities;
- first backward reaches policy/new conv2; after a tiny conv2 update, the next
  backward reaches new conv1;
- expanded checkpoint save/resume works.

## 4. Experiments that were rejected

Do not resume either run as the mainline.

### A. Full-network teacher-target pilot

Directory:

```text
runtime_v2/large64x8_pilot
```

Recipe: student behavior 200 sims, 50% stored policy targets replaced by a
frozen iter440 600-sim teacher, full 64x8 training, lr=3e-4, 300 steps/iter.

Results:

```text
iter1 vs expanded iter440 baseline at 48 sims: 14:26
iter5 vs expanded iter440 baseline at 48 sims:  5:35
iter5 internal gate:                               8:32
```

Diagnosis: only 6,562 replay samples, but 711k parameters were updated for
1,500 optimizer steps. The network catastrophically drifted. This failure does
not show that pure 64x8 AlphaZero is bad; it shows that this small-replay,
teacher-target, high-update recipe was wrong.

### B. Policy-head-only teacher-target warmup

Directory:

```text
runtime_v2/evolved64x8_policy_warmup
```

Recipe: frozen trunk/value/BN, policy head only, lr=1e-4, 100 steps/iter,
100% policy targets from frozen iter440 at 600 sims.

Stopped after five training iterations before gate completion because the user
rejected teacher distillation. It is not a pure AlphaZero continuation.

Durability status:

```text
latest.current = 4 0 1
checkpoint.latest.5.* exists, but iteration 5 gate was not completed/published
```

Do not interpret this directory as a valid finished checkpoint.

## 5. Engineering findings

### Dynamic batching

The opt-in dynamic batch service is numerically correct, thread-safe, and
tested, but it is slower with the current scalar handwritten convolution:

| network | one central batch48 | 48 worker-local nets |
|---|---:|---:|
| 32x4 | 249.9 eval/s | 1305.9 eval/s |
| 64x8 | 55.7 eval/s | 199.8 eval/s |

Actual 8-game/100-sim self-play:

```text
worker-local: 63.3s
central batch avg batch=6.49: 121.6s
```

Therefore leave `--batch-inference 0` for the pure run unless the convolution
kernel itself is replaced/optimized. The batching code is an experimental,
default-off capability, not the selected v1.2 training path.

### Trainer threads

64x8, identical replay/optimizer, 30 train steps, batch128:

| train threads | elapsed |
|---:|---:|
| 4 | 514.9s |
| 8 | 318.0s |
| 16 | 206.7s |
| 24 | 207.8s |

Use `--train-threads 16`.

### Existing independent service

The only running AlphaZero process at handoff is the old web trial service:

```text
PID 2500084
./bin/alphazero serve --model runtime/web_trial_iter320.net --port 8764 \
  --sims 120 --threads 8
```

There is no training process running. Do not confuse this service with a
trainer.

## 6. Pure AlphaZero continuation route

The intended next run is:

```text
iter440 weights
  -> exact 64x8 expansion
  -> fresh optimizer
  -> import iter440's 200k pure-self-play replay
  -> current 64x8 model performs its own 600-sim self-play
  -> pi = current 64x8 MCTS visit distribution
  -> z = current self-play terminal outcome
  -> full-network low-learning-rate updates
  -> gate against frozen expanded iter440 baseline
```

No teacher model, no teacher target probability, no policy-head-only mode.

Candidate command (not yet launched; review/tune before using):

```bash
mkdir -p runtime_v2/evolved64x8_pure
taskset -c 0-55 ./bin/alphazero train \
  --run-dir runtime_v2/evolved64x8_pure \
  --iterations 30 --resume 0 --cache 1 \
  --init-model runtime_v2/iter440-expanded-64x8.net \
  --init-best-model runtime_v2/iter440-expanded-64x8.net \
  --init-buffer runtime/checkpoint.buffer.440.bin \
  --workers 48 --games-per-iter 80 --sims 600 \
  --train-steps 100 --batch 128 --train-threads 16 \
  --lr .0001 --wd .0001 --value-weight 2 --hard-fraction .3 \
  --buffer 200000 --max-moves 200 --temp-moves 6 \
  --seed-hard-prob .3 --cpuct .8 --dir-eps .25 --dir-alpha .3 --fpu 0 \
  --gate-every 5 --gate-games 40 --gate-threshold .55 \
  --save-buffer-every 5 --trunk 64 --blocks 8 --seed 62440
```

This is intentionally conservative: full old replay, low LR, and only 100
steps per iteration. Do not increase update ratio until the first 5/10-iter
gates and snapshot evaluations are known.

Expected runtime from measured 64x8 throughput:

- self-play roughly 45-90 min/iteration (600 sims, 80 games);
- train roughly 12 min/100 steps at 16 threads;
- first 10-iteration direction signal: approximately 12-20 hours;
- credible 30-50 iteration candidate plus 48/96/600 evaluation: 3-7 days.

## 7. Acceptance gates

Do not publish merely because an internal 40-game gate passes.

Required before replacing stable iter440:

1. Paired/color-balanced deterministic evaluation at 48 sims shows a material
   advantage over iter440, not a statistical coin flip.
2. 96-sim evaluation also beats iter440.
3. 600-sim direct arena is non-inferior and preferably >55%.
4. L6 and L7, both colors, do not regress; use enough games to avoid the old
   20-game gate noise.
5. Repeat across multiple fixed seed sets/openings.
6. Keep `channels/stable.json` and all v1.x production assets unchanged until
   every gate passes.

Previous professional 48-sim round robin remains the baseline:

```text
iter440 ranked #1: 210/350 = 60.0%
iter440 beat fast iter4 31:19
iter440 beat iter360 36:14
```

## 8. Immediate handoff checklist

1. Review the current uncommitted `--init-buffer` change and this document.
2. Commit or amend the plan-file rename intentionally; do not leave both plan
   filenames in history.
3. Re-run `./bin/test_az` (last result: 8266 checks, 0 failures).
4. Optionally rerun ThreadSanitizer after the init-buffer change.
5. Start only the pure command after reviewing LR/update ratio.
6. Monitor exact trainer PID, train.log, pointer, memory/disk and error strings.
7. Freeze every 5-iteration snapshot for external evaluation.
8. Never publish a candidate before the acceptance gates above.

## 9. Repositories and production channels

- Source: https://github.com/chenxuan520/alphazero-gomoku
- Model assets: https://github.com/chenxuan520/deeplearning-model
- Stable manifest: https://azgomoku.011203.xyz/channels/stable.json
- Fast manifest: https://azgomoku.011203.xyz/channels/fast.json
- Deep manifest: https://azgomoku.011203.xyz/channels/deep.json

At handoff, stable/deep point to iter440 and fast points to the policy-only
fast iter4 experiment. None of these were changed by the unfinished 64x8 work.
