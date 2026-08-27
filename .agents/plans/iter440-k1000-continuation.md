# AlphaZero Gomoku — iter440 1000-sim continuation (k1000)

## Hypothesis

All post-iter440 work so far fine-tuned the 64x8 expansion. Both 600-sim and
900-sim recipes plateaued at ~55.8-56.4% vs iter440 @48 sims. The untried
combination: continue the **original 32x4 champion** (never plateaus-proven;
mainline was manually stopped while still climbing) with **higher-quality
self-play** (1000 sims/move). Bottleneck hypothesis is replay information
content (pi-target quality), so sims is the knob.

## Design

- init: `runtime/final_iter440.net` + `runtime/checkpoint.buffer.440.bin`
  (200k pure self-play replay), fresh optimizer. No teacher anywhere.
- recipe: 48 workers / 80 games / **1000 sims**, 200 steps/iter batch 128,
  lr 3e-4 (between mainline 1e-3 climb regime and our conservative 1e-4),
  wd 1e-4, value-weight 2, hard-fraction .3, temp-moves 6, cpuct .8,
  dir .25/.3, seed-hard .3 — mainline-aligned except sims & lr.
- internal gate: every 5 iters, 40 games vs **frozen iter440 itself**
  (init-best = final_iter440.net), direct and interpretable.
- run dir: `runtime_1000sim/`; PID 1174326, cores 0-55, launched 2026-08-27.

## Acceptance / stop rule

- Watch external 48-sim equivalence: vault the latest candidate every 5 iters
  (`runtime_1000sim/checkpoint.latest.N.net` -> candidates vault before
  retention rotates it away!) and run 120-game external arena vs
  `runtime/final_iter440.net` @48 sims at iter5/10/15/20.
- **Positive**: external @48 >= 58% sustained at two consecutive checkpoints
  (i.e., > +2pp over latest.30's 56.4%) AND @600 not worse than latest.30's
  65% -> this becomes the new strength champion; reconsider stable.
- **Negative (stop)**: iter20 external @48 still <= 56.4% (no material gain)
  -> the 1000-sim lever is exhausted too; terminate the whole
  pure-fine-tuning line permanently.
