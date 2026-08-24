# web_engine — experimental worker-parallel browser engine

**Status: EXPERIMENTAL. Not deployed anywhere. Production engine remains
`alphazero-gomoku-3412a43b.js` on azgomoku.011203.xyz.**

Origin: hand-derived from the deployed browser engine (2026-08-24) to host the
64x8 evolved model (`checkpoint.latest.30.net`) whose in-browser 48-sim move
time measured 33.5s with the stock engine — unusable.

## What changed vs stock

1. Conv fast path (~1.38×): BN scale folded into packed weights at load
   (idempotent), interior pixels skip bounds checks. Verified against stock
   forward on both 32x4 and 64x8 nets: max policy-logit diff 1.7e-5.
2. Worker-parallel MCTS with virtual-loss batching:
   `createSearchPool(modelBytes, workerCount, makeWorker)` +
   `searchPooled(pool, state, options)`; API-compatible with `search`
   (same options, same return shape, same `SearchSession` reuse semantics).
   Browser workers wrap cross-origin engine via fetch+Blob+importScripts;
   Node adapter in `pool_worker_node.js` for tests.

## Verification (all in Node 22 + headless Chrome)

- Tree invariants: `root.n == sims`, `Σ root-edge visits == sims` for
  W=2/4/8 × sims 24/48; session reuse: root.n = inherited + sims exactly.
- Duplicate-expansion race fixed via in-flight node guard; root pre-expanded
  synchronously (mirrors stock search).
- Parity-by-construction: 12 mid-game positions, 48 sims — chosen-action
  agreement 11/12, top-5 visit overlap 88% (residual = legit VL batch noise).

## Measured speed, 64x8 @ 48 sims, per AI move

| engine | this box's headless Chrome | Node 22 |
|---|---:|---:|
| stock | 33.5s | 14.7s |
| pooled W=4 | 7.2–8.3s (4.4×) | 7.1s (2.1×) |
| pooled W=8 | 6.4s (5.2×) | 5.6s (2.6×) |

96/120 sims scale better (W12 reaches ~3.2×). Ratio on real laptops should
improve further (faster cores than this shared host).

## Files

- `engine.js` — drop-in replacement candidate, one file, page+worker dual mode
- `test_pooled.js` — invariants/session/speed harness (`node test_pooled.js`)
- `bench.js` — stock-vs-opt forward latency/exactness (`node bench.js [iters]`)
- `pool_worker_node.js` — node:worker_threads adapter used by tests

Tests expect models at `../models/` relative to scripts; symlink or copy
`runtime_v2/evolved64x8_pure/checkpoint.latest.30.net` as `latest30.net` and
`runtime/final_iter440.net` as `iter440.net`.

## Deploy gates (unmet by user decision 2026-08-24)

- User chose "先不发布": manifests/Worker assets/labs.js wiring intentionally
  untouched. Publishing later requires: copy this engine to the deploy repo
  public/ with content-hash name, update channels/*.json, wire labs.js pool
  creation (workers=min(7, hardwareConcurrency-1)), `wrangler deploy`.
