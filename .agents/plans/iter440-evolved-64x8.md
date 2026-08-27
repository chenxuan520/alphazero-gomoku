# AlphaZero Gomoku iter440-evolved-64x8

## Goal

Expand frozen iter440 from 32x4 to 64x8 without changing its initial function,
then continue the pure AlphaZero self-play loop (behavior, policy targets and
value targets all come from the current 64x8 network's own MCTS and games) until
the evolved model clearly exceeds iter440. No frozen teacher model or teacher
targets. Preserve all existing stable artifacts. Use CPUs 0-55, up to 200GB
RAM, and an isolated experimental runtime.

## Acceptance

- iter440-evolved-64x8 beats iter440 materially at 48 and 96 simulations under paired,
  color-balanced, deterministic-seed evaluation.
- v2 is non-inferior at 600 simulations and clears L6/L7 both colors.
- No stable channel or frozen iter440 model is changed before all gates pass.

## Implementation stages

1. Benchmark cross-game dynamic batching without changing historical defaults.
2. Add exact-preserving 32x4 -> 64x8 Net2Net expansion.
3. Review concurrency, shutdown, checkpoint bootstrap and learnability tests.
4. Run isolated evolved-model pilots; monitor throughput, losses, gates and resources.
5. Add same-tree virtual-loss batching only if Phase A/kernel results justify it.
6. Evaluate 48/96/600 and publish as v1.2.0 only after acceptance.

## Current result

- **CLOSED 2026-08-27.** Final read: two recipes (600-sim continuation and
  900-sim continuation) converge to the same external level (~55.8-56.4% vs
  frozen iter440 @48 sims; ~60-65% @600), with no gate promotions late in
  either run (600-line: none after iter20; 900-line: none after iter2, gates
  0.550/0.400/0.425/0.500). User verdict: the margin does not justify
  replacing production at the common 48-sim budget; training fully stopped.
  Champion artifact: `runtime_v2/candidates/evolved64x8-latest30-28be769f.net`
  (SHA256-pinned, read-only). Production stable/deep intentionally unchanged.
- Parallel browser engine v2 (worker virtual-loss batching + fast conv)
  shipped independently of the model decision: GitHub Pages labs.js live,
  Cloudflare Worker assets deployed (version d7a30e53, commit
  deeplearning-model@42625ff). iter440 @48 sims measured at 1.8-2.0s/move on
  this host (was 3.7s).
- **COMPLETE — all acceptance gates passed** (unchanged from frozen record). Candidate
  `runtime_v2/candidates/evolved64x8-latest30-28be769f.net`
  (SHA256 `28be769f...f974a2`; the in-run copy was later rotated away by
  trainer retention, vault is canonical) beats frozen iter440 materially: 48 sims
  203:157 over 360 games across 3 seeds (56.4%, CI excludes 50%); 96 sims
  107:93; 600 sims 26:14 (65.0%); gauntlet L6/L7 both colors 87.5% vs 50%
  (white-side weakness of iter440 fully repaired). Full details in
  `EVOLVED64X8_ACCEPTANCE.md`. Production channels/assets untouched; publish
  decision left to the user.

- Pure run completed 2026-08-24 (30 iterations, ~61h). Internal 600-sim gates:
  0.550, 0.550, 0.600 (promoted), 0.625 (promoted), 0.475, 0.425.
  Evaluation caveat found: at temp 0 deterministic seeds, near-equal nets
  collapse to black-wins-always; acceptance uses temp-moves 6 variety, which
  matches the internal gate flavour and discriminates strength correctly.

- Two teacher-target pilots were rejected after user review: they are expert
  distillation rather than the requested pure current-network AlphaZero loop.
  The active route is now: exact iter440->64x8 expansion, fresh optimizer,
  import iter440's 200k pure self-play replay, then generate every new policy
  target with the 64x8 network's own 600-simulation MCTS. No teacher network.

- Dynamic batch inference is opt-in. It is numerically equivalent and tested,
  but on the current scalar CPU kernels it is slower than 48 worker-local nets
  (64x8: 55.7 vs 199.8 eval/s), so it is not the v2 default path.
- Net2Net expansion creates a 64x8, 711,597-parameter network from iter440.
  Five real positions have exact-zero logit/value differences. Added output
  channels retain independent random features; only links into existing
  outputs are zero. Added blocks use random conv1 and zero conv2, preserving
  the source function while first-step conv2 and second-step conv1 gradients
  prove the added capacity is learnable.
- Evaluator snapshots are created serially in self-play, duel, random eval and
  gauntlet, eliminating the historical packed-weight flag race.
- Normal and ThreadSanitizer suites: 8,256 checks, zero failures. Independent
  sign-off: APPROVED.
- Formal pilot: iter440 expanded to 64x8, 200-sim student behavior, 50% of
  stored policies relabeled by frozen iter440 at 600 sims, 48 games/iteration.
  Iteration 1: self-play 1,783.5s, 1,616 samples, 52.2% teacher targets;
  policy/value loss 1.2895/0.5514. Training-thread tuning reduced 300 steps
  from 5,010.6s at 4 threads to a projected ~2,067s at 16 threads; iteration 2
  completed at policy/value loss 1.8375/0.5213.

## Review log

Pending independent review before any long-running v2 training.

### 2026-08-21T03:45:08+00:00 — independent review

- Scope: current uncommitted diff against `e5ac32e` / `master`, including
  dynamic batching, self-play/CLI wiring, 32x4 -> 64x8 expansion, trainer
  bootstrap, tests, `.gitignore`, and this plan.
- Conclusion: **not approved; block long-running v2 training**.
- Builder blockers:
  1. `model_expand.cpp` zero-initializes every added convolution weight and
     gives every added channel the same BN state/bias. The widened channels are
     permutation-symmetric and receive identical updates; the policy-head
     “nonzero” gradient asserted by the test is only about `8.5e-10` repeated
     across all added channels (consistent with BN roundoff on the constant
     `0.01` activation), not evidence of independent learnable 64-channel
     capacity. Add exact-preserving symmetry breaking and a multi-step test
     proving added channels diverge and meaningful gradients reach the widened
     stem/blocks/heads.
  2. New trailing train options are silently ignored: for example,
     `train --batch-inference` starts the historical unbatched default instead
     of rejecting the missing `0|1` value. Validate missing values for the new
     CLI options and expose `--expand-init` in usage.
  3. Expansion compatibility does not compare BN epsilon. The trainer can
     return success while copying a source checkpoint into a destination with
     a different epsilon, which does not preserve inference numerically.
  4. Remove the `.gitignore` no-op: `runtime_v2/` was already ignored and the
     new duplicate entry is unrelated/non-minimal.
- Validation gaps to close: exercise `Stop()` while requests are queued/in
  flight; cover expanded trainer bootstrap plus checkpoint resume; cover a
  64x8 student with a 32x4 teacher in both local and batched evaluator paths.
- Evidence reviewed: clean Release build and `./bin/test_az` (`9377` checks,
  zero failures); ThreadSanitizer build/run of the same suite (zero reported
  races); `git diff --check` clean. The tests do not invalidate the expansion
  blocker above because they check only whether any added gradient exceeds
  `1e-10`.
- Minimality/performance: dynamic batching remains opt-in and no extra MCTS
  evaluation amplification was found; the plan's benchmark says it is slower
  than worker-local inference. The diff is not minimal because of the redundant
  `.gitignore` change. Final 48/96/600 and L6/L7 acceptance gates remain
  unexecuted.

### 2026-08-21T03:54:05+00:00 — independent re-review after fixes

- Scope: current uncommitted worktree against `e5ac32e` / `master`, with focus
  on the previous findings plus Net2Net learnability, smaller-teacher paths,
  dynamic-batch shutdown, trainer optimizer/checkpoint order, and historical
  defaults.
- Conclusion: **not approved; keep long-running v2 training blocked**.
- Confirmed fixed: all five new train options now reject a missing trailing
  value; `--expand-init` is documented; expansion compares BN epsilon and
  momentum; `.gitignore` is unchanged from the baseline; and an expanded
  trainer bootstrap/checkpoint/resume test now runs through two iterations.
- Builder blockers:
  1. **High — Net2Net learnability remains unproven.** The new symmetry break is
     only distinct positive BN biases on otherwise constant zero-convolution
     channels. For the 1x1 policy connection, the following training BN makes
     the input-gradient sum zero, so a constant added channel has zero analytic
     outgoing-weight gradient; the test now accepts `>1e-12` roundoff and only
     checks one backward pass. It performs no optimizer step and never proves
     widened stem channels, blocks, and heads become input-dependent and
     diverge over multiple steps. Replace this with a genuinely learnable,
     exact-preserving initialization and the previously requested multi-step
     evidence before spending a v2 run.
  2. **Medium — required concurrency/compatibility coverage is still absent.**
     The partial-batch test calls `Stop()` only after synchronous `Predict()`
     has returned; it does not stop with queued/in-flight callers. The teacher
     test still uses equal structures and only the local path. Add queued and
     in-flight shutdown coverage, plus a 64x8 student / 32x4 teacher test for
     both local and dynamic-batch evaluators. Static inspection found the
     smaller-teacher wiring plausible, but the plan-required paths remain
     unverified.
  3. **Medium — the new `expand` CLI still ignores missing trailing values.**
     `expand ... --trunk`, `--blocks`, or `--threads` falls through with the
     default instead of rejecting the malformed command, which can silently
     emit a model with unintended dimensions. Validate the option/value shape
     before parsing, as was done for the train options.
- No additional blocker was found in optimizer/checkpoint ordering: the
  optimizer is sized for the destination before expansion, the checkpoint
  stores the post-step model and matching optimizer state, and resume reloads
  both. Dynamic batching remains opt-in, and the unbatched default call path is
  materially unchanged. No request/MCTS amplification was found.
- Evidence: `./bin/test_az` reports 9,388 checks / 0 failures; direct CLI probes
  confirm the five train options reject missing values; `git diff --check` is
  clean. The worktree also contains an untracked generated `build_native/`
  tree, which must not be included in the source change.
- Minimality: the source abstractions are directly tied to the plan and no
  simpler call-chain change was identified, but the weak learnability check
  cannot justify the added expansion mechanism. Final 48/96/600 and L6/L7
  acceptance gates remain unexecuted.

### 2026-08-21T04:06:52Z — final independent re-review

- Scope: current uncommitted worktree against `e5ac32e` / `master`, including
  the untracked expansion sources, with focus on exactness, learnability,
  smaller-teacher paths, shutdown, checkpoint resume, CLI validation, and
  minimality.
- Conclusion: **not approved; one medium validation blocker remains**. No high
  production-code blocker was found.
- Confirmed fixed: expansion retains independent random newly added output
  features, zeros only added inputs into existing outputs, and initializes new
  blocks with random `conv1` plus zero `conv2`. First-step policy/`conv2` and
  second-step `conv1` gradients now demonstrate the intended staged learning.
  Five real iter440 positions produced byte-identical probe output after 64x8
  expansion. Expanded optimizer/checkpoint state resumes through iteration 2,
  and actual 64x8-student/32x4-teacher smoke runs completed in both local and
  batched modes with all targets supplied by the teacher. Missing-value probes
  for the new train/expand options reject the command.
- Builder blocker:
  1. **Medium — the shutdown regression test itself has a data race.**
     `test/test_main.cpp:789-790` invokes `CHECK` concurrently from caller
     threads, while `CHECK` mutates non-atomic globals `g_checks` and
     `g_failures`. ThreadSanitizer reports a race on `g_checks` at line 789 and
     exits nonzero, so the required concurrency/shutdown coverage is not yet a
     clean validation result. Keep assertions/counter mutation on the main
     thread (or otherwise make the harness race-free), retain deterministic
     queued/in-flight `Stop()` coverage, and rerun ThreadSanitizer without
     suppressing the report.
- Evidence: clean Release configure/build; `./bin/test_az` reports 8,270 checks
  / 0 failures and passed 10 repeated Release runs; `git diff --check` is clean;
  exact probes matched 5/5; actual local/batched teacher smokes completed; TSAN
  failed only on the test-harness race above in the halted run.
- Minimality/performance: the new abstractions are directly tied to the plan,
  dynamic batching remains opt-in, and no request/MCTS amplification or simpler
  correct implementation was identified. Generated untracked `build_native/`
  must remain outside the source change. Final 48/96/600 and L6/L7 acceptance
  gates remain pending.

### 2026-08-21T04:18:44Z — final approval re-review

- Scope: current uncommitted worktree against `e5ac32e` / `master`, including
  expansion, dynamic batching, self-play/teacher wiring, checkpoint resume,
  CLI validation, tests, and the affected evaluation/gate call chain.
- Conclusion: **not approved; one medium concurrency blocker remains**.
- Builder blocker:
  1. **Medium — evaluator snapshot copying is still concurrent in evaluation
     and acceptance paths.** `RunDuel` (`src/train/self_play.cpp:358-365`),
     `RunVsRandom` (`src/train/self_play.cpp:412-417`), and `RunGauntlet`
     (`src/arena/gauntlet.cpp:94-99`) call `AssignWeights` from worker threads.
     `AssignWeights` obtains source parameters through `mutable_weight()`, which
     writes `packed_weight_valid_`; concurrent copies therefore race on the
     shared source network. A direct TSAN run of
     `alphazero eval --model runtime/final_iter440.net --games 2 --sims 1
     --workers 2` reports this race at `self_play.cpp:417`. Snapshot these
     worker evaluators serially before launching threads, as self-play now does,
     and add coverage for the gate/eval path. This must be fixed before the
     48/96/600 and L6/L7 acceptance runs.
- Confirmed fixes: the shutdown test harness race is gone; exact-preserving
  asymmetric expansion, staged gradients, mixed-size teacher paths, expanded
  checkpoint resume, and missing-value rejection are present.
- Evidence: fresh Release configure/build and suite passed with 8,256 checks / 0
  failures; fresh TSAN configure/build and suite also passed with 8,256 checks /
  0 failures; `git diff --check` is clean. The direct TSAN production-path probe
  above demonstrates that the suite does not cover all concurrent snapshot
  call sites.
- Minimality/performance: dynamic batching remains opt-in; no MCTS request
  amplification or simpler implementation was found. Generated untracked
  `build_native/` and `build_tsan/` remain outside the intended source change.

### 2026-08-21T04:26:15Z — sign-off review

- Scope: current uncommitted worktree against `e5ac32e` / `master`, using this
  plan as the review baseline, including all previously reviewed expansion,
  batching, teacher, resume, CLI, evaluation, and acceptance-gate paths.
- Conclusion: **APPROVED; no high- or medium-severity findings**.
- The remaining concurrency blocker is fixed: self-play, duel, random eval,
  and gauntlet now initialize and copy every worker evaluator serially before
  any worker thread starts. No per-game or per-MCTS snapshot amplification was
  introduced.
- Validation: independently reran the normal and ThreadSanitizer suites; each
  passed 8,256 checks with zero failures. Two-worker ThreadSanitizer production
  probes for `eval`, `arena`, and `gauntlet` also completed without reports;
  `git diff --check e5ac32e` is clean. Net2Net exactness/staged learnability,
  mixed-size local and batched teacher paths, dynamic drain, expanded
  checkpoint resume, and CLI validation remain covered by the reviewed tests.
- Minimality/performance: the diff remains plan-scoped, dynamic batching is
  opt-in, and no repeated I/O, avoidable request amplification, or simpler safe
  implementation was identified. Generated `build_native/` and `build_tsan/`
  trees remain untracked validation artifacts and must stay outside the source
  change. Final 48/96/600 and L6/L7 runs remain model-acceptance gates, not
  blockers for this implementation sign-off.
