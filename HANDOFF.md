# Exact-card training — Claude handoff

Status: implementing (2026-10-23). User approved local CPU overnight training. No paid/cloud compute. Previous logging work is complete at df7c26e; see docs/decision-logging.md.

Goal: exact private cards/suits + visible board + full monetary history/stack in policy inputs; resumable long-running training; legal targeted sizing; exact revealed-opponent evidence; measured checkpoint promotion.

Plan/checkpoints:
1. Exact heads-up chip-accounting game and information-state keys; tests before implementation.
2. Durable outcome-sampling MCCFR, deterministic resume, configurable sizing (including exhaustive legal chip totals), frozen evaluation/promotion, live exact-policy lookup with explicit coverage fallback.
3. Revealed-hand action-by-action exact-card analysis and idempotent indexed storage; public-state focus curriculum without hidden-card leakage.
4. Full tests/reviews; bounded training/evaluation; launch an 8-hour single-CPU candidate run and record PID/status/commands here.

Design: use eval7 already installed; algorithm follows established outcome-sampling MCCFR mathematics (OpenSpiel reference). Exact tables do not generalize to unseen states. Default finite sizing menus are still action abstraction; exhaustive mode enumerates legal chip increments and costs much more. Checkpoints remain candidates until held-out evaluation gates pass. Live fallback remains explicit while coverage is insufficient. Do not claim a fully trained full-game Nash solution.

Research: OpenSpiel (sampling reference) https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/outcome_sampling_mccfr.py ; Deep CFR https://proceedings.mlr.press/v97/brown19b.html . TexasSolver exists locally but is an exact-board subgame solver, not durable whole-game self-play training; no source copied.

Unacceptable errors: opponent/future cards in live keys; retrospectively using revealed cards for earlier decisions; illegal sizes/minraises; partial resume/RNG checkpoints; deploying sparse/unvalidated policies. Revealed-showdown samples remain selection biased.

Unrelated untracked outputs from before this task remain untouched: output/match/equity_vs_random.json, output/match/rival_train.json, output/trap_study.log, output/trap_study/.

Step 1 GREEN: 350187f RED (missing exactgame API); 5 tests now pass. preflop/exactgame.py implements exact private/public cards, full history keys, integer chips, no-limit minimum/full/short all-in raises, zero-sum payouts, configurable sizing and exhaustive chip-size mode. Focus reconstruction retains off-menu observed sizes without rounding to a trained action. Training focus samples both private hands, never imports shown opponent cards into a past decision.

Step 2 RED: tests/test_exacttrain.py fails on missing exacttrain API. Tests specify importance weighting, deterministic interrupted/resumed training, atomic rollback including RNG, private-label-free focus roots, immutable candidate lookup with coverage gates, and rejecting sparse/losing promotion reports.

Step 2 core GREEN: 49740ea RED (missing exacttrain API); 11 exactgame/exacttrain tests pass. preflop/exacttrain.py adds single-writer SQLite/WAL transactions, signed outcome-sampling importance weights, reach-weighted average policy, code/config/focus fingerprints, RNG-atomic resume, immutable snapshots, read-only lookup with update thresholds, and declared evaluation gates. Default focus-root ranges are uniform compatible combinations (explicit approximation, not a solved betting-line posterior). Live lookup/CLI/evaluation wiring follow.

Integration RED: four tests fail as intended: Engine lacks exact-policy routing, revealed analysis/index APIs absent, parser lacks after-reveal analysis. Tests require covered candidates to bypass legacy bucket overrides, unpromoted fallback, exact suit/draw labels per action without later runout, and idempotent reveal storage.

Integration GREEN: 15 exact-game/training/integration tests pass. Engine can read a frozen exact-policy snapshot (covered states bypass bucket/heuristic overrides; unpromoted/unseen/insufficient states explicitly fall back). Audit model hashes include the exact snapshot. Raw opponent records now retain exact per-action cards/suits, evaluator score/kickers, draws/blockers, price and prior action history only at that street; an idempotent SQLite index stores revealed evidence separately. Unknown cards never enter this index until shown. CLI rebuild is available. Full legacy regression run is in progress.

Full regression: 76 tests passed. Review follow-up RED: focus subgames must have separate public-root namespaces (uniform subgame priors must not mix with whole-game posterior reach). CLI RED: missing scripts.train_exact API; tests require bounded budget, resumable run/status/snapshot/frozen evaluation, and failed promotion leaving no active artifact.

Operational GREEN: 17 exact-path tests pass with bounded CLI, status, snapshots, duplicate held-out evaluation and rejected promotion. Focus namespaces isolated; candidate-only focus lookup, no automatic live promotion of uniform-prior focus results. Runtime versions now guarded on resume; flop order losslessly canonicalized while preserving every card/suit and turn/river order. Reviewer fixes RED: invalid known cards, broken derived index losing raw archives, malformed policy metadata bypassing fallback.

Reviewer fixes GREEN: 20 exact-path tests pass. Raw hand JSONL is archived before best-effort derived indexing, with observable warnings; invalid/conflicting known cards are rejected; malformed metadata/strategy vectors fall back instead of crashing. Rebuilt historical reveal index: 58 revealed hands from 429 stored live hands, zero invalid records. No raw histories rewritten. Python and MLE reviews found no demonstrated betting-accounting or importance-weighting bug; all actionable findings are being addressed.

Operational follow-up RED: snapshot CLI currently attempts a writer lease and cannot freeze while overnight training is running. New regression requires consistent read-only backup while writer stays open, then independently resuming the frozen snapshot.

Review follow-ups: online snapshot fix passes. New Kuhn imperfect-information regression passes after 10k training iterations (exact best-response exploitability <0.12). Two deployment contracts are RED: promotion must preserve evaluated update threshold, and evaluation must reject a mutable WAL candidate. Also binding current code/baseline/runtime provenance at promotion and replacing per-batch whole-table scans with transactional counters before the overnight run.

Deployment fixes GREEN: evaluation/promotion hold a shared artifact lease, require standalone DELETE-journal snapshots (not writable WAL candidates), and stream hashes before/after. Snapshot replacement holds an exclusive destination lease. Promotion binds candidate bytes plus current source/baseline/runtime fingerprints; copied bytes are checked before activation. Live minimum visits cannot be lower than the evaluated threshold. Metadata aggregate counters remove per-batch full-table scans. Added tests cover stale provenance, successful promotion plumbing with explicitly synthetic metrics, runtime drift, and disk-budget stop. Final reviewer pass and full coverage underway.

Full coverage run before final lease follow-up: 88 tests passed. Final Python review found promotion destination replacement lacked the exclusive lease used by snapshots; regression now RED when attempting to promote over an actively trained output. Fix will share one leased/unique-temp destination helper. MLE final review otherwise approves current training/evaluation foundation; no remaining algorithm/leakage blockers reported.

Final validation GREEN: 88 full tests passed, measured coverage 89% (exactgame 93%, exacttrain 91%, exact-opponent 80%, training CLI 84%); compileall/diff checks pass. Both Python and MLE reviewers confirm no remaining blockers. Bounded whole-game run: 10,000 iterations, 56,592 exact infosets, 44,387 regret updates, ~61MB candidate. Frozen 100-deal duplicate evaluations: normal 1000-update gate serves 0% (correctly too sparse); exploratory min=1 serves 113/389 decisions (29.0%) with no illegal actions. Neither report passes promotion; no playing-strength claim. Reports and snapshot: output/exact/validation/. Overnight launch and concise final handoff follow.
