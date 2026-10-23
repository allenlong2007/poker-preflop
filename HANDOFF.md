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
