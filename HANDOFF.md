# Decision/version logging — current handoff

Status: step 1 GREEN; step 2 in progress (2026-10-23). No strategy changes intended.

Goal: attribute new decisions and completed hands to an exact server run/build; preserve structured decision evidence for later river/big-pot reviews.

Steps:
1. Add regression tests, run them RED, then implement run provenance and durable hand/decision links.
2. Add structured inputs, equity/price/range diagnostics, and before/after adjustment probabilities.
3. Verify compatibility and full tests; document the log schema and review command.

Historical records must remain unknown unless actual logged decisions establish provenance. Recommendations are not proof that an action was played.

Existing unrelated untracked files: output/match/equity_vs_random.json, output/match/rival_train.json, output/trap_study.log, output/trap_study/. Leave these alone.

Baseline: 49 tests pass (localhost test needs socket access outside sandbox). No CLAUDE.md or AGENTS.md existed in this repository.

Step 1 RED: `tests/test_audit.py` specifies run/build fingerprints, single run manifest, restart recovery, unknown historical attribution, cached-decision deduplication, and hand links. `.venv/bin/python -m pytest -q tests/test_audit.py` fails because `bot.audit` does not exist yet (the intended missing API). CLAUDE.md now points here.

Step 1 implementation: `bot/audit.py` adds run manifests, source/policy SHA-256 fingerprints, per-hand decision links recovered after restart, and tolerant recovery of legacy/truncated JSONL. `bot/opponent.py` records hero identity and explicitly unknown version on raw hand imports. Server integration/diagnostics follow in step 2. RED checkpoint: 2afe595.

Step 1 GREEN: `.venv/bin/python -m pytest -q tests/test_audit.py -k 'not server'` → 4 passed. Full server integration test is intentionally pending step 2.

Step 2 RED: server integration test fails on missing `hand_id`; all 3 `tests/test_decision_logging.py` cases fail on missing `Decision.diagnostics`. These cover reusing computed equity (no extra Monte Carlo pass), short-stack evidence, preflop re-raise evidence, and JSON serialization.
