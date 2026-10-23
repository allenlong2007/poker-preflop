# Decision/version logging — current handoff

Status: regression tests written; RED confirmed (2026-10-23). No strategy changes intended.

Goal: attribute new decisions and completed hands to an exact server run/build; preserve structured decision evidence for later river/big-pot reviews.

Steps:
1. Add regression tests, run them RED, then implement run provenance and durable hand/decision links.
2. Add structured inputs, equity/price/range diagnostics, and before/after adjustment probabilities.
3. Verify compatibility and full tests; document the log schema and review command.

Historical records must remain unknown unless actual logged decisions establish provenance. Recommendations are not proof that an action was played.

Existing unrelated untracked files: output/match/equity_vs_random.json, output/match/rival_train.json, output/trap_study.log, output/trap_study/. Leave these alone.

Baseline: 49 tests pass (localhost test needs socket access outside sandbox). No CLAUDE.md or AGENTS.md existed in this repository.

Step 1 RED: `tests/test_audit.py` specifies run/build fingerprints, single run manifest, restart recovery, unknown historical attribution, cached-decision deduplication, and hand links. `.venv/bin/python -m pytest -q tests/test_audit.py` fails because `bot.audit` does not exist yet (the intended missing API). CLAUDE.md now points here.
