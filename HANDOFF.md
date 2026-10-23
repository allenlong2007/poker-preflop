# Decision/version logging — Claude handoff

**Status: complete, 2026-10-23. Restart the PokerNow server to activate.** Strategy remains v6; run UUID + source/model hashes distinguish builds. No policies were retrained or historical logs rewritten.

Read this file first; `CLAUDE.md` points here. Exact schema/commands: `docs/decision-logging.md`. All task changes are checkpointed in Git; unrelated untracked output files were left alone.

## Shipped

- `bot/audit.py`: one run manifest; source/model SHA-256 fingerprints; versioned decisions; hand links recovered across restarts; legacy/truncated JSONL recovery.
- `bot/engine.py`, `bot/potodds.py`: numeric equity/price/range evidence, input state, opponent-read snapshot, original lookup probabilities, adjustment traces, legal options. Uses already computed equity; no extra simulation.
- `bot/pokernow/server.py`: logs hand/player/run IDs, exact history and size conversion; atomic cached polling; completed hands link to observed decisions.
- `bot/opponent.py`: hero identity; raw imports explicitly have unknown version.
- `scripts/live_review.py`: `--version N`, `--run ID`, `--versions-only`. Default report groups recorded versions and unknown historical hands.
- `tests/test_audit.py`, `tests/test_decision_logging.py`: 12 new regression cases.

Logs: `output/pokernow/{runs,decisions,actions}.jsonl`, completed hands in `output/pokernow/hands/` (all Git-ignored). Recommendations are not proof of played actions; join click reports by decision ID. Old/unobserved hands remain unknown. A hand spanning runs retains all observed run IDs. Hashes describe files at initialization; restart after code/model changes. Review evidence is not bit-for-bit replay.

## Validation

- Full suite: **61 passed**, `.venv/bin/python -m pytest -q` via coverage (localhost test needs socket access outside sandbox).
- Coverage command: `.venv/bin/python -m coverage run --data-file=/private/tmp/poker-preflop-logging.coverage --source=bot.audit,bot.engine,bot.potodds,bot.pokernow.server,scripts.live_review -m pytest -q`.
- Coverage: **87% total**, audit 100%, engine 93%, potodds 97%, server 63%, live_review 55%. Existing server/UI and detailed report paths remain partly uncovered. Coverage installed only in ignored .venv; no runtime dependency added.
- 100 seeded decisions across preflop, 4-bet, flop, river, and short-stack spots match pre-change engine at `2afe595` exactly: action, size, label, probabilities.
- `compileall` and `git diff --check` passed. Version summary smoke test: 429 old hands correctly remain unknown (−579.6bb), not attributed to v6.

## TDD checkpoint evidence

| RED checkpoint | Observed failure | GREEN evidence |
| --- | --- | --- |
| 2afe595 | missing bot.audit API | 296fa25: 4 provenance tests passed |
| 7e98959 | missing hand_id / Decision.diagnostics | da1955d: 8 new tests; 57 full tests passed |
| d7708f3 | missing review filters/prior fingerprint; absent-prior logging bug | fixed; 60 full tests passed |
| 5b74a02 | missing preflop adaptation baseline trace | fixed; final 61 full tests passed |

No pending implementation. Next useful work: collect logs from a restarted server, then audit river calls/big pots by exact run. Existing unrelated files: output/match/equity_vs_random.json, output/match/rival_train.json, output/trap_study.log, output/trap_study/.
