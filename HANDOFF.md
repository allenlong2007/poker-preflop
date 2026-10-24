# Exact-only coverage — Claude handoff

**Implemented, reviewed, 121 tests passed (2026-10-24). Overnight job prepared; launch/status details follow below.** User explicitly chose: **withhold action and queue exact training** for states lacking a validated exact policy. Read this first; full contract/commands/limits: `docs/exact-coverage.md`.

## Shipped

- `bot/engine.py`: default exact-required mode; exact private cards/suits/visible board/depth/full histories on all streets. No legacy lookup/adjustment or bucket/group/deep-stack substitute. Missing/unpromoted/undertrained/unsupported policies raise `ExactCoverageError` and persist the exact gap. Strict mode needs no legacy policy files. `exact_required=False` preserves explicit research benchmarks; call sites/tests updated accordingly. Deep-cell substitution removed globally.
- `bot/coverage.py`: durable `output/pokernow/coverage.sqlite`, exact target identities, per-reason fallback counts/fractions, exact/withheld/legacy source counts, street/depth/raise-count slices, idempotent archive import. Cached polls count once. New counters start with new instrumented attempts; no fake historical counts.
- `bot/pokernow/server.py`: returns `decision:null`/visible wait on gaps; no automatic substitute. Blocked audit events keep IDs/versions. `bot/audit.py` fingerprints strict-mode config. CLI `bot.ask` also queues gaps. No server was started/restarted; load the updated server before playing.
- `preflop/exactcoverage.py`: guided **whole-game-from-blinds** MCCFR, both players, literal-card/board chance 80/20 mixture with p/q correction; observed-prefix 90/10 action proposal with existing importance corrections. No uniform-prior focus policy substitution. Queued readiness requires update floor AND valid positive average strategy/matching actions. Resume verifies exact target table, source/config/runtime and manifest config.
- `preflop/exactgame.py`: observed exact raise totals augment configurable legal menus; repeated raises have no arbitrary depth cutoff. Empty-extra config fingerprints preserve old frozen artifact readability. `preflop/exacttrain.py` adds proposal hooks. `scripts/prepare_coverage.py` imports action-time contexts, hashes manifests, optionally expands every compatible literal private combo at requested public roots. `scripts/train_exact.py` adds bounded coverage mode/progress/early stop.

## Measured coverage and limits

- Plan: `output/exact/coverage-2026-10-24/targets.json` (ignored). **761 trainable exact cells: 357 preflop, 185 flop, 131 turn, 88 river; 247 exact depths; 68 extra observed sizing totals; 14 pot increments; histories through six prior bets/raises.** Queue also retains 16 unsupported histories (777 requests total); 935 old decision rows lack reconstructable contexts. Raw archives untouched.
- Bounded real-data validation: 2000 iterations, 21971 infosets, 13422 regret updates, 6.52s,~52.8MB; 0/761 cells reached 1000-update readiness. Files: `output/exact/coverage-2026-10-24/validation/`. Mechanics validation only; no strength or convergence claim.
- **121 tests passed**; production coverage summary below/final update. `compileall` and `git diff --check` passed. Coverage DB: `/private/tmp/poker-exact-coverage.coverage`. MLE/Python reviewers approve; regressions cover readiness-with-zero-average, manifest tampering/config drift, strict routing without legacy files, exact turn/river/long raises, private-card expansion, queue idempotency, unsupported gaps, and detailed fallback counters. Synthetic positive-policy routing fixtures are not training evidence.
- Historical source rollup: `output/exact/coverage-2026-10-24/historical-fallbacks.json`: 935 legacy decisions (422 preflop/69 solver/414 model/12 rule/18 short). Old logs lack detailed reason diagnostics; those frequencies are unknown. New live counters currently 0 attempts (archive import is not play).
- **Full board/game coverage is NOT complete.** Reports always retain `full_game_complete:false`. Counts concern the requested frontier only. Zero-own-reach historical lines can remain without average mass even with many visits. High variance/sparse coverage remain risks. No model promoted; default serving will withhold until a real frozen candidate passes independent evaluation/promotion. Broad opponent-read heuristics are bypassed in exact mode; evidence/shadow calibration retained.

## Reproduce / continue

```sh
.venv/bin/python -m bot.coverage
.venv/bin/python scripts/prepare_coverage.py --output output/exact/coverage-2026-10-24/targets.json
.venv/bin/python scripts/train_exact.py status --path output/exact/coverage-2026-10-24/overnight/candidate.sqlite
# Resume only after writer stops; do not change source/config during the run:
.venv/bin/python scripts/train_exact.py train --path output/exact/coverage-2026-10-24/overnight/candidate.sqlite --hours 8 --max-gb 16
```

New requests arriving after manifest freeze need a new manifest/job. Don't silently change a running distribution. Candidates never become live just because target visit counts are met.

Checkpoints: `16765ac` RED contract; `ffac880` strict serving/queue; `d571beb` guided full-game trainer; `4424868` CLI RED; `6225b3e` manifest/CLI/readiness; `b614d95` detailed fallback RED; final GREEN/docs commit follows. Full step ledger: `docs/exact-coverage-progress.md`.

Prior opponent-confidence handoff: `docs/handoff-opponent-confidence.md`. Previous completed exact run remains untouched/readable: 5896000 iterations, 15259454 infosets; `output/exact/overnight/latest.sqlite`; prior detailed handoff `docs/handoff-exact-card-training.md`. Do not resume it after training source/config changes.

Unrelated outputs left untouched: `output/match/equity_vs_random.json`, `output/match/rival_train.json`, `output/trap_study.log`, `output/trap_study/`.
