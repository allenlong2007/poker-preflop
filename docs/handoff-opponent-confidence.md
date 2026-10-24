# Opponent confidence — Claude handoff

**DONE (2026-10-24): evidence-gated opponent reads and shadow calibration implemented, reviewed, tested.** Restart the bot server to load the changes. No server restart, exact training, model promotion, or lengthy solver evaluation was performed.

## Current behavior

- `bot/reads.py`: deduplicates hero/hand IDs; isolates opponent identities; clusters repeated actions by hand. Anytime Hoeffding intervals, fixed family budget, >=40 distinct hands, <=0.35 interval width, >0.05 practical difference from a fixed reference. Conservative nearest-supported frequency vector; unqualified reads stay at baseline.
- `bot/adapt.py`: gate preflop locks before solving; publish only approved postflop nodes; pool observations/reference with identical node support and weights. Withhold selected-showdown size/strictness changes and population strictness for unknown opponents.
- `bot/engine.py` + `bot/potodds.py`: downstream reads require approval metadata. Shove overrides use a separate actual effective-stack jam channel, exact depths/prior raise amounts, and countable context alpha allocation. Ordinary raises, legacy 60%-stack labels, other stack depths, and conditional re-raise reach cannot qualify a direct shove read.
- `bot/bluffmodel.py`: live serving retains the frozen self-play prior. Shadow first-half fit/later-half validation reports hand-clustered Brier loss, mean residual, and paired improvement uncertainty. Requires 40+40 independently revealed hands and verified monotone source chronology. Selected reveals never authorize full-range live refits.
- Decision logs include `adaptation.read_evidence` plus shove-specific support/interval/reason. Source fingerprinting already covers these files. Read-only CLI: `.venv/bin/python -m bot.reads [OPPONENT_ID]`; legacy `bot.adapt show` routes to it.

## Measured evidence

- Existing local report: `output/pokernow/opponent_read_calibration.json` (ignored; data/source hashes included). **429 hands, 10 identities, zero accepted action reads**. Nine identities lack enough independent revealed hands; one has unverified chronology. No calibration or playing-strength improvement claimed.
- **103 tests passed**. `bot/reads.py` 98% coverage; five touched production modules 88% combined. Coverage database: `/private/tmp/poker-reads.coverage`. `compileall` and `git diff --check` passed.
- Python and MLE reviewers approve. Their three material findings all have RED/GREEN regressions: unrelated actions approving shoves; mismatched baseline mixtures reversing a read; near-jams masquerading as actual jams. Additional tests cover duplicates, sparse samples, cross-depth isolation, repeated-look null streams, future-label leakage, identity resets, unqualified legacy profiles, and read-only report provenance.
- Step commits: `4e2cc23` RED contract; `0cb0575` confidence gates; `100a8be` calibration + reviewed fixes. Final validation/docs commit follows; `git log` and `git diff` show exact changes.

## Limits and next work

Frequency bounds assume stable conditional behavior across hands and compare fixed policy benchmarks. They do not certify inferred hand ranges, poker EV, arbitrary drift, or all opponents jointly. Legacy broad strategy/range abstractions remain. Next: obtain representative delayed labels and durable source/timestamps; evaluate range/calibration slices before enabling any showdown-derived strategy changes. Keep selected reveals in shadow meanwhile. Full contract, commands, thresholds, and assumptions: `docs/opponent-confidence.md`.

## Previous exact-card training

Bounded run completed at `candidate_disk_limit`: **5,896,000 iterations; 15,259,454 infosets; 24,354,392 updates**. Saved candidate `output/exact/overnight/candidate.sqlite`; frozen checkpoint `output/exact/overnight/latest.sqlite`. This is not a fully trained/validated solver; nothing promoted. Previous detailed handoff preserved in `docs/handoff-exact-card-training.md`; implementation guide `docs/exact-card-training.md`. Completion monitor was removed. Do not restart training without a new request.

Unrelated untracked outputs left untouched: `output/match/equity_vs_random.json`, `output/match/rival_train.json`, `output/trap_study.log`, `output/trap_study/`.

