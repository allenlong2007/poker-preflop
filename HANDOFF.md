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

## NEW TASK — exact-only coverage, 2026-10-24 (RED)

User explicitly chose **withhold action and queue exact training** when no validated exact-card strategy exists. Contract: default/live engine cannot call legacy preflop averages, board texture/12-bucket turn/river tables, short-stack/deep-stack borrowing, or truncated raise rules for uncovered states. Exact keys preserve every card/suit, effective stack, legal chip amounts and complete raise history. Add durable attempt/fallback-frequency/target accounting and bounded resumable CPU coverage training; candidate training is not live promotion or universal board coverage. Full NLHE coverage cannot honestly be claimed from local CPU sampling. RED:2 missing `bot.coverage` failures; existing exact repeated-raise/key regression passes. Prior task summary above remains valid/history.

### Exact coverage step 2 — serving GREEN, targeted training RED

Default `Engine` is exact-required; uncovered decisions raise `ExactCoverageError`, persist the exact target and fallback counts, and never enter legacy methods. Server returns `decision:null` plus visible reason/coverage and caches gaps once; warmup skips legacy policies. CLI advice queues gaps too. Legacy research/evaluation/tests now opt in with `exact_required=False`; deep-cell substitution is removed even there, and fallback classes are logged.4 serving/queue regressions GREEN,41 broader tests GREEN (localhost socket test needs normal elevated test runner). Added3 RED tests for targeted coverage training/resume, chance-mixture correction, and observed off-menu raise totals. Existing code remains candid about untrained states; no candidate promoted.

### Exact coverage step 3 — targeted full-game training GREEN

`CoverageTrainer` now replays from blinds in the **whole_game namespace** rather than installing uniform-prior subgame strategies. Round-robin queued cells guide literal hero+board chance events with a full-support80/20 proposal and p/q correction;90% observed-history proposal (full recall, no raise-depth cutoff) uses the existing per-action importance correction. Both players update; sampling proposals never become policy features. `Config.extra_sizes_bb` includes exact observed raise totals in legal menus without nearest-size mapping, preserving old empty-extra config fingerprints.20 targeted/game/trainer tests GREEN, including saved/resumed exact target coverage and candidate-only serving. CLI manifest import/budget/progress integration and final MLE/Python review remain pending. Completion of queued cells will not mean complete NLHE board coverage.

### Exact coverage step 4 — CLI/manifest RED

Added2 failing regressions for reproducible queue/archive preparation, bounded training until queued update floor, and action-time-only board/private-card extraction. `scripts.prepare_coverage` missing as intended. Next implementation adds a hashed immutable exact target manifest, expanded raise menus including observed totals, per-street/depth/raise-count progress, and bounded CPU/time/disk integration. Raw archives remain untouched; preparation will not increment live fallback attempt counts.

### Exact coverage step 5 — manifest/CLI/readiness GREEN

Preparation now imports action-time targets idempotently without inflating live attempt counts, hashes manifests, preserves unsupported requests in the queue, and supports optional all-compatible-private-combo expansion. Full-game curriculum CLI has finite time/disk/iteration budgets and per-street/depth/raise progress. Readiness requires the same legal-action/finite-positive-average checks as serving, not update count alone. Resume verifies target-table hash and supplied manifest configuration (Python reviewer finding fixed with regression). Strict Engine now needs no legacy files and persists its default queue.21 coverage/CLI tests GREEN; full suite running. Actual bounded validation:2000 iterations in6.52s,21971infosets13422updates;0/761 targets at1000-update readiness. Plan includes357preflop/185flop/131turn/88river cells,247exact depths,68extra observed sizing totals.16 unsupported histories remain visible;935 old decision rows lack reconstructable contexts. No universal coverage, convergence, strength, or live eligibility claim.

### Exact coverage step 6 — full regression GREEN; fallback detail RED

**118 full-suite tests passed** before final telemetry refinement. MLE reviewer approves estimator/readiness. Python reviewer resume-config finding fixed and targeted21tests GREEN. New RED regression requires separate counts for legacy nearest preflop sizing and representative-flop hand-group averaging, rather than hiding both under broad legacy labels. These routes remain forbidden by default; counters apply only to explicit legacy research mode. No overnight job has started yet; training code will be frozen after final review/checks.
