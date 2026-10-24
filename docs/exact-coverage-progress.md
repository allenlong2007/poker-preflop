# Exact coverage implementation checkpoints
, 2026-10-24 (RED)

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


Final:121 full-suite tests passed. Both Python and MLE reviewers approve after readiness/config-integrity fixes. Detailed fallback counters RED/GREEN regression now passes. Full validation and background job status are in HANDOFF.md.
