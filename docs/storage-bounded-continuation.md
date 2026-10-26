# Storage-bounded continuation of exact coverage training — design

Status: **design only (2026-10-25).** Nothing here is implemented, no training was started, and no artifact was
modified or promoted. Measurements below are read-only queries of
`output/exact/coverage-2026-10-24/overnight/latest.sqlite` (same 1,095,300 iterations as the candidate).

## Goal and constraints

Train the 192 unready targets of the 2026-10-24 manifest (75/88 river, deep raise histories) within the existing
16 GiB candidate cap on local CPU, **without** weakening anything that decides what the bot may play:

- serving stays exact-only: no buckets, group averages, nearest stacks or substitutes;
- readiness stays ">= 1000 updates AND a valid positive average strategy with matching actions";
- training stays whole-game-from-blinds with the existing importance corrections (no isolated subgames);
- resume guards stay: no silent change of source, config, runtime or target manifest;
- promotion still requires an independent frozen evaluation;
- every stored regret / average value is preserved exactly (no lossy compression of training state).

## Where the 17 GB goes (measured)

| | preflop | flop | turn | river | all |
|---|---|---|---|---|---|
| infosets | 3,039,859 | 2,789,428 | 1,644,288 | 822,594 | 8,296,169 |
| visited exactly once | 80% | 92% | 93% | 95% | 88% |
| never regret-updated (`updates=0`) | 42% | 46% | 47% | 48% | 45% |
| `updates >= 1000` | 338 | 134 | 89 | 15 | 576 |
| key text, bytes (avg) | 247 | 289 | 327 | 365 | ~280 |
| actions + regrets + totals as JSON text, bytes (avg) | 958 | 976 | 808 | 632 | ~900 |
| actions per infoset (avg) | 23.3 | 27.5 | 24.1 | 20.1 | ~24 |

17,181,073,408 bytes / 8,296,169 rows ≈ **2.07 KB per infoset**. The causes:

1. **The key is verbose JSON stored twice.** `key TEXT PRIMARY KEY` on a rowid table keeps the ~280-byte key
   in the table and again in its automatic index (~560 bytes per row).
2. **Numbers are JSON text.** Each regret/average is a 17-20 character decimal string (`304045.00000000006`)
   instead of 8 bytes, and the action labels are repeated per row.
3. **45% of rows carry all-zero regret vectors** written out in full.
4. **Two full copies.** `latest.sqlite` is a full `sqlite3.backup` of the candidate, rewritten every 15 minutes
   (17 GB of writes per snapshot), so the run occupies 34 GB plus I/O.

Separately, **training effort is misallocated**: `CoverageTrainer._sample` round-robins over *all* 761 targets, so
after the run ~75% of guided trajectories go to targets that are already ready.

## Design

### A. Storage schema v2 (lossless)

```sql
CREATE TABLE menus (id INTEGER PRIMARY KEY, actions TEXT NOT NULL UNIQUE);      -- deduplicated action menus
CREATE TABLE nodes (
  digest  BLOB PRIMARY KEY,          -- BLAKE2b-128 of the canonical key JSON (bytes as today)
  menu    INTEGER NOT NULL REFERENCES menus(id),
  regrets BLOB,                      -- little-endian float64[len(menu)]; NULL means all zero
  total   BLOB,                      -- little-endian float64[len(menu)]; NULL means all zero
  visits  INTEGER NOT NULL, updates INTEGER NOT NULL, street INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE node_keys (digest BLOB PRIMARY KEY, key TEXT NOT NULL) WITHOUT ROWID;  -- see below
```

- **float64 bit-exact.** A JSON decimal written by Python's `repr` parses back to the same double, so
  float64 BLOBs hold exactly the current values. float32 is rejected: regret sums like 304045.00000000006
  already exceed its precision.
- **Full key text is kept where an exact identity is asserted.** That means every coverage target, and every node
  that reaches the readiness floor; `node_keys` rows are written at the moment `updates` first crosses the floor.
  Serving and export read a node only after checking `node_keys.key` equals the requested canonical key, so a
  digest collision can never serve a wrong policy. With ~1e8 nodes the BLAKE2b-128 collision probability is
  ~1e-23. A collision inside training would merge two infosets' regrets, so the trainer also stores the key for
  any node whose digest is inserted twice with different text, and fails loudly.
- **Estimated size:** 16 (digest) + ~3 (menu id) + 192 (24 float64 regrets, NULL for ~45%) + 192 (totals)
  + ~8 (ints) + ~20 (B-tree overhead) ≈ **300-430 bytes/infoset, 5-7x smaller.** Under the same 16 GiB cap
  that is room for roughly **40M infosets instead of 8.3M**. Encoding BLOBs is also cheaper than formatting and
  parsing JSON, so iterations per second should not drop. This is to be measured, not assumed.

### B. Verified migration of the existing candidate

`scripts/migrate_exact_storage.py SOURCE DEST`:

1. Open SOURCE read-only and stream rows in key order into a new DEST (never in place).
2. Copy metadata unchanged. Add `storage_schema: 2` and a `migration` record: source path, source SHA-256 and
   iterations, schema-1 and schema-2 storage code hashes, row counts, and the verifier's result.
3. **Verify every row** against SOURCE by key: same menu, bit-identical float64 regrets and totals (an all-zero
   vector equals NULL), same visits, updates and street. Also check aggregate counts, that every target's
   coverage/readiness result is identical, and that `ExactPolicy.lookup` agrees on all targets and a random
   sample of 100k nodes.
4. DEST is usable only if verification passes. SOURCE is not deleted automatically: freeing the old 17 GB files
   is a separate step that needs user approval.

Expected cost: one streaming pass of the 17 GB file plus a verification pass (~2 x 3 min of I/O plus JSON parsing
of 8.3M rows: roughly 1-2 h of CPU), and ~3-4 GB for DEST.

### C. Resume guard: separate the algorithm fingerprint from the storage fingerprint

Today one code hash covers the trainer, so any storage change blocks resume. Split it:

- **algorithm fingerprint:** game rules and menus (`preflop/exactgame.py`), the MCCFR walk and updates, the
  sampling and importance corrections (`preflop/exacttrain.py`, `preflop/exactcoverage.py`), the training config
  and the target manifest. Resume requires it unchanged, except for a declared phase transition (D).
- **storage fingerprint:** the encoder/decoder module. A change is accepted only if the artifact carries a
  verified `migration` record for exactly that storage hash.

Equivalence test (tiny config, fixed seed): train N iterations in schema 1, migrate, train M more in schema 2. The
result must be bit-identical, node by node, to N+M iterations trained in schema 2 from the same start. RNG state is
part of the metadata and carries over.

### D. Phase 2 sampling: guide only unready targets (a declared config change)

Replace the round-robin over all targets with a round-robin over **targets still below readiness**, recomputed
after every batch. When a target becomes ready it leaves the guided rotation; it keeps getting updates through
the 20% uniform deals and shared paths.

- Each trajectory keeps its own correction: the chance weight `p/q` for the target it was guided to, and the
  90/10 history proposal with the existing outcome-sampling corrections. Choosing *which* target to guide toward is
  a deterministic schedule, as now, so the per-trajectory importance weights keep the same formula.
- About 4x more guided trajectories per unready target (761 → 192 in rotation, shrinking as targets finish).
- This changes the sampling distribution, so it is **not a silent resume**. Metadata records `phase: 2`,
  `scheduler: unready_round_robin`, the phase start iteration, and the parent phase's config hash. Phase-1
  statistics stay attributable, and the resume guard accepts only this declared transition.
- Readiness, the target manifest and the 20% uniform full-game share are unchanged. Lowering the uniform share
  would save the most storage (it creates most visit-once nodes), but it changes the training prior, so it is out
  of scope.

### E. Snapshots without a second full copy

Between batches the loop already runs `PRAGMA wal_checkpoint(TRUNCATE)`, so the main file is consistent there.
Take snapshots at that point with an **APFS clone** (`clonefile`, i.e. `cp -c`): instant, and copy-on-write
blocks are shared until the candidate rewrites them. Snapshot cost becomes proportional to the pages that changed,
not 17 GB per snapshot. On non-APFS volumes fall back to the existing `sqlite3.backup`. The 16 GiB cap stays on
candidate+WAL; status also reports the snapshot's actual allocated bytes (`du`, not `st_size`).

### F. Small serving / evaluation artifact

`scripts/train_exact.py export-ready`: write a separate frozen artifact containing only nodes that pass the
unchanged readiness and validity checks, with full key text, the action menu and the average strategy (no
regrets). It is expected to be KB-MB in size. Frozen evaluation and promotion read this artifact. ExactPolicy
verifies its checksum and the key text, and applies the same `min_updates` and validity checks it applies today.
Everything else still withholds. The full candidate is kept only for resuming.

### G. Budget behaviour

Keep `--max-gb 16`, the free-disk guard and the existing stop reasons. Add projected bytes per infoset and per
street to the status file so a stop can be predicted. **Not proposed:** evicting visit-once or zero-regret
non-target nodes. That is "forgetting", a change to the algorithm whose effect on the targets' values is unmeasured;
it would need its own design, approval and lineage.

## Implementation plan (each step RED test first, then GREEN, then HANDOFF note)

1. RED: codec round-trip (bit-exact float64, NULL zero vectors, menu dedupe); digest-collision detection;
   `node_keys` written when a node crosses the floor; lookup refuses a node without a matching key text.
2. GREEN: schema-v2 store behind the existing Trainer and ExactPolicy interfaces; split fingerprints.
3. RED/GREEN: migration and verifier, plus the train-migrate-resume equivalence test on a tiny config.
4. RED/GREEN: phase-2 unready scheduler with declared-phase metadata, and a guard rejecting undeclared changes.
5. RED/GREEN: clone snapshot with backup fallback; `export-ready` artifact; evaluation and promotion read it.
6. Validation: full test suite, coverage, compileall, reviewers, then a 2000-iteration bounded mechanics run
   (as in the 2026-10-24 validation) comparing bytes per infoset and iterations per second.
7. Operations, each step with user approval:
   - migrate and verify the candidate;
   - decide whether to delete the old 17 GB `latest.sqlite`;
   - launch phase 2 overnight (local CPU, nice +10, 1 thread, 16 GiB cap);
   - export-ready, then an independent frozen evaluation;
   - promotion as a separate decision.

## Expected outcome and risks

- About 5-7x more infosets under the same cap, and ~4x faster progress per remaining target. Whether one more
  overnight run readies the remaining river and deep-raise targets is **unknown**: river targets had the lowest
  update rates, and visits are not convergence.
- Readiness counts still say nothing about strength. `full_game_complete:false` and the evaluation gate remain.
- Migration verification costs time but is required; any mismatch aborts and leaves the source untouched.
- The algorithm fingerprint must cover every training-relevant module, so storage refactors can't hide algorithm
  drift. Reviewers should check the split.

## Decisions for the user

1. Approve schema v2 plus the verified migration (A-C)?
2. Approve the phase-2 unready-only scheduler as a declared config change (D)?
3. After verification, delete the old 17 GB `latest.sqlite` (and later the schema-1 candidate)?
4. Keep the 16 GiB cap, or set a different budget (390 GB free now)?
