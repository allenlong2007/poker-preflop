# Exact-card training — Claude handoff

**Status: implemented, reviewed, tested; local candidate run COMPLETED at its 16 GiB storage limit (2026-10-23).** Read this first. User approved local CPU with overnight runs; no cloud spend or automatic promotion. Detailed design/commands/limits: `docs/exact-card-training.md`. Prior logging work: `docs/decision-logging.md`.

## Completed job — checkpoint saved; evaluation still required

- Actual PID **68736**, managed exec session **32039**. Two detached launch attempts exited before starting; `training.pid` now contains the verified real PID.
- Candidate: `output/exact/overnight/candidate.sqlite`; status: `candidate.status.json`; log: `training.log`; PID: `training.pid` in the same directory. Generated files are Git-ignored.
- One worker, numerical threads limited to one, nice priority +10. Budget: 8 hours, **16 GiB candidate+WAL cap**, free-disk guard; it can stop earlier on storage/numeric limits. Latest frozen snapshot every 900s and at clean exit. Keep Codex/Mac running for the managed process.
- Last verified progress: **42,200 iterations, 215,164 exact infosets, 165,611 regret updates**, ~212MB at 33s; advancing normally. This snapshot will become stale: query the status file.
- Config: 0.01bb chips; stack pool 6/10/15/25/40/60/100bb; minimum/all-in plus 25/50/75/100/150/200%-pot increments. No focus dataset in this run.

```sh
.venv/bin/python scripts/train_exact.py status --path output/exact/overnight/candidate.sqlite
# Resume only after previous process stops; saved RNG/config are reused:
.venv/bin/python scripts/train_exact.py train --path output/exact/overnight/candidate.sqlite --hours 8 --batch 200 --max-gb 16
# Immutable evaluation snapshot while writer runs:
.venv/bin/python scripts/train_exact.py snapshot --path output/exact/overnight/candidate.sqlite --output output/exact/overnight/frozen-evaluation.sqlite
```

SIGTERM to the verified PID stops after the current batch. Do not trust an old PID without checking. Source/config/dependency drift rejects resume: do not edit training code during this run. Last implementation commit: **84c01c4**; final handoff commit follows it and changes no Python.

## Shipped

- `preflop/exactgame.py`: exact private cards/suits, visible board, stack/chips, full betting recall, legal repeated raises/all-ins. Only hole order and simultaneous flop order canonicalized; no textures or strength buckets.
- `preflop/exacttrain.py`: signed outcome-sampling MCCFR, reach-weighted average strategy, SQLite/WAL + RNG-atomic checkpoints, single-writer lease, runtime/code/config guards, cheap aggregate counters, online frozen backup, read-only serving and minimum-update gates.
- `scripts/train_exact.py`: finite CPU/storage budgets, status/resume/snapshot, frozen duplicate held-out evaluation and promotion. Evaluation requires immutable standalone snapshots, not WAL candidates. Promotion binds bytes/source/baseline/runtime and evaluated serving threshold; destination replacement is leased/atomic.
- `bot/engine.py`: covered exact policies bypass legacy bucket/rule overrides. Unpromoted/unseen/undertrained/unsupported lookups explicitly log fallback. `bot/audit.py` fingerprints the exact artifact.
- `bot/exact_opponent.py` + `bot/opponent.py`: exact per-action after-reveal evidence (cards, evaluator/kickers, suits/draw completions, amounts, price, earlier history), validated and indexed idempotently. Raw JSONL is archived first; index failures cannot lose hands. Historical rebuild indexed **58 revealed hands from 429 stored hands, zero invalid records**. Raw history files untouched.

## Evidence and remaining limits

- **88 tests passed**, measured modules **89% coverage**; new exactgame 93%, exacttrain 91%, revealed analysis 80%, CLI 84%. `compileall` and `git diff --check` passed. Coverage file: `/private/tmp/poker-exact.coverage`.
- Same sampler passes a Kuhn imperfect-information regression: exact best-response exploitability <0.12 after 10k iterations. This is not an NLHE exploitability certificate.
- Bounded NLHE validation: 10k iterations → 56,592 infosets/44,387 regret updates. Artifacts in `output/exact/validation/`. Frozen 100-deal duplicate tests: 1000-update gate serves 0% (too sparse); exploratory min=1 serves 113/389 decisions (29.0%), zero illegal actions. **Neither passes promotion; no new playing-strength claim.** Synthetic passing metrics exist only in temporary unit tests to verify promotion plumbing.
- Python and MLE reviewers approve; all material findings fixed (raw capture ordering, card/metadata validation, dependency drift, focus isolation, artifact immutability, evaluated threshold, provenance, destination races, status-scan scaling).
- Exact tables cannot generalize to unseen states. Default size menus remain an action abstraction; `--exhaustive` enumerates every configured legal chip total at much higher cost. No neural approximation or full-game Nash guarantee.
- Focus roots use uniform compatible private ranges, separate namespaces, and candidate-only lookup; reports serving focus policies cannot promote without future posterior validation. Shown-hand samples remain selection biased and are not silently converted into confident live exploits.
- **Live policy remains legacy fallback until a real checkpoint passes evaluation/promotion.** No active exact model was installed. Restart server to load new logging/evidence code; promoted models also need restart.

All steps have RED/GREEN Git checkpoints (start 350187f, game 7abceec, trainer d24ad51, integration ca24252, lifecycle 57869f3; final reviewed implementation 84c01c4). Use `git log` for integrity-fix checkpoints. Next: let the candidate run, freeze it, evaluate at ≥2000 duplicate deals/min_updates≥1000, inspect coverage/error slices before promotion. No recurring scheduler installed.

Unrelated pre-existing untracked outputs left alone: output/match/equity_vs_random.json, output/match/rival_train.json, output/trap_study.log, output/trap_study/.


## Completion detected 2026-10-23

Run stopped cleanly at `candidate_disk_limit` after 18,106.64 seconds (~5h 2m), before its 8-hour time budget. Final: **5,896,000 iterations; 15,259,454 exact information sets; 24,354,392 regret updates**. Candidate size: 17,180,086,272 bytes (~16 GiB). Frozen checkpoint `output/exact/overnight/latest.sqlite` exists and its metadata confirms all 5,896,000 iterations. Candidate/status/log remain in the same directory. This completes this bounded run, not full solver training or validation; no model was promoted and no new run/evaluation was started. Completion monitor is being removed after this one notification. Earlier running-progress/PID notes above are historical.

## Opponent confidence work — 2026-10-24, step 1 (RED)

New task: gate opponent adaptation on uncertainty before changing strategy. Regression tests in `tests/test_read_confidence.py` cover sparse reads, independent hand counts, conservative supported reads, missing provenance, showdown selection bias, and population leakage. RED confirmed: `bot.reads` does not exist yet. Exact training is complete as above; do not restart or promote it. No production changes in this step.

### Step 2 — confidence gates implemented; calibration RED next

`bot/reads.py` adds distinct-hand deduplication/clustering, simultaneous anytime Hoeffding intervals, minimum support/width/practical-effect gates, and conservative frequency projection. `bot/adapt.py` gates preflop locks and serves only supported postflop nodes; population strictness, selected-reveal size/strictness and live bluff refits are withheld. `bot/engine.py` logs gate evidence and requires support for direct shove overrides (conditional shove ranges abstain). First 4 regression tests GREEN. Added temporal calibration regression now RED (`calibration_report` missing), plus identity isolation test GREEN. Full regression suite/review still pending; no calibration improvement or strength claim yet.

### Step 3 — held-out calibration + reviewed gate corrections

Shadow calibration now fits the first half of chronologically ordered revealed hands, validates on the later half, and reports per-hand Brier loss/residual plus paired improvement uncertainty. Never promotes selected-reveal fits. `python -m bot.reads [opponent]` emits local reproducible reports. Full suite passed **96 tests** before the latest exact-jam refinement. Review found/fixed (with RED/GREEN regressions) unrelated actions qualifying shove reads and mismatched postflop baseline mixtures. A further regression now separates actual effective-stack jams from legacy60%-stack labels, and isolates exact depth/prefix-size contexts with countable alpha allocation. Final regression/re-review pending. Preliminary existing-data report:10 identities/429hands; zero approved action reads; insufficient independent reveals/chronology for opponent calibration. No strength improvement claimed; exact training untouched.
