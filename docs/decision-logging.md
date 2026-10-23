# Decision and version logs

Restart `python -m bot.pokernow.server` after code or policy changes. Logging is automatic;
the strategy version remains v6 because this change adds evidence, not a new playing policy.
Use source/policy hashes and run IDs to distinguish builds within a version.

All session files live in `output/pokernow/` (Git-ignored). No historical files are rewritten.

| File | Evidence |
| --- | --- |
| `runs.jsonl` | One schema-v1 manifest per run that logs a decision: run UUID, start timestamp (Unix UTC), bot version/name, Git HEAD (null outside Git), source SHA-256, model SHA-256, policy paths, solver-flop/translation/pot-odds flags and runtime adjustment settings. |
| `decisions.jsonl` | One event per observed decision point: schema version, run ID, bot version, PokerNow hand ID, hero/opponent IDs, table snapshot, engine inputs, and recommendation. Cached/concurrent polls do not resample or append duplicates. |
| `hands/<opponent>.jsonl` | Completed hand outcome plus hero ID/name, `bot_version`, and `audit` links to decision IDs/run IDs when observed. Raw imports have `bot_version: null`; older records can lack the field. |
| `actions.jsonl` | Existing browser click reports. Join their `id` to `decision.id`; a recommendation alone does not establish that an action was played successfully. |

`decision.diagnostics` contains:

- `context`: exact cards, position, street history, starting effective stack, current pot and call amount (all amounts in bb).
- `base`: source/node/probabilities before the logged adjustments; this is the translated lookup policy, not a fresh exact-board solve.
- `lookup` where applicable: pot type, texture, hand bucket, line, and whether an untrained cell borrowed a deep-stack line.
- `adjustments`: changed probability distributions, with stage name and before/after values in application order.
- `adaptation`: strictness, preflop blend weight, bluff/defend scales, observed hand count, opponent profile, and active bluff-model coefficients when loaded.
- `legal_options`: engine action/size for each final option. `decision.engine_action` preserves the selected engine size; `decision.to_chips`/`to_bb` preserve the server's legal table sizing. `decision.probs` retains the existing rounded display values.
- Numeric evidence under `pot_odds`, `short_stack`, `vs_shove`, `small_raise`, or `reraise_war` when those calculations run. Includes equity and price, plus range combo count or top-range share, realization, thresholds/weights as appropriate. Equity is reused from the playing calculation; logging adds no equity simulation.

Values are estimates from the current range model. Missing evidence means the calculation did not run,
not zero equity. Probability traces record changes, so a no-op adjustment has no trace entry.

The source hash covers Python files in `bot/`, `preflop/`, and `scripts/`. The model hash covers
the selected postflop `.npz` files, preflop policy, enabled solver-flop JSON files, available bluff prior,
adaptation baseline, and equity matrix. Hashes describe files at audit initialization; do not hot-swap
files during a server run. Git HEAD alone does not identify uncommitted changes; use the hashes.
These records support review, not bit-for-bit replay of RNG/background adaptation.

On restart, schema-v1 decisions rebuild hand links using hand ID **and hero ID**. A hand spanning runs
lists all observed run IDs; mixed strategy versions have `bot_version: null`. Unobserved historical
hands never inherit the running version. Malformed/truncated decision lines are skipped with a warning;
a missing final newline is repaired before appending a new event.

Compact review commands:

```sh
.venv/bin/python scripts/live_review.py --versions-only
.venv/bin/python scripts/live_review.py --version 6
.venv/bin/python scripts/live_review.py --run <run_id> --versions-only
```

`--run` includes a hand if any observed decision belongs to that run; it is not a per-run allocation of
the hand's winnings. Outcome summaries retain unknown historical hands as their own group.

For the current implementation status, checkpoints, and validation, read `HANDOFF.md`.
