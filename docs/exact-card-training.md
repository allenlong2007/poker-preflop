# Exact-card training and revealed-hand evidence

The new training path uses exact information states, not the legacy board textures or hand buckets.
It is implemented and tested; it is **not a fully trained solution of heads-up no-limit hold'em**.
The live engine uses an exact checkpoint only after promotion and only for sufficiently trained
states. Default serving now withholds uncovered actions and queues exact training; legacy fallback is an explicit research opt-in. See `docs/exact-coverage.md`. Overnight training alone cannot
cover the enormous full-game state space; this tabular method does not generalize to unseen states.

## Information and sizing

`preflop/exactgame.py` models heads-up no-limit betting from the blinds through showdown:

- Exact two-card private combinations and suits, exact visible flop/turn/river cards, position,
  effective stack, integer chip contributions, minimum raise, and complete action history.
- Opponent private cards and future board cards never enter a policy key. Hole-card order and
  simultaneous flop-card order are canonicalized without changing any card or suit; turn and
  river chronology stays distinct. Suits are not replaced by hand classes or texture labels.
- Legal checks, limps, folds, calls, repeated raises, full/short all-ins, and zero-sum accounting.
  The game assumes 0.5/1bb blinds, equal effective stacks of at least 1bb, no rake, no antes, and two
  players. Unequal chips beyond the effective stack cannot be won in this heads-up game.
- Default size choices: minimum legal size, all-in, and 25/50/75/100/150/200% of the pot after
  calling (raises add that increment above the call). The learned mixture over these amounts is
  specific to the exact state. This menu is still an **action abstraction**.
- `--fractions` changes that menu. `--exhaustive --chip-bb 0.01` enumerates **every legal chip total**
  at 0.01bb precision, including short all-ins. It is much more expensive. Continuous real-valued
  sizing is not claimed. Off-menu observed sizes stay exact; amounts finer than the configured
  chip precision cause a visible unsupported-state fallback, not snapping to a neighboring size.

## Training and restart

`preflop/exacttrain.py` implements outcome-sampling Monte Carlo CFR with signed importance-weighted
regrets and reach-weighted average strategies. This uses the established algorithm described by
[OpenSpiel's implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/outcome_sampling_mccfr.py).
It uses the existing NumPy/eval7 dependencies; no OpenSpiel or neural-training dependency is added.

SQLite/WAL transactions commit node statistics, iteration, RNG state, and aggregate counters together.
A single-writer lease prevents two trainers modifying the same candidate. Resume validates game
configuration, focus dataset, training source fingerprint, and Python/NumPy/eval7/SQLite versions.
An interrupted batch rolls back. Changing training code/configuration requires a new candidate or a
separately designed migration; do not edit/hot-swap these during an overnight run.

All generated candidates/logs/reports are Git-ignored under `output/exact/`.

```sh
# First run; repeats all exact private combinations across these stack depths.
.venv/bin/python scripts/train_exact.py train --hours 8 --max-gb 16

# Bounded experiment; a separate candidate for a different training configuration.
.venv/bin/python scripts/train_exact.py train \
  --path output/exact/study/candidate.sqlite --iterations 10000 --stacks 100

# Resume the same candidate with its saved RNG/configuration.
.venv/bin/python scripts/train_exact.py train --hours 8 --max-gb 16
.venv/bin/python scripts/train_exact.py status
```

Hours/iterations are finite budgets checked between atomic batches. Training stops earlier if its
candidate-plus-WAL limit or free-disk guard is reached. Snapshot copies use additional disk space;
the normal peak can include candidate, previous snapshot, and replacement snapshot. Numeric
instability raises an error and preserves the last committed batch rather than silently clipping
importance weights. SIGINT/SIGTERM finish the current batch and stop. A hard kill loses only the
uncommitted batch. Inspect status/log files; do not infer liveness from an old PID alone.

The initial overnight process, actual command, PID, and output paths are recorded in `HANDOFF.md`.
It is one background worker with numerical library threads limited to one, at reduced CPU priority.
No paid compute, recurring scheduler, or automatic promotion is configured.

## Focus subgames and known limits

`train --focus output/pokernow/decisions.jsonl --max-focus 32` uses recent public states as a
curriculum (80% focus roots, 20% whole-game roots). Both players' private hands are freshly sampled;
the recorded hero cards and later revealed opponent cards are deliberately ignored. Each focus
root has its own namespace, so its artificial prior cannot contaminate full-game regrets.

Focus subgames currently start with **uniform compatible private ranges**, not the correct
posterior for an arbitrary prior betting line. Their policies are candidate/shadow-only; reports
that actually serve focus policies cannot pass automatic promotion. Validated posterior ranges
are a remaining requirement before using these policies live.

For wider generalization, a future system can add function approximation such as
[Deep CFR](https://proceedings.mlr.press/v97/brown19b.html). This implementation does not claim
neural generalization, a practical exhaustive traversal of full NLHE, or a full-game exploitability
certificate. The imperfect-information convergence test uses tractable Kuhn poker, not NLHE.

## Frozen evaluation and promotion

```sh
# Safe while a trainer is running; use a distinct immutable evaluation filename.
.venv/bin/python scripts/train_exact.py snapshot \
  --output output/exact/frozen-evaluation.sqlite
.venv/bin/python scripts/train_exact.py evaluate \
  --path output/exact/frozen-evaluation.sqlite --deals 2000 \
  --output output/exact/evaluation.json
.venv/bin/python scripts/train_exact.py promote \
  --path output/exact/frozen-evaluation.sqlite --report output/exact/evaluation.json
```

Evaluation holds a shared lease on a self-contained frozen snapshot, uses independent held-out RNG
seeds and duplicate deals with swapped seats, and never trains during the match. Mutable/WAL
candidates are rejected. Atomic snapshot/promotion replacement takes an exclusive destination
lease. Artifact hashes are streamed and checked before/after; promotion checks source, fallback
policy, and runtime provenance as well as the exact evaluated bytes.

Declared minimum promotion gates:

- At least 2,000 held-out duplicate deals.
- Exact policy used in at least 25% of candidate decisions, with at least 1,000 regret updates in
  each served state. Production cannot lower the threshold used in evaluation.
- A 95% lower bound above −5bb/100 against the frozen-run legacy baseline, and no illegal actions.
- No uniform-prior focus policies served; compatible candidate/report/code/runtime fingerprints.

These are empirical guardrails, not proof of equilibrium or superiority against every opponent.
A thin checkpoint normally fails coverage. `--min-updates 1` is available for exploratory evaluation
only and cannot authorize live promotion. A promoted artifact becomes `output/exact/active.sqlite`;
restart the server to load it. Live logs include exact lookup status/visits/root/config fingerprint
and the model hash. Absent/unseen/undertrained/incompatible policies now withhold actions by default and persist the exact training request. Legacy research mode explicitly opts into logged fallback behavior.

## Revealed opponent hands

Raw hand JSONL remains authoritative. After a hand finishes, `bot/exact_opponent.py` derives exact
per-action evidence from the opponent cards actually shown: board visible at that action, earlier
history, precise amounts and price, evaluator score (including ranks/kickers), suit/rank counts,
and exact one-card flush/straight completion cards. It validates all known cards and preserves
literal combos rather than replacing them with a category. No later turn/river enters an earlier
action's features. This is after-reveal analysis, never an earlier inference-time input.

The index `output/pokernow/exact_evidence.sqlite` is idempotent by hero/hand/opponent identity.
Index failures are logged after the raw archive is saved; it can be rebuilt without rewriting
historical hands:

```sh
.venv/bin/python -m bot.exact_opponent --rebuild
.venv/bin/python -m bot.exact_opponent --opponent PLAYER_ID
```

Only shown cards are labeled. Showdowns are selection biased and do not establish the opponent's
entire betting range. The exact evidence is available for review/future validated opponent models;
this change does not silently turn sparse revealed samples into a confident new live exploit.
