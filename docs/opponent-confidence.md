# Opponent confidence and calibration

Contract: an opponent read changes live play only after independent-hand uncertainty supports a practically meaningful action-frequency difference. Missing evidence keeps the baseline. Exact-card training/checkpoints are unchanged.

## Serving gates

- `bot/reads.py` retains descriptive profiles separately from serving profiles. Deduplicate by hero/hand ID; unrelated opponent identities cannot pool. Existing explicit user-approved aliases still apply.
- Each hand contributes one bounded mean per action/node. Multiple raises or streets do not inflate that node's independent sample count. The target is an equal-hand mean action frequency, rather than an action-weighted frequency.
- With `n` distinct hands, use Hoeffding radius `sqrt(log(2*128*n*(n+1)/0.05)/(2*n))`, clipped to [0,1]. Spending `alpha/(n*(n+1))` gives a conservative bound across repeated looks, with a fixed 128-channel budget. Registered action channels plus calibration and exact-jam allocation fit that budget. Coverage is **per opponent**, conditional on stable conditional frequencies across hands. It does not cover arbitrary drift or all opponents jointly.
- Require at least 40 hands, maximum interval width 0.35, and an interval excluding the fixed reference by more than 0.05. These conservative settings often need hundreds of hands; 40 alone never grants approval. Thresholds are declared in source and copied into decision logs.
- Move to the nearest probability vector inside the joint confidence box, rather than the observed extreme. Preflop compares with the fixed equilibrium frequencies; postflop compares with the fixed self-play reference (or documented range defaults). Reference sampling uncertainty is not estimated: these are fixed policy benchmarks, not claims about an unknown population mean.
- Downstream pot-odds and trapping paths consume only approved nodes. Pooled postflop ratios compare matching nodes with identical opportunity weights, avoiding mixture reversals. Selected-reveal trap nodes cannot bypass this gate.
- Shove overrides use a separate actual effective-stack commitment channel. A 60bb raise at 100bb is not a jam. Exact depth and preceding raise amounts have separate contexts. Contexts receive first-observation ranks and additional `1/(rank*(rank+1))` alpha spending. Conditional re-raise reach is withheld until a joint range model exists.
- Frequency confidence does **not** validate the heuristic that shoves contain the strongest hands, broad legacy action/texture abstractions, or an EV improvement. Exact-card solver policies remain governed by their existing promotion checks.

## Revealed cards and calibration

Raw exact revealed-card evidence remains stored and available. Opponents select what they reveal; increasing that sample cannot establish the bluff frequency of all unseen bets. Therefore live size scales stay 1, showdown strictness stays 0, population strictness is withheld, and `for_opponent()` returns the frozen self-play prior. Unknown opponents do not inherit another player's read.

Shadow reports fit the first half of independently revealed hands and validate on the later half. They require at least 40 training and 40 validation hands. Monotonically increasing hand numbers within a single source identity are required; chronology that cannot be verified abstains. A durable source/timestamp contract remains future work.

Training uses the first labeled bet per hand; validation averages all labeled bets into one loss per hand. Reports compare candidate/prior Brier loss, mean probability residual, and a bounded paired Brier-improvement interval. Brier measures both calibration and discrimination; this is not a pure calibration score or a guarantee on unshown hands. Even a positive interval stays shadow-only until representative labels and range validation exist. The first-half split remains a diagnostic if reports are repeated; it cannot authorize promotion or tune thresholds on validation data.

## Inspect and reproduce

```sh
.venv/bin/python -m bot.reads > output/pokernow/opponent_read_calibration.json
.venv/bin/python -m bot.reads OPPONENT_ID
.venv/bin/python -m pytest tests/test_read_confidence.py tests/test_bot.py -q
```

The JSON includes data/source hashes, per-node support, intervals, references, accepted/withheld reasons, and shadow metrics. Regular decision audit diagnostics include `adaptation.read_evidence` and a shove-specific gate report. Restart the bot server to load this code; no server restart or model promotion is performed automatically.

Acceptance evidence: sparse and duplicate samples abstain; repeated actions cluster by hand; supported synthetic rates move conservatively; unrelated actions cannot approve shoves; actual jams cannot transfer across depths; comparison mixtures match; future labels cannot enter fitting; unknown opponents reset; seeded null streams stay neutral across repeated looks. Tests validate mechanics and bounded synthetic cases, not live strength.


The subsequent exact-only serving change (`docs/exact-coverage.md`) bypasses these legacy adaptation overrides in default live mode. Opponent evidence and shadow calibration remain available; modifying a validated exact policy using opponent reads requires a future exact range/strategy validation path.
