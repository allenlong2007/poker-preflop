# Heads-Up Preflop Strategy (100bb, Button vs Big Blind)

A data-science look at heads-up no-limit hold'em preflop strategy. Stage 1 asks:
**if only all-in equity mattered, which hands should the Button raise, what size
should it open to, and what should the Big Blind 3-bet?**

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                 # equity engine vs known reference numbers
.venv/bin/python scripts/build_equity.py      # ~2 min -> data/ (already included)
.venv/bin/python scripts/solve.py             # ~10 s  -> output/ charts + CSVs
.venv/bin/python scripts/solve.py --open 3 --threebet 10   # try other sizes
```

## Layout

| Path | What it does |
|---|---|
| `preflop/cards.py` | 169 hand classes, 1,326 combos, 13x13 grid order |
| `preflop/equity.py` | eval7 Monte Carlo equity, equity vs random, 169x169 matrix with card removal |
| `preflop/solver.py` | BTN vs BB game tree solved with CFR+, plus EV and exploitability |
| `preflop/plots.py` | 13x13 range charts, line charts |
| `scripts/` | `build_equity.py` (step 1), `solve.py` (step 2) |
| `tests/` | AA vs random ~85%, AA vs KK ~82%, 22 vs AKo ~coin flip, combo counts |

## The model

```
BTN (SB, 0.5bb):  fold | raise to s          (never limp)
BB  (1bb):        fold | call | 3-bet to t    (t = 4s by default, 9bb vs 2.25)
BTN:              fold | call | all-in 100bb
BB:               fold | call
```

Any call goes straight to showdown and each player wins `pot x equity`.
That's the key simplification: no postflop betting, and position is worth nothing.
CFR+ finds the Nash equilibrium of this game. Exploitability is under 0.1 milli-bb
per hand, so the strategies are effectively exact *for this model*.

## Stage 1 results

| Open | BTN EV (bb/hand) | BTN raises | BB folds | BB calls | BB 3-bets |
|---|---|---|---|---|---|
| 2.0 | -0.119 | 68% | 0% | 79% | 21% |
| **2.25** | **-0.108** | **65%** | **0%** | **81%** | **19%** |
| 2.5 | -0.097 | 63% | 0% | 82% | 18% |
| 3.0 | -0.078 | 60% | 14% | 69% | 17% |
| 3.5 | -0.074 | 58% | 29% | 55% | 16% |
| 4.0 | -0.079 | 57% | 41% | 45% | 14% |

* **BTN at 2.25bb raises 65%** and folds weak offsuit hands and low unconnected
  suited hands (T2s, 9xs-and-below gappers, 87o, 76o...).
* **BB 3-bets ~19%**: pairs 22+, Ax down to A7o / A2s, KQ/KJ/KTs, plus suited
  connectors T9s/98s/87s. **BB never folds** to a 2.25bb open, because even 32o has
  about 32% equity and only needs 28% to call.
* **The equity-only model picks ~3.5bb as the best open, not 2.25bb.** Small opens
  give the BB a great price to see every flop, and nothing here punishes the BB
  for being out of position. Solvers that model postflop play favour 2–2.5bb, so
  the gap is a measure of how much postflop play and position matter. That is
  stage 2.

## Experiment 2: Big Blind defense vs a 2.25–2.50bb open

`scripts/bb_defense.py` (~6 min) solves every BTN open from 2.25 to 2.50bb in
0.05bb steps and tries every BB 3-bet size from 6 to 14bb (0.25bb steps), keeping
the one that earns the BB the most. It runs two models:

* **equity**: raw all-in equity, position is worth nothing.
* **position**: in pots that see a flop, the BTN realizes 110% of its equity and
  the BB 85% (`r_ip`, `r_oop` in `solve()`). These two numbers are illustrative
  assumptions, not measured.

| BTN open | Best 3-bet (equity) | Best 3-bet (position) | BB fold / call / 3-bet (position) |
|---|---|---|---|
| 2.25 | 8.0bb (3.6x) | 10.0bb (4.4x) | 5% / 77% / 18% |
| 2.30 | 8.0bb | 10.5bb | 6% / 76% / 18% |
| 2.35 | 8.25bb | 10.75bb | 9% / 73% / 18% |
| 2.40 | 8.5bb | 11.25bb | 10% / 73% / 17% |
| 2.45 | 8.5bb | 11.5bb | 10% / 73% / 17% |
| 2.50 | 8.75bb (3.5x) | 11.5bb (4.6x) | 13% / 70% / 17% |

* **The best 3-bet scales with the open: about 3.5x in the equity model and about 4.5x
  once the BB is penalized for being out of position.** Playing a bloated pot out of
  position is costly, so the BB 3-bets bigger to make the BTN fold more often
  (the BTN folds ~45% to the 3-bet at every size).
* **The EV curves are flat near the top.** Anything within about ±0.75bb of the
  best size costs under 1 milli-bb per hand (see `good_3bet_range_bb` in `summary.csv`).
* **The 3-bet range barely moves across 2.25–2.50.** It's a linear "value" range:
  pairs 55+, Ax suited down to A4s, offsuit aces down to A7o, broadway hands.
  What changes is the **fold** range. With the position penalty, the BB folds 5%
  vs 2.25bb and 13% vs 2.50bb: every 0.05bb on the open pushes the weakest
  offsuit hands (83o, 74o, 64o, 53o, 43o...) from call to fold.

Outputs in `output/bb_defense/`: `summary.csv`, `bb_ranges.csv` (every hand,
every size, both models), `ev_by_3bet_size.csv`, and range / EV charts.

## Next steps

1. Add equity realization (IP realizes > 100%, OOP < 100%) and compare to published solver charts.
2. Model the gap between this model and solver frequencies with hand features (suited, connected, high card).
3. Sweep stack depth (10 → 200bb) and 3-bet size.
4. Multiway: rank hands by equity vs 1–8 random opponents (87s rises, K9o falls).
