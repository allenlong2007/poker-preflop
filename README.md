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
| `preflop/solver.py` | BTN vs BB preflop game tree solved with CFR+ (with equity realization), EV, exploitability |
| `preflop/sizing.py` | Parallel open-size / 3-bet-size searches |
| `preflop/realization.py` | Per-hand playability bonuses (experiment 5) |
| `preflop/postflop.py` | Rule-based postflop simulator: pot-odds calling, balanced bluffing |
| `preflop/flopgame.py` | Learned flop c-bet / check-raise game (CFR+ per board texture x hand bucket) |
| `preflop/turngame.py` | Learned turn game after a c-bet is called (shared check/bet/raise tree + CFR+ loop) |
| `preflop/rivergame.py` | Learned river game after the BB check-calls the turn |
| `preflop/sizedgame.py` | Flop / turn / river with several bet sizes per street (experiment 13) |
| `scripts/solver_compare.py` | Runs TexasSolver on 12 flops and compares it with the model (experiment 14) |
| `preflop/fullgame.py` | Flop + turn + river learned together with Monte Carlo CFR, 12 hand buckets, configurable sizes (experiments 15–16) |
| `preflop/plots.py` | Range grids, heatmaps, bar and line charts |
| `scripts/` | One script per experiment (named in each section below) |
| `tests/` | Equity reference numbers, combo counts, postflop hand-strength rules |

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

## Experiment 3: Sweeping the BB's out-of-position penalty

Only the ratio `r_ip / r_oop` changes the results, so `scripts/realization_sweep.py`
fixes the BTN at 1.0 and sweeps the BB's `r_oop` from 0.75 to 1.0. Experiment 2's
1.10 / 0.85 "position model" is r_oop ≈ 0.77 on this scale.

![realization sweep](output/realization_sweep/realization_sweep.png)

| r_oop | Best 3-bet vs 2.25 | BB fold vs 2.25 | Best 3-bet vs 2.50 | BB fold vs 2.50 |
|---|---|---|---|---|
| 1.00 | 8.0bb (3.6x) | 0% | 8.75bb (3.5x) | 0% |
| 0.90 | 8.75bb (3.9x) | 0% | 9.75bb (3.9x) | 5% |
| 0.80 | 9.75bb (4.3x) | 4% | 11.0bb (4.4x) | 11% |
| 0.75 | 10.5bb (4.7x) | 5% | 11.75bb (4.7x) | 15% |

* **The best 3-bet size grows almost linearly as the penalty grows**, from about 3.5x
  the open with no penalty to about 4.7x at r_oop 0.75.
* **Folding starts only once the penalty is real.** Against 2.25bb the BB keeps
  defending everything until r_oop drops below ~0.86. Against 2.50bb, folds start
  at ~0.97.
* **The 3-bet frequency falls slightly** (20% → 17%): a bigger size means a tighter range.
  The jagged lines come from the 0.25bb size grid.

## Experiment 4: The Button chooses its open size too

`scripts/sizing_game.py` lets the Button choose an open from 2.0 to 4.0bb, knowing the
BB will answer with its best 3-bet size (2.5x–6x the open). Each player uses one size
for every hand.

![sizing game](output/sizing_game/btn_ev_by_open_size.png)

| r_oop | BTN best open | BB 3-bets to | BTN raises | BB fold / call / 3-bet | Cost of opening 2.2bb |
|---|---|---|---|---|---|
| 1.00 | 3.4bb | 11.9bb (3.5x) | 58% | 25% / 59% / 17% | 3.7bb/100 |
| 0.90 | 3.2bb | 12.4bb (3.9x) | 67% | 28% / 56% / 17% | 3.4bb/100 |
| 0.80 | 3.1bb | 14.3bb (4.6x) | 79% | 29% / 55% / 17% | 3.1bb/100 |
| 0.75 | 3.1bb | 15.5bb (5.0x) | 86% | 29% / 55% / 16% | 3.3bb/100 |

* **The Button's best open stays at 3.1–3.4bb whatever the penalty.** Penalizing the BB
  makes the Button profitable and lets it open far more hands (58% → 86%), but the best
  size only drops from 3.4 to 3.1bb.
* **So this model still can't produce 2.25bb opens.** Small opens are what solvers that
  play the full game recommend. The model is missing whatever makes small opens work:
  postflop betting, hand-by-hand realization (suited/connected hands realize more), and
  mixing sizes by hand.
* **The BB's answer to a bigger open is a bigger 3-bet**, reaching 5x the open at r_oop 0.75.

## Experiment 5: Hand-by-hand realization — where 2.25bb appears

`scripts/hand_realization.py` adds two kinds of per-hand realization on top of the
BB's position penalty (r_oop = 0.85), then reruns the open-size game (opens
1.75–3.5bb every 0.05bb, BB best-responds with its 3-bet size):

* **Playability (k):** suited, connected and broadway hands realize more
  (`preflop/realization.py`, hand-picked bonuses).
* **Strength (alpha):** the BTN's pot share becomes `E^a / (E^a + 0.85 (1-E)^a)`.
  With a > 1, strong hands realize more than their equity and weak hands less,
  because after the flop strong hands win bigger pots and weak hands fold. At a = 1.5,
  a 30%-equity hand realizes about 73% of its equity and a 60% hand about 108%.

![EV by open size](output/hand_realization/btn_ev_by_open_alpha.png)

| Model | BTN best open | Opens within 1 mbb of best | Cost of 2.25 (bb/100) | BB 3-bet vs 2.25 |
|---|---|---|---|---|
| position only | 3.10bb | 2.95–3.25 | 2.84 | 9.6bb |
| playability k=2 | 3.25bb | 3.05–3.40 | 2.79 | 9.0bb |
| strength a=1.25 | 2.65bb | 2.50–2.80 | 0.78 | 7.9bb |
| strength a=1.4 | 2.45bb | 2.35–2.60 | 0.20 | 7.3bb |
| **strength a=1.5** | **2.35bb** | **2.25–2.50** | **0.06** | **7.3bb** |
| **strength a=1.6** | **2.25bb** | **2.15–2.35** | **0.00** | **6.75bb** |
| strength a=1.75 | 2.10bb | 2.00–2.25 | 0.09 | 6.75bb |
| strength a=2.0 | 1.95bb | 1.85–2.05 | 0.44 | 6.2bb |
| both a=1.5 k=2 | 2.40bb | 2.25–2.50 | 0.09 | 7.3bb |

* **Playability bonuses don't explain small opens.** They help both players' good
  hands, so they mostly cancel out, and the best open stays above 3bb.
* **The strength effect does.** At a ≈ 1.5–1.6, a 2.25bb open is the best size (or
  within 0.06bb/100 of it). The more postflop play rewards strong hands and punishes
  weak ones, the smaller the best open: a smoothly falling curve from 3.1bb to 1.95bb.
* **Why:** a small open gets called by many weak BB hands that will underperform
  after the flop. The BTN *wants* those calls, so it keeps the price low.
* At a = 1.5 vs a 2.25bb open, the BTN raises 71%. The BB folds 26%, calls 53%,
  and 3-bets 21% to ~7.3bb (about 3.25x), a smaller 3-bet than in the equity-only models.

Caveat: alpha is a single knob chosen to match a known answer, not fitted to
data. Calibrating it against real solver EVs is the next step.

## Experiment 6: Calibrating realization, then measuring it by simulation

### 6a. Calibration to published solver numbers (`scripts/calibrate.py`)

Targets for the BB vs a ~2.5bb open at 100bb, from the
[Preflop Wizard heads-up guide](https://www.preflopwizard.app/blog/poker-heads-up-strategy):
fold 30–35%, 3-bet 15–20%, 3-bet size 9–10bb. This is a weak source: an aggregate of
unnamed solver runs that include limping, which this model doesn't allow.

A grid search over α (1.0–2.0) and r_oop (0.70–1.00) finds **α = 1.4, r_oop = 0.75**
fits every target (fold 31%, 3-bet 21% to 9bb). α is well pinned down (1.3–1.5);
r_oop less so (0.70–0.80 all fit).

![calibration error](output/calibration/calibration_error.png)

**Out-of-sample check:** the open size was not a target, yet the calibrated model's
best BTN open is **2.45bb** (2.35–2.6 within 1 milli-bb), inside the 2–2.5bb that
solvers use. Opening 2.25bb instead costs only 0.19bb/100.

### 6b. Measuring realization with simple postflop rules (`scripts/simulate_realization.py`)

`preflop/postflop.py` plays hands out from the flop. Both players follow the same rules:

| Strength (from the cards each player sees) | First to act / checked to | Facing a bet |
|---|---|---|
| **Strong**: two pair+ that improves the board, overpair, top pair with T+ kicker | bet 2/3 pot | call |
| **Medium**: any other pair using a hole card, or a flush draw / open-ended straight draw (not on the river) | check | call |
| **Weak**: everything else | bluff 1/3 of the time, else check | fold |

The BB acts first every street; one bet per street at most and no raises. Hands are dealt
from the model's BTN opening range vs BB calling range at a 2.25bb open, 1.4M hands
per round. Then α and r_oop are fitted to the results, the preflop game is re-solved
with them, and the loop repeats until they stop moving (2 rounds).

| | α | r_oop | BTN best open |
|---|---|---|---|
| Calibrated to solver numbers (6a) | 1.40 | 0.75 | 2.45bb |
| **Simulated from simple rules (6b)** | **1.36** | **0.80** | **2.45bb** (2.35–2.65) |

On average the BTN wins 63% of the flop pot with 55% equity.

![realization fit](output/simulated_realization/realization_fit.png)

Per-hand realization (realized share ÷ raw equity):

* **BTN (in position):** big pairs realize the most (AA 2.7x, KK 2.3x, small pairs ~1.4x,
  because the rules pay them off with any pair). Suited connectors 1.3–1.4x.
  Offsuit weak aces and kings only 0.8–0.9x.
* **BB (out of position):** offsuit trash (A2o–A5o, K2o–K6o, Q2o–Q6o) realizes only
  ~0.6–0.7 of its equity; suited connectors and broadway offsuit ~1.0–1.2.

![BTN realization](output/simulated_realization/realization_btn.png)
![BB realization](output/simulated_realization/realization_bb.png)

### 6c. How much do the rules matter? (`scripts/rule_sensitivity.py`)

| Rules | α | r_oop | BTN best open |
|---|---|---|---|
| baseline (bluff 1/3, bet 2/3 pot) | 1.35 | 0.78 | 2.50bb |
| never bluff | 1.70 | 0.95 | 2.25bb |
| bluff 2/3 of the time | 1.60 | 0.54 | 2.70bb |
| bet 1/2 pot | 1.05 | 0.89 | 3.10bb |
| bet full pot | 2.12 | 0.50* | 2.10bb |

\* hits the edge of the fit grid.

**Bet size is the biggest driver.** The rules call with any pair whatever the bet size,
so bigger bets just extract more from strong hands, making α larger and the best open
smaller. That the 2/3-pot baseline matches the solver calibration is encouraging,
but partly luck of the rule choice. The obvious next step is calling based on pot odds.

### 6d. Pot-odds calling and balanced bluffing

The 6b rules had two flaws: medium hands called any bet regardless of price, and
bettors bluffed a fixed 1/3 of their weak hands. `preflop/postflop.py` now defaults to:

* **Pot-odds calling:** facing a bet, call if your chance of beating the *bettor* is at least
  `bet / (pot + 2·bet)` (28.6% vs 2/3 pot). Strong hands always call. The chance comes from a
  learned table: hand strength (equity vs a random hand on this board, eval7 Monte Carlo)
  → how often hands that strong actually beat a bettor on that street. Example: on the river,
  a hand that beats 70% of random hands beats a bettor only ~30% of the time.
* **Balanced bluffing:** in each spot (BB leading / BTN after a check, by street), weak hands bluff
  just often enough that bluffs are `bet / (pot + 2·bet)` of all bets, the mix that leaves a
  caller indifferent. Learned rates are 4–9% of weak hands on the flop and 16–18% on the river.

Both are learned from two warm-up batches before the main batch.
(Pot-odds calling with the old fixed 1/3 bluffing ran away to α ≈ 2.9: callers adapt but
bettors don't, and the BTN gets more bluffing spots.)

| Rules | α | r_oop | BTN realizes / equity | BTN best open |
|---|---|---|---|---|
| **pot odds + balanced, 2/3 pot** | **1.25** | **0.92** | 62% / 58% | **2.75bb** (2.6–2.85) |
| pot odds + balanced, 1/2 pot | 1.13 | 0.93 | 61% / 58% | 3.00bb |
| pot odds + balanced, full pot | 1.66 | 0.88 | 67% / 58% | 2.25bb |
| pot odds + fixed bluff 1/3 | 2.91 | 0.69 | 80% / 58% | 1.75bb |
| old rules (call any pair, bluff 1/3) | 1.45 | 0.85 | 67% / 58% | 2.40bb |

Smarter rules give a smaller α (1.25) and a less extreme position penalty, so the best
open moves up to ~2.75bb, and 2.25bb costs 1.0bb/100. Bet size still matters, but less
(open 2.25–3.0bb across 1/2–full pot, vs 2.1–3.1bb with the old rules).

## Experiment 7: The Big Blind vs different raise sizes, preflop and postflop

`scripts/bb_study.py`. For each BTN open (2.0–3.5bb), realization is re-measured *at that
size* (solve → simulate with pot-odds / balanced rules → refit, 2 rounds), then the BB's
preflop and postflop play is summarized. α stays 1.26–1.36 at every size, so in this
range the open size barely changes how well hands realize.

### Preflop

![BB preflop by open](output/bb_study/bb_preflop_by_open.png)

| BTN open | BB fold | call | 3-bet (to) | Fold limit (1 − MDF) |
|---|---|---|---|---|
| 2.0 | 2% | 74% | 23% (6.5bb) | 50% |
| 2.25 | 10% | 69% | 21% (7.3bb) | 54% |
| 2.5 | 24% | 55% | 20% (8.1bb) | 57% |
| 2.75 | 31% | 50% | 19% (9.6bb) | 60% |
| 3.0 | 39% | 42% | 19% (9.75bb) | 63% |
| 3.5 | 47% | 35% | 18% (11.4bb) | 67% |

* **Raise size mostly changes the BB's folds, not its 3-bets.** Each +0.25bb on the open moves
  ~5–10% of hands from call to fold, while the 3-bet share drifts only from 23% to 18%.
* **The best 3-bet stays ~3.25x the open** at every size.
* **The BB always defends more than MDF requires** (MDF = 1.5 / (open + 1): the BTN risks
  open − 0.5 to win 1.5). Calling has real value because the BB's hand still realizes ~90%
  of its equity out of position.
* Vs 2.25bb the BB folds only 11 hands (72o, 93o, 83o-style trash). Vs 3.0bb it folds most
  weak offsuit hands and low suited gappers (T2s–T4s, 92s–95s, 82s–85s, 72s–74s, 62s–64s, 52s–54s).

![BB preflop vs 2.25](output/bb_study/bb_preflop_2.25.png)

### Postflop (BB-called pots, simulated)

![BB postflop by open](output/bb_study/bb_postflop_by_open.png)
![How pots end](output/bb_study/bb_pot_endings_by_open.png)

* **The BB realizes ~90% of its equity** (0.89–0.91) at every open size.
* **Called pots end the same way regardless of open size:** BB folds ~25%, BTN folds 17–21%,
  showdown 54–58%.
* **The BB leads more as the open grows** (flop leads 8% → 12%). Its calling range is tighter and
  stronger, so it flops more strong hands. River leads are steady at ~21%.
* **The BB folds 55–75% when the BTN bets.** That's correct against these rules: a balanced
  BTN betting range is ~71% value, so folding weak hands is right. Real players c-bet much
  more widely, so the BB should fold less.
* **BB bluffs work 54–61% of the time** (more often vs bigger opens).
* **Calling beats folding by +0.44 to +0.56bb per called hand.** The BB loses ~0.5bb on
  average postflop, but folding loses 1bb.

![BB call profit vs 2.25](output/bb_study/bb_call_profit_2.25.png)

Per hand (vs 2.25bb): **broadway offsuit hands are the most profitable calls** (QJo +1.8, KTo +1.8,
QTo +1.5) along with high suited hands (K9s +1.4, Q9s +1.2). Weak offsuit hands barely break
even (T5o +0.02, T3o −0.01, J2o −0.03, T2o −0.16): those are the next folds as the open grows.
Vs 3.0bb the worst calls (Q2o −0.21, J5o −0.08, K2o −0.03) are already losing.

## Experiment 8: If the BB always checks, what does the Button c-bet, and when should the BB check-raise?

`scripts/cbet_study.py` + `preflop/flopgame.py`. BTN opens 2.25bb, BB calls (ranges from the
preflop solver). On the flop:

```
BB:  always checks
BTN: check | c-bet 1/3 pot
BB:  fold | call | check-raise to 3x the c-bet
BTN: fold | call                      (after a check-raise)
```

Turn and river use the pot-odds / balanced rules. **No flop decision is hand-written.** Each is
learned per (board texture, hand bucket) with CFR+ and Monte Carlo rollouts: every hand's flop
lines are all played to the end, the value of each option is averaged per texture × bucket, and
options that would have done better get played more next round (60 rounds, then 800k hands).

* **Hand buckets:** two pair+, overpair, top pair T+ kicker, top pair weak kicker, weaker pair,
  draw (flush draw / open-ender), two overcards, air.
* **Board texture:** high card (A / K / Q-J / T-or-lower) × paired × suits (rainbow / two-tone /
  monotone) × connected (three ranks within a 5-card span).

### Overall

| BTN c-bets | BB folds / calls / check-raises vs a c-bet | BTN calls a check-raise |
|---|---|---|
| 58% of flops | 32% / 50% / 18% | 80% |

### The Button's c-bet range

![BTN c-bet heatmap](output/cbet_study/btn_cbet_heatmap.png)

* **Value and draws always bet:** two pair+ 99%, draws 96%, overpairs 80%, top pair 78–81%.
* **Weak pairs mostly check** (44%), keeping the pot small with showdown value.
* **Air bets half the time overall, but it depends on the board.** 77% on A-high flops, 62% K-high,
  51% Q/J-high, only 13% on T-high-or-lower. High boards favor the BTN's range (it has more
  big cards after raising preflop); low, connected boards favor the BB's wider calling range.
* **By texture:** c-bets 85% on A-high flops vs 43% on T-high-or-lower; 46% on connected
  boards and 44% on monotone boards vs ~60% elsewhere.

![BTN c-bet range](output/cbet_study/btn_cbet_range.png)

### When should the BB check-raise?

![BB check-raise heatmap](output/cbet_study/bb_xr_heatmap.png)

* **By hand:** two pair+ 97% (value), draws 81% (semi-bluff), top pair T+ kicker 67%, top pair
  weak kicker 31%. Weaker pairs and overcards mostly *call* (86% / 80%). Air folds 51% and
  check-raises as a bluff 7%.
* **By board, the BB check-raises most on boards that favor it:**
  * paired boards 27% vs 17% unpaired
  * T-high-or-lower 22% vs A-high 15%
  * top textures: paired low/middle boards and low connected boards (26–31%)
* **It check-raises least on A-high boards** (12–14%), where the BTN's range is strongest.
* **Texture changes which hands raise:**
  * top pair T+ kicker raises 80% on disconnected boards but only 22% on connected ones
    (too many better hands and draws around)
  * draws raise 94% on two-tone boards but only 20% on monotone ones (a flush draw on a
    monotone board is weaker)
  * air bluff-raises more on paired boards (18% vs 5%)

![BB vs c-bet by preflop hand](output/cbet_study/bb_vs_cbet_range.png)

![Texture effects](output/cbet_study/texture_effects.png)

## Experiment 9: The BB study with a Button that bets more

`python scripts/bb_study.py cbet` reruns experiment 7 with the learned flop from experiment 8:
the BB checks the flop, and the BTN's c-bets and the BB's fold / call / check-raise are
learned at every open size (~18 min on an idle machine; each size is saved as it finishes).

### Preflop

![BB preflop by open](output/bb_study_cbet/bb_preflop_by_open.png)

| BTN open | BB fold / call / 3-bet (BTN bets more) | rules-only BTN (exp. 7) | 3-bet to | BB realization (exp. 7) |
|---|---|---|---|---|
| 2.0 | 10% / 68% / 22% | 2% / 74% / 23% | 7.0bb (3.5x) | 0.82 (0.89) |
| 2.25 | 22% / 58% / 21% | 10% / 69% / 21% | 7.9bb (3.5x) | 0.83 (0.90) |
| 2.5 | 31% / 49% / 20% | 24% / 55% / 20% | 8.75bb (3.5x) | 0.83 (0.90) |
| 2.75 | 36% / 45% / 20% | 31% / 50% / 19% | 10.3bb (3.75x) | 0.83 (0.90) |
| 3.0 | 38% / 44% / 18% | 39% / 42% / 19% | 11.25bb (3.75x) | 0.84 (0.90) |
| 3.5 | 46% / 36% / 18% | 47% / 35% / 18% | 13.1bb (3.75x) | 0.85 (0.91) |

* **A Button that c-bets often cuts the BB's realization from ~0.90 to ~0.83** (α ≈ 1.35, r_oop 0.74–0.81).
* **The BB should fold more vs small opens:** +8 to +12 percentage points at 2.0–2.25bb. The gap
  closes by 3.0bb, where the BB's calling range is already tight.
* **The BB 3-bets bigger:** ~3.5–3.75x the open instead of ~3.25x. Taking the pot preflop is
  worth more when calling means facing frequent c-bets out of position.
* **The 3-bet share is unchanged** (22% → 18% as the open grows), and the BB still defends
  more than MDF at every size.
* **Calling still beats folding** by +0.43 to +0.49bb per called hand.

### Postflop

![BB vs c-bet by open](output/bb_study_cbet/bb_vs_cbet_by_open.png)

| BTN open | BTN c-bets | BB fold / call / check-raise vs c-bet | BTN calls a check-raise |
|---|---|---|---|
| 2.0 | 54% | 33% / 43% / 24% | 83% |
| 2.25 | 53% | 30% / 46% / 24% | 82% |
| 2.5 | 50% | 29% / 49% / 23% | 84% |
| 2.75 | 48% | 28% / 49% / 24% | 81% |
| 3.0 | 46% | 29% / 47% / 24% | 81% |
| 3.5 | 44% | 28% / 48% / 25% | 82% |

* **The bigger the open, the less the BTN c-bets** (54% → 44%). The BB's calling range gets
  tighter and stronger, so fewer flops favor the BTN.
* **The BB's answer to a c-bet barely depends on the open:** fold ~28–33%, check-raise ~24%.
  It folds slightly less vs bigger opens, because its range is stronger.
* **Called pots:** BB folds ~28–33%, BTN folds ~17%, showdown ~51–54%. That's more BB folds and
  fewer showdowns than with the rules-only BTN.

![How pots end](output/bb_study_cbet/bb_pot_endings_by_open.png)

## Experiment 10: A Button that c-bets 75% on its boards and 50% on the BB's

`scripts/cbet_frequency_test.py`, 2.25bb open.

**Which boards favor whom:** for each flop texture, the BTN's average equity vs the BB's calling
range (measured from the simulated showdowns). The BTN averages 58%; textures above that
favor the BTN (46% of flops). A-high and K-high boards favor the BTN (61–64%). T-high-or-lower
unpaired boards favor the BB (52–55%), especially connected ones.

**The fixed strategy:** c-bet exactly 75% on BTN boards and 50% on BB boards, filling the frequency
in priority order: two pair+, overpair, top pair, then draws, then air (bluffs), then two
overcards, with weak pairs checking last. The BB (and the BTN's answer to a check-raise) learn
their best response.

![Scenario comparison](output/cbet_frequency/scenario_comparison.png)

| | BTN c-bets | BB fold / call / check-raise vs c-bet | BTN EV from the flop (bb/hand) |
|---|---|---|---|
| learned c-bets | 57% (69% BTN boards, 47% BB boards) | 34% / 47% / 19% | 0.705 |
| fixed 50–75% | 62% (75% / 50%) | 16% / 59% / 25% | 0.643 |

* **The learned BTN already c-bets ~69% / 47% by board.** The 50–75% rule gets the frequencies
  roughly right.
* **It still loses 0.06bb per flop, because of *which* hands bet.** The rule always bets top pair
  and draws, bluffs 78% of its air, and checks nearly all weak pairs and overcards. That betting
  range is polarized and bluff-heavy, so the BB stops folding: it **calls air 62% of the time**
  (36% vs the learned BTN), folds only 16% overall, and check-raises more (25%), especially
  with weak top pairs (44% vs 31%).
* **Lesson:** a good c-bet frequency is not enough. The hands inside it have to be balanced. The
  learned BTN mixes in some weak pairs and overcards and checks back some air.

![BB vs fixed c-bets by hand](output/cbet_frequency/bb_vs_fixed_by_bucket.png)

## Experiment 11: The BB called the c-bet. What now on the turn?

`scripts/turn_study.py` + `preflop/turngame.py`. After a flop c-bet and BB call, the turn is a
learned game:

```
BB:  check | lead 2/3 pot
 after a check:  BTN check | bet 2/3 pot  ->  BB fold | call | check-raise 3x  ->  BTN fold | call
 after a lead:   BTN fold | call | raise 3x  ->  BB fold | call
```

* **Inputs:** hands are weighted by how likely the flop c-bet and call were under each flop
  strategy from experiment 10. The river uses the rules.
* **Turn texture:** what the turn card did (blank / overcard / pairs the board / flush card /
  straight card) × dry or wet flop.
* **Learning:** decisions are learned with CFR+ per texture × hand bucket (40 rounds, then
  800k hands), with fixed seeds, so reruns give identical numbers.

What the BB holds on the turn after calling a learned c-bet: weaker pairs 42%, air 35% (floats
that missed), two pair+ 7%, top pair 8%, draws 7%.

### The BB's turn strategy (after learned flop c-bets)

![BB turn lines by hand](output/turn_study/bb_turn_lines_by_hand_learned.png)

| BB hand | Leads | vs a BTN bet: fold / call / check-raise |
|---|---|---|
| two pair+ | 42% | 0% / 7% / 93% |
| top pair, T+ kicker | 19% | 8% / 70% / 22% |
| top pair, weak kicker | 20% | 6% / 87% / 7% |
| weaker pair | 4% | 26% / 68% / 6% |
| draw | 9% | 68% / 14% / 18% |
| two overcards | 9% | 91% / 3% / 5% |
| air | 8% | 97% / 1% / 2% |

* **Mostly check.** The BB leads only 9% of turns; the BTN bets 70% when checked to.
* **Strong hands:** two pair+ splits between leading (42%) and check-raising (93% of the times
  it checks and faces a bet).
* **Pairs check-call.** Top pair calls 70–87%, weaker pairs 68%.
* **Draws mostly fold to a 2/3-pot bet (68%).** One card to come gives ~18–20% equity vs the
  28.6% needed, and the river rules pay off little extra, so there are few implied odds.
  Some draws check-raise as semi-bluffs (18%).
* **Air and overcards give up** (91–97% fold); air bluff-leads 8%.

![BB turn lines by turn card](output/turn_study/bb_turn_lines_by_card_learned.png)

**By turn card:**

* **Board pairs: the BB leads 27%** (vs 3–7% on other turns): two pair+ leads 71% and weak top
  pairs 79%. Turns that pair the board help the BB's range, which is full of middling pairs.
  The BTN bets less (58%), and the BB folds more when it does bet (60%).
* **Flush-completing turns:** the BB almost never leads (3%) but check-raises the most (13%).
  Draws fold 78% (their remaining draw is usually weaker once three suited cards are out), and
  weaker pairs fold more (36%).
* **Overcard turns:** the BTN bets the most (81%), since overcards hit its preflop-raising range;
  the BB leads only 5%.
* **Blank and straight-completing turns sit in between.** Straight cards give the BB's draws the
  most check-raises (22%).
* **Dry vs wet flop:** on dry flops draws fold far less (23% vs 76%) and check-raise 49%. The few
  draws there are usually strong (open-enders with overcards).

![BB fold heatmap](output/turn_study/bb_fold_heatmap_learned.png)

### After the fixed 50–75% flop c-bets

The BB arrives on the turn with more air (47% of hands, from floating the bluff-heavy c-bets)
and its turn strategy shifts:

* **Weaker pairs and draws defend more vs a turn bet:** pairs fold 14% (vs 26%); draws fold 37%
  (vs 68%) and call 45%. The BTN's turn betting range is weaker, carrying air from the flop.
* **More check-raises with top pair** (31% / 18% for strong / weak kicker, vs 22% / 7%) and on
  board-pairing turns (18% vs 11%).
* **Fewer leads on board-pairing turns** (14% vs 27%).

![BB check-raise heatmap](output/turn_study/bb_xr_heatmap_learned.png)

## Experiment 12: The BB called the turn too. What now on the river?

`scripts/river_study.py` + `preflop/rivergame.py`. The line: BTN c-bets the flop, BB calls; BB checks
the turn, BTN bets 2/3 pot, BB calls. The river uses the same learned tree as the turn (check / lead,
bet, fold / call / check-raise, raises; 2/3-pot bets, 3x raises).

* **Exact values:** every river line ends in a fold or a showdown, so each hand's values are exact.
* **Weighting:** hands are weighted by how likely they reach this river under the learned flop *and*
  turn strategies.
* **Texture and buckets:** river texture = what the river card did. River buckets match the flop's,
  with "draw" becoming **missed draw**.
* **Learning:** 60 CFR+ rounds, then 1.5M hands.

**What the BB holds on this river:** weaker pairs 73%, two pair+ 14%, top pair 10%, missed draws 2%.
After check-calling two streets, its range is mostly bluff-catchers.

### The BB's river strategy (after learned flop c-bets)

![BB river lines by hand](output/river_study/bb_river_lines_by_hand_learned.png)

| BB hand | Leads | vs a BTN bet: fold / call / check-raise | Wins when it calls |
|---|---|---|---|
| two pair+ | 15% | 0% / 27% / 73% | 72% |
| top pair (either kicker) | 1–2% | 2–3% / 96% / 2% | 46–48% |
| weaker pair | 2% | 54% / 42% / 5% | **28%** |
| missed draw | 6% | 94% / 0% / 6% | — |
| air | 6% | 80% / 0% / 20% | — |

* **Check and let the BTN bet.** The BB leads only 4% of rivers; the BTN bets 79% when checked to.
* **Weaker pairs are pure bluff-catchers at the exact break-even point.** Calling a 2/3-pot bet needs
  28.6% equity, and when they call they win **28%**. The learned BB mixes fold (54%) and call (42%),
  which is what equilibrium theory says an indifferent bluff-catcher should do.
* **Top pair always calls** (96%) and wins about half the time.
* **Two pair+ check-raises for value (73%),** except on rivers that pair the board (31%) or put a third
  suited card out (68%). There, two pair can be beaten by full houses and flushes, so it calls instead.
* **Missed draws give up (94% fold).** Their bluffing role goes to air, which check-raises as a bluff 20%
  of the time: hands with no showdown value are the ones that can afford to bluff.

### Benchmarks: MDF and bluff share

![BB defense vs MDF](output/river_study/defense_vs_mdf.png)

* **The BB defends 57% vs a river bet,** a little under the 60% MDF. That's fine here: the BTN's river
  bets aren't a pure bluff-or-nuts range, so the BB doesn't have to defend the full MDF to keep the
  BTN from betting any two cards.
* **34% of the BTN's river bets lose at showdown,** vs 28.6% for a perfectly balanced bettor. The excess
  is mostly thin value bets that run into the BB's better hands, not pure bluffs.

![BTN bluff share](output/river_study/btn_bluff_share.png)

### By river card

![BB river lines by river card](output/river_study/bb_river_lines_by_card_learned.png)

| River card | BB leads | BTN bets when checked to | BB fold / call / check-raise | BB wins when calling |
|---|---|---|---|---|
| blank | 4% | 83% | 43% / 43% / 14% | 33% |
| overcard | 1% | 95% | 45% / 43% / 12% | 31% |
| pairs the board | 8% | 58% | 43% / 49% / 9% | 42% |
| flush card | 4% | 87% | 43% / 42% / 15% | 37% |
| straight card | 1% | 83% | 42% / 46% / 12% | 33% |

* **Board-pairing rivers favor the BB:** the BTN bets least (58%), the BB leads most (8%; missed draws
  lead 13% as bluffs), and the BB's calls win most often (42%).
* **Overcard rivers favor the BTN:** it bets 95% and the BB almost never leads.
* **Flush rivers bring the most check-raises** (15%).

### After the fixed 50–75% flop c-bets

Nearly the same river strategy. The BB arrives with more missed draws (7% vs 2%) and air (2% vs 1%),
and that air leads the river as a bluff 26% of the time. Defense stays at 57%; the BTN's losing-bet
share drops to 32%.

## Experiment 13: Multiple bet sizes on every street

`scripts/bet_sizing_study.py` + `preflop/sizedgame.py`. Same line as experiments 8, 11 and 12
(2.25bb open; BB checks the flop and calls a c-bet; BB check-calls a turn bet), but every bet or
lead now picks a size:

| Street | Sizes (fraction of the pot) |
|---|---|
| flop c-bet | 1/3, 3/4 |
| turn bet / lead | 1/2, pot |
| river bet / lead | 1/2, pot, 1.5x (overbet) |

Raises stay at 3x the bet. Every response is learned separately for each size faced. The turn
and river remember earlier sizes, so each flop size (and each flop-size × turn-size pair) gets its
own strategy, since the pot differs.

**Validation:** `python scripts/bet_sizing_study.py single` runs the same code with one size per
street. It reproduces experiment 8 (c-bets 58.5%; BB fold / call / check-raise 34 / 47 / 19%),
experiment 11 (turn 52 / 38 / 10%; BB leads 26% when the turn pairs the board) and experiment 12
(river 43 / 44 / 13%; the BB's calls win 36%; 34% of BTN bets lose at showdown).

### The Button: big bets are polarized, small bets are merged

![River BTN sizes by hand](output/bet_sizing/river_btn_sizes_by_hand.png)

* **Flop (c-bets 61%):**
  * big (3/4) with two pair+ (73%) and draws (78%)
  * small (1/3) with overpairs and top pair (50–69%)
  * weak pairs mostly check (57%)
  * air splits check / small / big (46 / 28 / 27%)
* **Flop by board:** A-high boards are mostly small bets (57%) with few checks (11%). Boards that
  favor the BB get checked half the time.
* **Turn (bets 74% when checked to):** pot-size with two pair+ (72%) and draws (62%, semi-bluffs);
  half-pot with overpairs and top pair.
* **River (bets 79%):** the textbook polarized / merged split. Two pair+ bets pot or 1.5x 89% of
  the time (46% overbets). Overpairs and top pair mostly bet 1/2 pot (43–51%). The bluffs (air,
  missed draws) go big: pot or overbet 54–61%.

![Flop BTN sizes by high card](output/bet_sizing/flop_btn_sizes_by_high_card.png)

### The Big Blind: how it answers each size

![River BB vs size](output/bet_sizing/river_bb_vs_size.png)

| Street | BTN bet | BB fold / call / check-raise | BB defends | MDF |
|---|---|---|---|---|
| flop | 1/3 pot | 42% / 31% / 27% | 58% | 75% |
| flop | 3/4 pot | 59% / 25% / 16% | 41% | 57% |
| turn | 1/2 pot | 40% / 40% / 20% | 60% | 67% |
| turn | pot | 58% / 29% / 13% | 42% | 50% |
| river | 1/2 pot | 40% / 36% / 24% | 60% | 67% |
| river | pot | 54% / 32% / 14% | 46% | 50% |
| river | 1.5x pot | 63% / 29% / 8% | 37% | 40% |

* **Small bets get check-raised a lot.** Vs a 1/3-pot flop c-bet, the BB check-raises 27% (vs 19% when
  1/3 was the only size). The BTN bets its strongest hands big, so its small-bet range is capped,
  and the BB attacks it: top pair raises 61–76%, weaker pairs 33%, draws 88%. Vs a 3/4-pot c-bet
  it raises only its best hands and draws.
* **Bigger bets get more folds,** but the BB stays close to MDF on the turn and river and well
  below it on the flop. The BTN's big flop bets are mostly value and draws, so folding is right.
* **River bluff-catching follows the theory.** A balanced bettor's share of bluffs should rise
  with bet size (25% / 33% / 37.5% for 1/2 / pot / 1.5x). Here the share of BTN bets that lose
  at showdown rises the same way: 33% / 37% / 38%.
  * Vs a half-pot bet the BB's calls win 27%, just above the 25% break-even.
  * Vs pot and overbets they win 42% and 48%, well above break-even. The BB only calls big bets
    with strong hands.

![River bluff-catching](output/bet_sizing/river_bluff_catching.png)

* **Turn leads become block bets.** With a half-pot option, the BB leads the turn far more with
  medium hands: top pair 24–32% (mostly half-pot), two pair+ 35% (half split between sizes).
  It leads 31% when the turn pairs the board.
* **River leads are small.** Medium and weak hands lead 1/2 pot as blockers or bluffs: top pair
  weak kicker 21%, air 20%, missed draws 18%. Big leads are rare.

![Turn BB first to act](output/bet_sizing/turn_bb_first_by_hand.png)

![Flop BB check-raises by size](output/bet_sizing/flop_bb_xr_by_size_hand.png)

## Experiment 14: How close is the model to a real solver?

`scripts/solver_compare.py` solves 12 flops with **TexasSolver**
([bupticybee/TexasSolver](https://github.com/bupticybee/TexasSolver), AGPL-3.0, not included in this
repo: download the v0.2.0 macOS release into `tools/`; the binary runs under Rosetta on Apple
silicon). TexasSolver plays the real game on exact cards, with no buckets or rules. Each flop uses
the same preflop ranges, pot (4.5bb) and stacks (97.75bb) as experiment 13, and the same flop
decisions: the BB checks, the BTN checks or bets 1/3 or 3/4 pot, and the BB folds, calls or
raises (about 3x).

**The solver's later streets have to be smaller than the model's.** With leads, raises and one size
on the turn and river, each flop needed 10–16 GB of memory and ~3 min per iteration, which is
impractical on a laptop. The main comparison uses a **medium** tree: either player can bet the turn
(2/3 pot) and the river (3/4 pot), with no raises there. Every flop solved to 0.22–0.30% exploitability
(~3–13 min each, three at a time). A **simple** tree, where the BB can only call or fold on the turn
and river, is kept as a sensitivity check.

Flops: A♠7♦2♣, A♥K♦5♥, K♥8♦3♣, Q♦J♥4♦, T♦9♥6♣, 8♠6♠4♦, 7♣5♦3♥, J♥7♥2♥ (single-suit), K♦K♣4♠, 7♥7♦2♠, T♠5♣2♦, 9♣6♣5♦.

### Board level: the model gets the pattern right

![C-bet scatter](output/solver_compare/medium/cbet_scatter.png)

| Decision | Solver avg | Model avg | Mean abs. difference | Correlation across flops |
|---|---|---|---|---|
| BTN c-bets | 67% | 52% | 15 pts | **0.88** |
| BTN bets big (share of its bets) | 69% | 49% | 26 pts | 0.52 |
| BB folds vs 1/3-pot c-bet | 29% | 36% | 8 pts | 0.77 |
| BB check-raises vs 1/3-pot c-bet | 18% | 26% | 8 pts | 0.62 |
| BB folds vs 3/4-pot c-bet | 52% | 59% | 7 pts | **0.88** |
| BB check-raises vs 3/4-pot c-bet | 10% | 16% | 6 pts | **0.85** |

* **C-bet frequency tracks the solver closely across boards (r = 0.88).** Both bet nearly everything on
  A-high flops (solver 99–100%, model 89–95%) and bet least on low connected boards (solver 28–39%,
  model 26–30%). The BB's response to a big bet is also well matched (r = 0.85–0.88).
* **The model c-bets too little on K-high and paired boards.** On K♥8♦3♣ and K♦K♣4♠ the solver bets
  96–100%, mostly *small*; the model bets ~60%. Those boards look "neutral" to the model's texture
  features, but with these ranges the BTN has a big advantage there (the BB's calling range has few
  kings).
* **The model under-uses the big size,** especially on low boards. There the solver bets 3/4 pot with
  almost every bet (98% on 8♠6♠4♦ and 7♣5♦3♥) and checks the rest, a very polarized strategy.

### Hand level: where the abstraction breaks down

![BB check-raise by hand](output/solver_compare/medium/xr_by_bucket_small.png)

* **The BB never folds a pair or draw to a small c-bet in the solver** (0%), but the model folds top
  pair 12–22% and weak pairs 8% of the time.
* **The model check-raises medium hands far too often:**
  * top pair with a weak kicker 51% (solver 18%)
  * weaker pairs 31% (solver 9%)
  * draws 70% (solver 39%)

  The solver *calls* with them instead (pairs 82–91%, draws 61%). Only two pair+ (92–99%) and air (12%) match.
* **The solver's BTN protects its checking range:** it checks 35% of overpairs and 18–25% of top pairs.
  It also bluffs more: air bets 71% vs the model's 45%. The model bets nearly all of its strong
  hands and checks too much air.
* **Why:** the model's buckets lump very different hands together. "Top pair, T+ kicker" includes
  both AK on A-high and T9 on T-high, and "air" includes backdoor draws and overcards with
  blockers. The model's turn and river are also played by simple rules, so it can't plan the
  multi-street lines (check-call, slowplay, delayed bluffs) that make calling and checking
  valuable in the solver.

### Sensitivity: what the BB can do on later streets changes the flop

With the **simple** tree (the BB can never bet the turn or river), the solver c-bets **87%** of flops
instead of 67% and bets big 72% of the time. Giving the BB later-street betting rights shrinks the
BTN's flop betting by 20 points. Flop strategy depends heavily on what each player can do on later
streets, which is also why the model's rules-based turn and river limit its flop accuracy.

## Experiment 15: The whole hand learned together

`scripts/fullgame_study.py` + `preflop/fullgame.py`. Earlier experiments learned one street at a time and
played later streets with rules, so no player could plan multi-street lines. Here flop, turn and river are
**one game**, learned with external-sampling Monte Carlo CFR: for each sampled hand, the learning player
tries every option while the opponent's moves are sampled. A flop decision is valued by how the rest of
the hand actually plays out.

* **Hand buckets (12):** straight or better, set / trips, two pair, overpair, top pair (T+ kicker / weak
  kicker), middle pair, weak pair, strong draw, weak draw (gutshot / backdoor flush), overcards, air.
  On the river the draws become missed draws.
* **Line memory:** later streets know who bet last on the previous street and how deep the stacks are
  vs the pot.
* **Sizes:**
  * BB flop lead ("donk") 1/3 pot
  * BTN flop bets 1/3 / 3/4
  * turn leads 1/2, bets 1/2 / pot
  * river leads 1/2, bets 1/2 / pot / 1.5x
  * raises 3x
* **Training:** 250 batches × 60k hands per scenario, then each is evaluated exactly on the same 300k
  hands. Re-training the `donk` scenario with a different seed gave the same result to 0.001bb, so
  differences between scenarios aren't training noise.

Results are the BB's **net result for the whole hand, in bb per called pot**:

| Scenario | BB net |
|---|---|
| base: BB never donks | −0.869 |
| donk: BB may lead the flop 1/3 pot (learned) | −0.872 |
| donk_often: BB leads 50% of flops with every hand | **−1.156** |
| no_early_bluffs: BB never bluffs the flop or turn | −0.916 |
| slowplay 0% / 25% / 50% / 75% of strong hands | −0.882 / **−0.876** / −0.879 / −0.887 |

### Multi-street lines: check-calling with medium hands

![Check-call chain](output/fullgame/bb_check_call_chain.png)

How often the BB check-calls a bet on the flop, then the turn too, then the river too, by its flop hand
(the BTN has to bet each time):

| BB flop hand | Check-calls the flop | ... and the turn | ... and the river |
|---|---|---|---|
| top pair, weak kicker | 47% | **19%** | 4% |
| middle pair | 57% | **14%** | 2% |
| weak pair | 44% | 4% | 1% |
| top pair, T+ kicker | 23% | 8% | 1% |
| two pair | 18% | 5% | 2% |
| strong draw | 23% | 3% | 1% |

* **Check-calling twice is a medium-hand line,** mainly weak top pair and middle pair. They have enough
  showdown value to call twice but are too weak to raise.
* **Weak pairs usually stop after one call;** their turn calls drop to 4%.
* **Strong top pair and two pair call less** because they check-raise more often.
* **Calling down all three streets is rare** (≤4%): by the river, most medium hands either face a check or
  give up.

### Slowplaying: how often, and with what

![Slowplay sweep](output/fullgame/slowplay_sweep.png)

* **The best slowplay share is about 25%.** Forcing the BB to just call a flop bet with 0 / 25 / 50 / 75%
  of its strong hands (straight+, sets, two pair) gives −0.882 / −0.876 / −0.879 / −0.887bb. Never
  slowplaying costs 0.006bb per called pot and slowplaying 75% costs 0.011bb. The value curve is flat,
  so getting the mix roughly right matters more than the exact number.
* **The learned strategy agrees.** Facing a flop bet, the BB calls instead of raising with:
  * two pair 25% of the time
  * straights or better 26%
  * sets 18%, the least: sets are the hands that most want the pot to grow now
* **The BTN almost never slowplays:** it checks back strong hands on the flop only 4–7% of the time.

### Bluff timing: preflop and river vs flop and turn

![BB aggression by street](output/fullgame/bb_aggression_by_street_donk.png)

* **In the learned strategy, half of the BB's postflop bluffs happen on the flop** (check-raise and donk
  bluffs): flop 0.071, turn 0.031, river 0.041 bluffs per called pot, plus semi-bluffs with draws on the
  flop and turn.
* **Moving bluffs to the river costs the BB.** Banning weak-hand bets and raises on the flop and turn raises
  its river bluffs by 27% (0.041 → 0.052 per hand) but costs **0.044bb per called pot** (−0.872 → −0.916).
  Early bluffs work because the BTN's wide c-betting range folds a lot to check-raises.
* **Preflop bluffs barely exist in this model:** the BB 3-bets 21% of its hands, but only 0.45% of hands
  (22) are 3-bets with under 50% equity vs the BTN's opening range. The preflop model uses equity
  realization, not postflop play, so it has no reason to 3-bet bluff. Adding real 3-bet-bluff value
  would need a preflop model that plays out the postflop game.

### Donk bets: often, small, and on which boards?

![Donk heatmap](output/fullgame/donk_heatmap.png)

* **When it may choose, the BB donks rarely:**
  * A-high boards 4%, K-high 6%, Q/J-high 7%, T-high or lower 13%
  * paired boards 16%, single-suit boards 12%
* **It donks mostly with strong top pairs and sets on boards that favor it:** top pair T+ kicker donks
  52% on paired boards and 48% on single-suit boards; sets donk 34% on low boards.
* **Having the donk option is worth almost nothing:** −0.009 to +0.024bb per called pot by board, and
  −0.003 overall. That matches solver lore that donking adds little when the caller is out of position.

![Donk value by board](output/fullgame/donk_value_by_board.png)

* **Donking 50% of flops with every hand is a big leak: −0.29bb per called pot.** The BTN raises and floats
  a weak, capped donking range. Board by board, the cost (bb per 10 called pots):

| Board | Cost of donking 50% |
|---|---|
| A-high | −3.9 |
| K-high | −3.2 |
| Q/J-high | −2.8 |
| connected | −2.5 |
| T-high or lower | −2.1 |
| single-suit | −2.0 |

* **If the BB wants to donk often, low and single-suit boards are where it hurts least,** because those boards
  favor the BB's range. A-high and K-high boards, where the BTN's range is strongest, are the worst. On
  every texture, small frequent donking with the whole range still loses vs donking selectively.

## Experiment 16: Best sizes vs every open, the BTN's 2.25bb plan, and TexasSolver again

`scripts/open_size_study.py` (train / solve / report). The whole-hand model (experiment 15) is trained
at BTN opens of **2.0, 2.25, 2.5 and 3.0bb**, with preflop ranges from the preflop solver at each size.
It has a bigger sizing menu so it can pick its own sizes:

| Street | BB leads | BTN bets | Raises (either player) |
|---|---|---|---|
| flop | 1/3 or 3/4 pot (donk) | 1/3 or 3/4 pot | 2.5x or 4x |
| turn | 1/2 or pot | 1/2 or pot | 2.5x or 4x |
| river | 1/2 or pot | 1/2, pot or 1.5x | 2.5x or 4x |

The generalized model reproduces experiment 15 exactly when given its old sizes. Each open size took
16–23 min to train (250 batches × 60k hands) and was evaluated exactly on 150k hands.

### The BB vs each open size

| BTN open | BB preflop fold / call / 3-bet (to) | BB net per called pot | BTN bets the flop when checked to | BB vs 1/3 c-bet: fold / call / raise 2.5x / raise 4x |
|---|---|---|---|---|
| 2.0 | 1% / 76% / 23% (6.5) | −0.85 | 78% | 48 / 21 / 19 / 12% |
| 2.25 | 10% / 69% / 21% (7.3) | −0.85 | 75% | 46 / 19 / 23 / 12% |
| 2.5 | 19% / 61% / 20% (8.1) | −0.81 | 72% | 45 / 22 / 22 / 12% |
| 3.0 | 35% / 47% / 18% (9.75) | −0.84 | 69% | 40 / 26 / 24 / 10% |

![BB vs 1/3 c-bet by open](output/open_sizes/bb_vs_flop_13.png)

* **Bigger opens get a tighter BB, and its postflop results barely change** (−0.81 to −0.85bb per called
  pot). The adjustment happens preflop (folding 1% → 35%); afterwards the BB folds less to c-bets
  (48% → 40%) because its range is stronger, and the BTN c-bets less (78% → 69%).
* **Raise sizes depend on the street:**
  * **Flop:** the BB prefers the **small 2.5x check-raise** (19–24% vs a 1/3 c-bet, 12% vs a 3/4 c-bet),
    with 4x as a smaller, stronger part (10–12% / 5%).
  * **Turn:** vs a half-pot bet the **4x raise** is used more (10–11%) than 2.5x (7–10%).
  * **River:** vs a half-pot bet the BB raises almost only **4x** (15%, vs 2% at 2.5x), a polarized
    nuts-or-bluff raise. Vs pot and overbets it raises rarely, and small.
* **BB leads:**
  * flop donks are rare: 9–11%, mostly 1/3 pot (5–7%)
  * turn leads grow with the open: 12% → 18%, split evenly between 1/2 and pot
  * the river is where the BB leads most: ~31% (1/2 pot 18%, pot 13%)

### The BTN's 2.25bb plan: c-bet, turn barrel, river

![BTN flop c-bet by hand](output/open_sizes/btn_flop_cbet_by_hand.png)

* **Flop c-bet: 75% of flops** (1/3 pot 40%, 3/4 pot 36%).
  * **By board:** A-high flops are bet almost always (check 11%, mostly 3/4 pot); K-high mostly 1/3 pot
    (50%); low, connected and single-suit flops are checked 34–36% of the time.
  * **By hand:** two pair+ goes big (65–78% 3/4 pot). Overpairs, weak top pair and middle pair go small
    (55–56% 1/3 pot). Weak pairs and overcards check (65–66%). Air bets 86% (split small / big), and
    draws bet small (strong draws 52% at 1/3).
* **Turn barrel after a called c-bet:**
  * value keeps betting: straights 99%, sets 92%, two pair 84%, mostly pot-size
  * top pair barrels 66–73%
  * middle and weak pairs check (83–94%) to control the pot
  * draws semi-bluff: weak draws 81%, strong draws 56%
  * air barrels 42%

![BTN turn barrel](output/open_sizes/btn_turn_barrel.png)

* **River after a called barrel: the bluffs are the missed weak draws.**
  * value overbets: straights+ bet 100% (65% at 1.5x), sets 99% (49% at 1.5x), two pair 86%
  * top pair mostly checks (64–78%)
  * the main bluff is **missed gutshots / backdoors: they bet 83%, 54% as a 1.5x overbet**
  * pure air bets only 25% and missed strong draws 23%. Hands with a little showdown value check;
    the weakest missed draws bluff big.

![BTN river bets](output/open_sizes/btn_river_bets.png)

### The most probable lines (2.25bb)

Across all called pots, the most common outcomes are short:

| Line | Probability | BTN holds (strong / pair / draw / air) |
|---|---|---|
| BTN bets 3/4, BB folds | 19.0% | 10 / 26 / 18 / 46% |
| BTN bets 1/3, BB folds | 16.5% | 3 / 39 / 18 / 40% |
| checked down on every street | 4.1% | 0 / 62 / 8 / 31% |
| flop checked through, BTN bets pot on the turn, BB folds | 3.3% | 3 / 34 / 21 / 42% |
| BTN bets 1/3, BB raises 2.5x, BTN folds | 2.0% | 0 / 2 / 2 / 97% |

**When the BTN c-bets and barrels and gets called twice,** the river goes:
* **check-check** in 47% of these lines (most common: 3/4 then 1/2, 17%; 3/4 then pot, 16%)
* **a BTN 1.5x overbet** in 25% (the BB folds to it in 16%)

The BTN's river overbets hold 17% strong hands, 12% pairs, 38% missed draws and 33% air: mostly bluffs.
37% of the BTN's air ends with a flop c-bet the BB folds to.

### TexasSolver comparison (flop decisions, medium tree)

Solved at **2.25bb (12 flops)** and, new, **3.0bb (6 flops)** with that open's ranges (0.25–0.29%
exploitability, 2–8 min each, three at a time).

| Decision | 2.25bb: solver / model / correlation | 3.0bb: solver / model / correlation |
|---|---|---|
| BTN c-bets | 67% / 70% / **0.97** | 61% / 66% / 0.86 |
| BTN bets small (1/3) | 25% / 35% / 0.89 | 33% / 39% / **0.98** |
| BB folds vs 1/3 | 29% / 42% / 0.78 | 35% / 41% / 0.88 |
| BB raises vs 1/3 | 18% / 29% / 0.13 | 15% / 29% / **0.94** |
| BB folds vs 3/4 | 52% / 60% / 0.83 | 55% / 57% / 0.76 |
| BB calls vs 3/4 | 38% / 25% / **0.94** | 38% / 29% / 0.50 |

![C-bet scatter 2.25](output/open_sizes/solver_cbet_scatter_225.png)

* **Much closer than the street-by-street model.** C-bet frequency per flop correlates **0.97** with the
  solver at 2.25bb (0.88 in experiment 14), and the BTN's c-bet by hand is close: air 87% vs 76%,
  top pair 74–76% vs 75–82%, two pair+ 90–96% vs 100%.
* **The old "BB folds pairs" error is gone:** the model now folds pairs and draws 0–5% to a small
  c-bet, like the solver.
* **It tracks how the open size changes things.** On K♥8♦3♣ the solver's c-bet drops from 100% at
  2.25bb to 59% at 3.0bb (the BB's tighter calling range has more kings); the model goes 83% → 56%.
* **The remaining gap: the BB check-raises too much and calls too little.** It raises middle pair 54%
  and weak top pair 64% vs the solver's 7–18%. The pattern across flops matches at 3.0bb (r = 0.94),
  but the level is about twice the solver's. Likely causes (not yet tested):
  * the model's BTN bluffs air about 10 points more than the solver, which makes raising profitable
  * the model can raise a cheaper 2.5x, while the solver's raise is about 3x
  * the solver's tree has no turn or river raises, so a flop raise there carries less threat

![Check-raise by hand](output/open_sizes/solver_xr_by_bucket_225.png)

## Experiment 17: The BTN's 2.25bb ranges, and exploiting a BB that donks often

`scripts/donk_exploit_study.py`, using the 2.25bb whole-hand model from experiment 16 (same sizes).

### The BTN's opening and c-bet ranges

![BTN opening range](output/donk_exploit/btn_open_range.png)

* **The BTN opens 68% of hands at 2.25bb.**
* **It c-bets 75% of flops when the BB checks:** 39% at 1/3 pot, 35% at 3/4 pot (averaged over random flops).
  * **A-high flops: 89%. T-high-or-lower flops: 64%.**
* **Most hands c-bet 70–80%.** The exceptions check more:
  * **small pocket pairs** (22–66: 44–57%) keep showdown value
  * **big offsuit broadways** (AKo 55%, AQo 62%, AJo 63%) bet A-high flops ~85% but check low flops
    two-thirds of the time (32–33%): they have overcards and showdown value, not much to bet for
  * **overpairs bet more on low boards:** AA c-bets 95% on A-high and 89% on low flops, KK 77% / 87%

![BTN c-bet range](output/donk_exploit/btn_cbet_range.png)

Per-hand c-bet ranges on A-high and low flops, and the share bet big:
`btn_cbet_range_A-high.png`, `btn_cbet_range_T-high_or_lower.png`, `btn_cbet_big_share.png`,
data in `btn_ranges_225.csv`.

### Exploiting three frequent donkers

Each BB opponent donks 1/3 pot with fixed frequencies by hand. Everything else is re-learned, so the
BTN learns a best response:

| Opponent | Donk rate by hand (value / middle-weak pair / draw / air) | Donks the flop | Donks are value / pair / draw / air | BTN with equilibrium play | BTN best response | Gain (bb / 100 called pots) |
|---|---|---|---|---|---|---|
| value donker | 90 / 30 / 30 / 5% | 27% | 44 / 23 / 25 / 8% | +0.94 | +1.06 | **+12** |
| balanced donker | 80 / 30 / 50 / 25% | 39% | 27 / 16 / 28 / 29% | +0.93 | +1.00 | **+6** |
| bluffy donker | 60 / 30 / 60 / 50% | 49% | 16 / 12 / 27 / 45% | +0.97 | +1.16 | **+19** |

(Against an equilibrium BB, which donks ~9%, the BTN nets +0.85bb per called pot.)

![Exploit value](output/donk_exploit/exploit_value.png)

* **Donking often loses for the BB even when the BTN doesn't adjust:** the BTN's ordinary strategy
  already earns +0.09bb more per called pot.
* **The best response depends on what the donks contain:**

| BTN hand vs a 1/3-pot donk | vs equilibrium BB | vs value donker | vs balanced | vs bluffy donker |
|---|---|---|---|---|
| air: fold / raise | 45% / 26% | **92% / 4%** | 23% / 39% | **2% / 70%** |
| overcards: fold / raise | 10% / 35% | 72% / 6% | 13% / 22% | 1% / 50% |
| middle pair: raise | 50% | 35% | 48% | **85%** |
| sets: raise (rest call) | 88% | **48%** | 82% | 97% |
| top pair T+ kicker: raise | 73% | 85% | 86% | 95% |

* **Vs the value donker:** fold air and overcards (92% / 72%), and **slowplay sets** (raise only 48%; just
  call the rest so the donker keeps betting its strong hands). Top pair and two pair still raise.
* **Vs the bluffy donker:** **raise almost everything** (air 70%, middle pair 85%) and fold nothing.
  The donks are 45% air, so a raise wins the pot right away most of the time.
* **Vs the balanced donker:** close to equilibrium play, so there's less to gain (+6).

![BTN vs donks, bluffy](output/donk_exploit/btn_vs_donk_bluffy.png)

### Figuring out the donker during a session (Bayesian inference)

The BTN starts not knowing which donker it faces: equal prior on the three profiles. After every hand it
updates the posterior from what it can see:
* did the BB donk?
* if the BTN called or raised, did the BB fold, or what did it show down?

The likelihoods were estimated from 300k simulated hands per profile. Once one profile has ≥ 70%
posterior, the BTN switches to that profile's best response. 300 sessions × 400 hands per opponent and
strategy:

![Inference speed](output/donk_exploit/inference_speed.png)

| Opponent | Hands until the posterior on the truth hits 90% (median) | BTN static | BTN adaptive | adaptive + probe raises | BTN knows the truth |
|---|---|---|---|---|---|
| value donker | 35–45 | 101 | **106** | 100 | 106 |
| balanced donker | 83–92 | 99 | 99 | 104 | 101 |
| bluffy donker | 42–49 | 102 | **121** | 110 | 124 |

(BTN result in bb per 100 called pots; the standard error is about ±4.)

* **The donker is identified within ~40 hands, or ~90 for the balanced one.** The balanced profile sits
  between the other two, so it takes the most evidence. By 150 hands the posterior on the truth averages
  88–96%.
* **Adapting captures nearly all the exploit:** vs the bluffy donker the adaptive BTN earns 121 vs 124 for a
  BTN that knew all along, and vs the value donker 106 vs 106.
* **Probe raises didn't help.** Raising medium hands into donks early (40% of the time until 90% sure) did
  not identify the opponent any faster. The donk *frequency* and the showdowns already carry most of
  the information. It cost EV vs the value donker (−6) and the bluffy donker (−11 vs plain adaptive),
  because probing raises the wrong hands at the wrong time. The best response already raises a lot vs
  bluffy donkers; a separate "test raise" adds risk without much new information.

## Experiment 18: Donk sizes × open sizes, and the BTN as the short stack

`scripts/donk_size_stack_study.py` (`donk` / `stacks` / `report`).

### Part A: four donk sizes at four open sizes

At BTN opens of 2.0 / 2.25 / 2.5 / 3.0bb (pots of 4 / 4.5 / 5 / 6bb), the BB may donk **1/4, 1/2, 3/4 or
pot**; everything else matches experiment 16. Each open was trained for 250 batches (24–35 min).

**"What if" evaluation (new in `fullgame.py`):** for every board and BB hand, every flop option is played
out against the BTN's actual responses. That gives the value of donking each size vs checking in that spot.

| BTN open (pot) | BB donks (learned) | 1/4 | 1/2 | 3/4 | pot | Avg cost of donking vs checking, 1/4 → pot (bb / 100 called pots) |
|---|---|---|---|---|---|---|
| 2.0 (4bb) | 13% | 4.7% | 3.4% | 2.8% | 2.0% | −19 / −27 / −36 / −45 |
| 2.25 (4.5bb) | 14% | 5.2% | 3.2% | 3.1% | 2.1% | −20 / −31 / −41 / −49 |
| 2.5 (5bb) | 14% | 5.1% | 3.4% | 2.7% | 2.4% | −22 / −31 / −40 / −50 |
| 3.0 (6bb) | 15% | 6.5% | 3.7% | 2.7% | 2.3% | −26 / −37 / −48 / −55 |

* **Checking is the better default at every open size and on every board type.** Averaged over all of
  the BB's hands, donking costs more the bigger it is (−0.2bb per called pot at 1/4 pot, −0.5 at pot)
  and more in bigger pots (3.0bb opens).
* **The smallest donk (1/4 pot) is the least bad and the most used.** Bigger pots push the BB to donk
  slightly more (13% → 15%), mostly at 1/4.
* **Donking is break-even (within 1bb / 100 of checking) in about a quarter of spots** (22–28%), mainly:
  * **paired boards with top pair:** break-even 71% of the time, donking mostly 1/4–1/2 pot
  * **connected boards with two pair+:** 66%, sizes split toward 3/4–pot
  * **K-high boards with two pair+:** 51%
  * **low boards (T-high or lower) with draws or air:** 35–38%
  * **A-high boards almost never:** ≤ 15%
* **The BTN folds more vs bigger donks** (11–13% vs 1/4, 46–48% vs pot) **and raises the small donk a lot**
  (53–62% vs 1/4 pot): a tiny donk invites a raise.
* **Caveat:** in some spots the learned average strategy donks more than the what-if values justify
  (e.g. two pair+ on A-high: donks 23%, but break-even in only 9% of those spots). Those cells look
  under-converged; the what-if values, measured against the BTN's actual responses, are the safer guide.

![Where the BB donks](output/donk_sizes_stacks/donk_used_2_25.png)

### Part B: the BTN as the short stack

**Only the effective (smaller) stack matters heads-up.** A 50bb BTN vs a 150bb BB plays exactly a 50bb game,
because chips beyond the shorter stack can't be won or lost. The study compares 25 / 50 / 100bb effective
at a 2.25bb open (100bb = experiment 16's model).

| | 25bb | 50bb | 100bb |
|---|---|---|---|
| stack-to-pot ratio after the call | 5.1 | 10.6 | 21.7 |
| BTN's best open size (preflop model) | **2.25bb** | 2.75bb | 2.75bb |
| BTN opens (at 2.25bb) | 66% | 71% | 68% |
| BB fold / call / 3-bet | 11 / 68 / 21% | 9 / 73 / 18% | 10 / 69 / 21% |
| BB 3-bet size | **13.5bb (near all-in)** | 7.3bb | 7.3bb |
| BTN folds / jams vs a 3-bet | **65% / 11%** | 49% / 12% | 42% / 8% |
| BB calls a BTN jam | **100%** | 38% | 20% |
| BTN c-bets (1/3 / 3/4) | **62%** (41 / 21) | 68% (35 / 33) | 75% (40 / 36) |
| BTN barrels the turn after a called c-bet | 58% | 53% | 51% |
| BTN bets the river after a called barrel | 60% | 52% | 47% |
| BB vs 1/3 c-bet: fold / call / raise | 48 / 19 / 33% | 47 / 23 / 30% | 46 / 19 / 35% |
| stacks all-in before the river | **3.0%** | 0.6% | ~0% |
| BB net per called pot | −0.74 | −0.74 | −0.85 |

![Best open by stack](output/donk_sizes_stacks/best_open_by_stack.png)

* **Preflop at 25bb, everything becomes "all-in or fold" after a 3-bet.** The BB's best 3-bet is 13.5bb
  (over half the stack); the BTN folds 65% to it, jams 11%, and the BB always calls the jam. The BTN's
  best open shrinks to 2.25bb (2.0–2.25 are about equal), since raises are a bigger share of the stack.
* **The BTN's postflop edge shrinks with the stack:** the BB loses 0.74bb per called pot at 25–50bb vs 0.85
  at 100bb. There's less room to use position over three streets.
* **Short-stacked, the BTN c-bets less and smaller** (62%, mostly 1/3 pot), **but follows through more**
  (turn barrel 58%, river bet after a barrel 60%): with a low stack-to-pot ratio, a hand that bets the flop
  is often committed.
* **By hand at 25bb:**
  * **strong hands slowplay more:** two pair+ c-bets only 61–72% (vs 92–95% at 100bb), since the stack is
    easy to get in later
  * **top pairs bet almost always** (91–94%)
  * **weak pairs check** (13%)
  * **the BB check-raises top pair 92–93% vs a small c-bet**, effectively getting it in

![BTN c-bet by stack](output/donk_sizes_stacks/stacks_btn_cbet.png)

## Experiment 19: The BB's ranges and sizes when stacks are short, by BTN open size

`scripts/bb_short_stack_study.py` (`preflop` / `postflop` / `report`).

Only the effective (smaller) stack matters heads-up, so a 25bb BB vs a 175bb BTN plays exactly the same
game as a 25bb BTN vs a 175bb BB (experiment 18). This study covers **20–100bb effective** against BTN opens
of **2.0–3.0bb**.

**What's new:**
* **All-in 3-bets.** The BB can now 3-bet all-in, in addition to 2.5x–6x the open. Calling an all-in
  3-bet means no postflop play (`solver.py`).
* **Realization by stack.** The preflop model's BB realization (0.915) was fit at 100bb. The whole-hand
  models from experiment 18 show how much better the BB does with shorter stacks, so r_oop is scaled by that
  ratio: **1.00 at 20–25bb, 0.96 at 50bb, 0.92 at 100bb** (capped at 1.0, the BTN's level). With shallow
  stacks there are fewer streets of betting left, so position is worth less.

### Preflop: fold, call, 3-bet and the 3-bet size

![BB vs open size, by stack](output/bb_short_stack/bb_freq_by_open.png)

| Stack | BB's best 3-bet | BB folds vs 2.0 / 2.5 / 3.0 open | BB 3-bets | BTN folds to the 3-bet |
|---|---|---|---|---|
| 20bb | **all-in** vs every open | 0 / 18 / 38% | 24–25% | 55–70% |
| 25bb | **all-in** vs every open | 0 / 16 / 37% | 22–23% | 64–76% |
| 30bb | **all-in** vs every open | 0 / 15 / 37% | 21–22% | 70–80% |
| 40bb | 6.5bb vs 2.0; **all-in** vs 2.25+ | 0 / 16 / 36% | 18–20% | 50% / 77–83% |
| 50bb | 3.25x (6.5–8.1bb) vs 2.0–2.5; **all-in** vs 2.75+ | 0 / 16 / 35% | 17–19% | 50% / 82–84% |
| 75–100bb | ~3.25x the open (6.5bb vs 2.0, 9.75bb vs 3.0) | 0–1 / 18–19 / 35% | 18–23% | 41–48% |

![Best 3-bet size](output/bb_short_stack/bb_3bet_size.png)

* **The fold rate depends on the open size, not the stack.** The BB folds almost nothing vs a 2.0bb open and
  ~35–38% vs a 3.0bb open at every depth. A bigger open lays the BB a worse price, but that price is the same
  at any stack.
* **At 30bb and below, every 3-bet is all-in.** A jam earns the BB 5–15 milli-bb per hand more than the best
  smaller 3-bet (only +2 at 20bb vs a 2.75–3.0bb open). A small 3-bet commits a third to half of the
  stack anyway, and the BTN can jam over it.
* **At 40–50bb it depends on the open:** a small 3-bet vs small opens, all-in vs big opens (2.25+ at 40bb,
  2.75+ at 50bb). Vs a bigger open, a 3.25x 3-bet is already a large share of the stack. Near the switch
  point the options are within a few milli-bb of each other, so either is fine.
* **At 75bb and above, an all-in 3-bet is a clear mistake** (−38 to −90 milli-bb vs a small 3-bet). Several
  3-bet sizes are about equally good: anything from ~3x to ~3.75x the open is within 2 milli-bb of the best.
* **A wider BB range costs the BTN more when stacks are short:** the BTN folds 64–80% to an all-in 3-bet vs
  41–48% to a small 3-bet at 100bb.

![All-in vs small 3-bet](output/bb_short_stack/bb_jam_vs_small.png)

**What the ranges look like (25bb):** the all-in range is the strong hands plus **every pair, suited aces,
and suited connectors and one-gappers** (T9s–54s, 97s–64s). Offsuit hands like A2o–A8o, KJo, K9o and Q8o
**call**. They have showdown value but aren't good enough to jam.

![BB vs 2.25 open, 25bb](output/bb_short_stack/bb_range_25bb_2_25.png)

| BB vs open | 2.0bb | 3.0bb |
|---|---|---|
| 25bb: jams | 23%: all pairs, A3s+, A9o+, KTs+, KQo, QTs+, suited connectors / one-gappers to 54s / 64s, 98o | 22%: similar, plus A2s and J7s |
| 25bb: folds | nothing | 37%: offsuit hands below ~J6o / T7o, weak suited hands (T3s–T2s, 9x–7x low) |
| 50bb: 3-bets | 19% to 6.5bb: all pairs, most suited aces, A7o+, KQo, KTs+, QTs+, T9s / 98s / 87s | 18% all-in: pairs, broadways, suited connectors; suited A5s–A2s now **call** |

* **The bigger the open, the more the BB's continuing range shifts toward hands that play well all-in.**
  Pairs and suited connectors keep jamming while the weakest offsuit hands fold.

Per-hand charts: `bb_range_{25,50,100}bb_{open}.png`, `hands_3bet_{2,3}.png`, `hands_fold_{2,3}.png`.
Data: `preflop_summary.csv`, `threebet_sizes.csv`, `notable_hands.csv`.

### Postflop: how the short-stacked BB plays a called open

Whole-hand models (experiment 16 sizes, 250 batches each) at 25 / 50bb for 2.0 / 2.25 / 3.0bb opens, using
the BB's stack-adjusted calling ranges. 100bb = experiment 16's model (its ranges are slightly different).

| | 25bb / 2.0 | 25bb / 2.25 | 25bb / 3.0 | 50bb / 2.0 | 50bb / 2.25 | 50bb / 3.0 | 100bb / 2.25 |
|---|---|---|---|---|---|---|---|
| stack-to-pot ratio after the call | 5.8 | 5.1 | **3.7** | 12.0 | 10.6 | 7.8 | 21.7 |
| BB net per called pot | −0.80 (−20% of pot) | −0.86 (−19%) | −0.98 (−16%) | −0.72 (−18%) | −0.77 (−17%) | −0.86 (−14%) | −0.85 (−19%) |
| BB donks the flop | 10% | 12% | **18%** | 9% | 11% | 12% | 9% |
| BB vs 1/3 c-bet: fold / call / raise | 50 / 18 / 32% | 50 / 18 / 32% | 47 / 20 / 33% | 48 / 19 / 33% | 48 / 23 / 30% | 44 / 27 / 30% | 46 / 19 / 35% |
| BB vs 3/4 c-bet: fold / call / raise | 60 / 24 / 16% | 60 / 23 / 17% | 64 / **11** / **25**% | 58 / 27 / 15% | 56 / 27 / 18% | 54 / 26 / 20% | 59 / 25 / 17% |
| BB leads the turn after calling the flop | 14% | 16% | **35%** | 12% | 13% | 14% | 10% |
| BB vs turn barrel: fold / call / raise | 48 / 31 / 21% | 47 / 28 / 25% | 48 / 23 / 30% | 48 / 37 / 15% | 47 / 36 / 17% | 49 / 34 / 18% | 50 / 38 / 12% |
| BB vs river bet after a barrel: fold / call / raise | 46 / 25 / 28% | 44 / 21 / 35% | 43 / 20 / 37% | 53 / 28 / 19% | 53 / 28 / 19% | 52 / 23 / 25% | 54 / 35 / 11% |
| all-in before the river | 2.5% | 3.0% | **8.4%** | 0.2% | 0.6% | 1.1% | ~0% |

![BB vs a 1/3 c-bet](output/bb_short_stack/post_vs_cbet.png)

* **Shorter stacks and bigger opens push the BB from calling to raising.** The fold rates barely move
  (~45–50% vs a small c-bet everywhere). The turn and river raise rates are where the difference shows:
  21–30% vs a turn barrel at 25bb vs 12% at 100bb, and 28–37% vs a river bet vs 11%. With a low
  stack-to-pot ratio, a raise is often all-in, so calling and raising later merge into "get it in now".
* **At 25bb vs a 3.0bb open (SPR 3.7) it's close to all-in-or-fold.** Vs a 3/4-pot c-bet the BB calls only
  11% and raises 25%. It donks the flop 18% and leads the turn 35% after calling, and 8% of hands are all-in
  before the river.
* **The BB loses less, as a share of the pot, the bigger the open** (−20% → −16% at 25bb, −18% → −14% at
  50bb). It defends a tighter range vs a big open, so the called pots are better for it. In bb, the bigger pot
  still costs more.

![BB check-raises by hand](output/bb_short_stack/post_hand_vs_third_raise.png)

**By hand, vs a 1/3-pot c-bet:**
* **top pair / overpair:** check-raises **86–93% at 25bb** vs 66–73% at 50bb. Short-stacked, raise and get it in.
* **two pair+:** raises 70–79% at 25bb (the rest slowplay by calling) vs 76–92% at 50bb.
* **middle / weak pairs:** raise less as the open grows. At 50bb it's 60% vs a 2.0 open and 28% vs a 3.0
  open, where calling becomes the main play (71%).
* **weak draws:** at 25bb vs a 3.0 open they raise 56% (vs 31–33% vs smaller opens), using fold equity
  while it still exists.
* **air:** folds 81–91% everywhere.

Data: `postflop_summary.csv`, `postflop_by_hand.csv`; also `post_hand_vs_third_fold.png` and
`post_hand_turn_vs_barrel_fold.png`.

**Caveats:**
* The preflop model has no BTN open-jam or limp, which matter at 20bb and below.
* Its 3-bet pots use the same realization formula as single-raised pots.
* The stack adjustment comes from three whole-hand models at a 2.25bb open and is extrapolated to 20bb.
* The whole-hand model's BB realization is lower than the 0.915 fit even at 100bb. The study uses the
  *relative* stack effect, not the absolute level.
* One seed per configuration.

## Experiment 20: BTN open-jams and limps for stacks under 20bb

`preflop/shortstack.py` (new solver) and `scripts/short_stack_jam_limp.py` (~1 min).

Experiments 1–19 followed the "never limp" rule, and the BTN could only fold or raise. At short stacks that
leaves a lot out, so the new preflop tree has more options:

```
BTN:  fold | limp | raise to s | all-in
  limp   -> BB: check | raise to r | all-in       (vs the raise, BTN: fold | call | all-in)
  raise  -> BB: fold | call | all-in             (experiment 19: all-in is the best 3-bet at <= 30bb)
  all-in -> BB: fold | call
```

* **Same assumptions as before.** Pots that see a flop use the realization formula with the BB's
  stack-adjusted r_oop of 1.0 (experiment 19); all-in pots use raw equity.
* **Sizes are chosen by each player.** For each stack the BB picks its best raise size vs a limp
  (2.5–5bb) for every open size, and the BTN picks its best open (2.0–3.0bb) given that.
* **Checks:**
  * With limps and open-jams turned off, the new solver reproduces `solver.py` exactly (unit test).
  * Every solution is within 0.04 milli-bb of an equilibrium.
  * With limps off, it matches the known heads-up push/fold numbers: at 10bb the BTN shoves ~58% and the BB
    calls ~37%.

### What the BTN should do, by stack

![BTN actions by stack](output/short_stack/btn_actions_by_stack.png)

| Stack | Fold | Limp | Raise | All-in | Best open | BB calls an all-in | BTN EV (mbb / hand) |
|---|---|---|---|---|---|---|---|
| 6bb | 32% | 0% | 0% | **68%** | – | 54% | +38 |
| 8bb | 30% | 15% | 0% | **55%** | – | 44% | +2 |
| 10bb | 26% | 30% | 0% | **44%** | – | 37% | −24 |
| 12bb | 23% | **44%** | 0% | 33% | – | 32% | −40 |
| 15bb | 20% | **51%** | 4% | 25% | 2.25 | 26% | −50 |
| 17bb | 19% | **52%** | 9% | 20% | 2.25 | 23% | −56 |
| 20bb | 16% | **57%** | 18% | 8% | 2.25 | 20% | −58 |
| 25bb | 16% | **55%** | 28% | 1% | 2.5 | 18% | −53 |

* **Below 10bb: mostly all-in or fold.** At 8bb the BTN shoves 55%: any ace, any suited king, most suited
  queens, broadways, pairs 22–99 (TT+ limp to trap) and suited connectors. It folds only trash (T2s, 9x–7x low suited and low offsuit).
* **10–20bb: limping takes over.** The BTN limps 30–57%. That includes **trapping** with premiums
  (AA, KK, QQ, JJ limp at 8–15bb), plus most medium offsuit hands that want to see a cheap flop.
  * **The all-in range is the hands that play badly after the flop but have good equity when called**
    (15bb): suited A7s–A2s, offsuit A9o–A2o and broadways, suited connectors 98s–53s, small pairs 22–44.
  * **Small raises appear from ~15bb** (4%) and grow to 28% at 25bb.
* **Open-jams fade out by 20–25bb** (8% at 20bb, 1% at 25bb).

### What each option is worth

![Option value](output/short_stack/option_value.png)

| Stack | Value of limping | Value of open-jamming | Value of both (vs raise-only) |
|---|---|---|---|
| 6bb | 0 | **+141** | +144 |
| 8bb | +6 | **+116** | +132 |
| 10bb | +16 | **+84** | +112 |
| 12bb | **+40** | +37 | +79 |
| 15bb | **+37** | +29 | +75 |
| 20bb | **+42** | +6 | +55 |
| 25bb | **+44** | 0 | +47 |

(milli-bb per hand the BTN loses if that option is removed)

* **Open-jamming is worth the most at 10bb and below** (+84 to +141 mbb per hand), and almost nothing by 25bb.
* **Limping is worth ~40 mbb per hand from 12bb up.** The "never limp" rule costs the BTN a lot at short
  stacks in this model.
* **Caveat on limps:** a limped pot uses the same realization formula as a raised pot, and the BB has r_oop = 1.0
  at these depths (it realizes equity as well as the BTN). Limped pots have not been checked with the
  whole-hand model yet.

### How the BB should respond

| Stack | vs a limp: check / raise / all-in | BB's raise size vs a limp | vs a raise: fold / call / all-in | Calls an all-in |
|---|---|---|---|---|
| 8bb | 50 / 18 / 33% | 3.5bb | (BTN never raises) | 44% |
| 10bb | 53 / 15 / 32% | 3.5bb | (BTN never raises) | 37% |
| 12bb | 57 / 14 / 30% | 3.5bb | (BTN never raises) | 32% |
| 15bb | 60 / 18 / 22% | 3.5bb | 18 / 55 / 28% | 26% |
| 20bb | 61 / 22 / 16% | 3bb | 17 / 60 / 23% | 20% |
| 25bb | 61 / 29 / 10% | 3bb | 27 / 53 / 20% | 18% |

![BB vs a limp, 12bb](output/short_stack/bb_vs_limp_12bb.png)

* **Vs a limp, the BB is polarized three ways:**
  * **raise small with premiums and strong kings/queens** (AA–99, AK–AJ suited, KQ, KJ), hoping to get
    called or jammed on
  * **jam with hands that want fold equity but play badly after the flop:** suited aces and kings, offsuit
    aces, suited connectors, small and medium pairs
  * **check the rest**, plus a few small-raise bluffs with junk (82o, 93o, 32s)
* **Vs the BTN's all-in, call tighter as stacks grow:** 44% of hands at 8bb, 37% at 10bb, 26% at 15bb,
  20% at 20bb.
  * At 10bb that's any ace, any suited king, K5o+, Q6s+, J8s+, T8s+, and pairs.
  * At 20bb it's A4s+, A7o+, K9s+, KJo+, Q9s+, J9s+, T8s+, and 33+.
* **Vs a raise at 15–25bb, the BB rarely folds** (17–27%), calls 53–60%, and 3-bets all-in 20–28%. The
  BTN's raising range is narrow, because most of its hands limp or jam.

Per-hand charts: `btn_{8,10,12,15,20}bb.png`, `bb_vs_jam_*.png`, `bb_vs_limp_*.png`, `bb_vs_raise_{15,20}bb.png`;
data in `summary.csv` and `by_hand.csv`.

**What this changes:**
* **Experiments 18–19 assumed the BTN never limps.** At 25bb this experiment has it limping 55% and
  raising only 28%, so the raise-only results at 20–25bb describe a BTN that is giving up ~45 mbb per hand.
* **The BTN's EV is negative from 10bb up in this model.** That's because the BB realizes as well as the
  BTN at these depths. The position edge is only in the whole-hand models, which haven't been run for limped
  pots.

## Experiment 21: The whole-hand model on limped pots (10–25bb)

`scripts/limped_pot_study.py` (`train` / `report`).

Experiment 20 valued limped pots with the preflop realization formula only. Here the whole-hand model
(experiment 16 sizes, 250 batches, 11–20 min per model) plays them out:

* **Limped pots (2bb pot) at 10 / 12 / 15 / 20 / 25bb.** Ranges are experiment 20's: the BTN's limps vs the
  BB's checks.
* **Raised pots at 15 / 20 / 25bb**, with the BTN's raises vs the BB's calls, so limped and raised pots
  are measured by the same model.

### Limped pots are worth about twice what the preflop formula said

![Limped pot value](output/limped_pots/limped_value.png)

| | 10bb | 12bb | 15bb | 20bb | 25bb |
|---|---|---|---|---|---|
| BTN net per limped pot, whole-hand model | +0.44 (22%) | +0.39 (19%) | +0.40 (20%) | +0.32 (16%) | +0.31 (16%) |
| same, preflop formula (experiment 20) | +0.22 (11%) | +0.19 (10%) | +0.17 (9%) | +0.14 (7%) | +0.12 (6%) |
| BTN net per raised pot, whole-hand / formula | – | – | 25% / 17% | 24% / 15% | 19% / 12% |

* **The formula undersells the BTN's position in every pot,** not just limped ones. Its BB realization of 1.0
  implies the BB realizes as well as the BTN; the whole-hand model implies **0.57–0.64 in limped pots** and
  **0.63–0.69 in raised pots**.
* **Limped pots favor the BTN slightly more than raised pots:** the limped / raised ratio of implied r_oop
  is 0.87, 1.02 and 0.92 at 15 / 20 / 25bb, pooled to **0.93**. Each pair of models has different ranges,
  so the per-stack ratios are noisy.

### Re-solving the preflop tree

Two calibrations of experiment 20's tree:
* **relative:** raised pots keep r_oop = 1.0 (experiment 19); limped pots get 1.0 × 0.93. This changes only
  how limped pots compare with raised ones.
* **absolute:** both pot types use the whole-hand model's implied r_oop. This puts the BTN's full
  postflop edge into the preflop model.

| Stack | Exp 20: fold / limp / raise / all-in | Relative | Absolute |
|---|---|---|---|
| 8bb | 30 / 15 / 0 / 55% | 28 / 18 / 0 / 54% | 24 / 31 / 0 / 45% |
| 10bb | 26 / 30 / 0 / 44% | 24 / 34 / 0 / 42% | 15 / 57 / 7 / 22% |
| 12bb | 23 / 44 / 0 / 33% | 22 / 46 / 0 / 32% | 10 / 61 / 21 / 7% |
| 15bb | 20 / 51 / 4 / 25% | 17 / **58** / 1 / 24% | 2 / **79** / 14 / 5% |
| 20bb | 16 / 57 / 18 / 8% | 15 / **62** / 16 / 7% | 2 / **70** / 28 / 1% |
| 25bb | 16 / 55 / 28 / 1% | 14 / **61** / 25 / 0% | 1 / **76** / 23 / 0% |

![Limping value by version](output/limped_pots/limp_value_versions.png)

* **The limping result holds up and gets stronger.** Limping is worth **+47–55 mbb per hand at 12–25bb**
  (relative), up from +37–44. With the BTN's full postflop edge (absolute) it's worth +47–77.
* **Relative:** the BTN limps a bit more (58–63% at 15–25bb), folds a bit less, and open-jams about as
  often.
* **Absolute:** with its full postflop edge, the **BTN plays 98–99% of hands at 15–25bb** (limping 70–82%),
  and its EV turns positive (+25 to +103 mbb per hand). That matches the usual heads-up picture, where the
  SB / BTN has the edge. Below 10bb all-ins still dominate. The BB stops making small raises vs a limp and
  instead **jams 23–58%** (more at shorter stacks) or checks.

Data: `preflop_resolved.csv`, `realization.csv`; the re-solved charts are in `resolve_relative/` and
`resolve_absolute/` (same files as `output/short_stack/`).

### How limped pots are played

| | 10bb | 12bb | 15bb | 20bb | 25bb |
|---|---|---|---|---|---|
| stack-to-pot ratio | 4.5 | 5.5 | 7.0 | 9.5 | 12.0 |
| BB leads the flop | 7% | 9% | 10% | 12% | 13% |
| BTN bets when checked to (1/3 / 3/4) | 70% (45 / 25) | 71% (47 / 23) | 69% (43 / 26) | 68% (40 / 28) | 67% (41 / 26) |
| BB vs a flop bet: fold / call / raise | 54 / 22 / 24% | 52 / 21 / 26% | 51 / 25 / 24% | 50 / 25 / 25% | 49 / 24 / 27% |
| BTN vs a lead: fold / call / raise | 26 / 34 / 41% | 28 / 33 / 39% | 28 / 34 / 39% | 30 / 32 / 38% | 29 / 36 / 34% |
| BTN bets the turn after a checked flop | 61% | 54% | 52% | 50% | 51% |
| checked down on every street | 4% | 5% | 5% | 5% | 6% |
| all-in before the river | 4.3% | 2.9% | 1.8% | 0.7% | 0.2% |

![BTN bets the flop, limped pots](output/limped_pots/limped_btn_bets_flop.png)

* **When the BB checks, the BTN bets about 70%, like a c-bet.** Nobody has shown strength preflop, so the
  BTN's position does the work.
  * **air bets 72–79%** (mostly small)
  * **top pairs bet 87–94%**
  * **middle / weak pairs check more as stacks get deeper** (65% → 43%)
  * **two pair+ slowplays at 10bb** (bets 60%) **but bets 90–95% at 15–25bb**
* **The BB defends by hand strength:**
  * **air folds 86–93%**
  * **weak draws fold 26–40%**, much less than in raised pots (48–64%), because the limped pot gives a
    better price against a small bet
  * **top pairs and better almost never fold**, and raise 47–86%: two pair+ raises more as stacks get deeper
* **The BB leads 7–13%,** mostly top pair (23–27%) and two pair+ (13–35%). The BTN raises those leads 34–41%.

Data: `postflop_summary.csv`, `postflop_by_hand.csv`; also `limped_bb_leads_flop.png` and
`limped_bb_folds_to_bet.png`.

**Caveats:**
* The whole-hand model's absolute realization has the same open question as experiment 16: its BB
  check-raises about 2× as often as TexasSolver's. The relative version is the safer read.
* One seed per model. The limped / raised ratio varies 0.87–1.02 across stacks.

## Next steps

1. **Fix the BB's check-raise level** (experiment 16): give the model a ~3x raise like the solver's and test
   whether the BTN's extra air c-bets make raising too profitable.
2. **Learn the opponent model from data:** estimate a real opponent's donk frequencies by hand from hand
   histories, instead of picking from three fixed profiles.
3. **Preflop bluffs from postflop play:** feed the whole-hand values back into the preflop solver.
4. **One consistent realization:** feed the whole-hand model's values for raised *and* limped pots back
   into experiments 18–20 at every stack (experiment 21's absolute version), once the check-raise level is fixed.
5. **Multiway** play.
