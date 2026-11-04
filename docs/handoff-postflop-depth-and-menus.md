# Handoff: postflop models at real stack depths, and wider bet-size menus

Written 2026-11-04 by Claude for Codex. Owner: Allen. Live bot: HUBot v7 (tag `v7`, f07411a). Read this whole file
before changing anything.

## Goal

Two weaknesses in v7's postflop play, measured on Allen's 935 live PokerNow decisions:

1. **Stack depth.** Every postflop model was trained at 100bb and is reused at any effective stack. The live median
   effective stack was 79bb, and a quarter of decisions were under 57bb. Flop cells don't even separate depths
   (`line` is always 0 on the flop; turn/river only split SPR into >=4, 1.5-4, <1.5). The deep-stack substitution for
   untrained cells (exp 40) is a symptom of this: it fired in only 4 of 935 decisions.
   Also, heads-up only the effective stack (the smaller stack) matters, so "hero 150bb vs a 60bb player" is the
   60bb game. 172 of the 935 decisions were like that; the effective stack never went above 100bb.
2. **Off-menu bet sizes.** 154 of 935 decisions (16%) faced a size the models weren't trained on (faced sizes, as
   pot fractions: 5th-95th percentile 0.26-1.5). The engine maps them onto trained sizes (`size_weights`,
   pseudo-harmonic). At 100bb v7 only breaks even vs the solver-like held-out bot `geo` (geometric sizes):
   -6 ± 14 bb/100, and vs `trapper` -8 ± 14.

Do this as two pilots on single-raised pots (`srp`, the most common pot), A/B each against v7, and scale up only a
pilot that passes the gate below. Then report to Allen. **Allen decides adoption; do not bump the version, tag, push,
or touch the live server.**

## Hard rules (these come from Allen's standing instructions)

- **Never change `Engine` defaults or live serving.** `bot/pokernow/server.py` must keep
  `Engine(pot_odds=True, exact_required=False)`. A previous change (ffac880) made exact-only serving the default and
  the live bot would have withheld every decision. Everything new must be opt-in (constructor argument or attribute),
  default off, until Allen adopts it.
- **Never overwrite existing policy files.** `output/match/policy/` and `output/match/policy_fine/` are what v7
  plays. The A/B baseline reads them too (it symlinks `output/`). New models go only to new directories:
  `output/match/policy_depth/<depth>/<pot>.npz` and `output/match/policy_menu/<pot>.npz` (and
  `policy_menu_depth/<depth>/` if both are adopted).
- **One heavy training job at a time.** Launch long runs detached so closing a session doesn't kill them:
  `subprocess.Popen([... "caffeinate", "-is", ...], start_new_session=True)` with stdout to a log file. Don't use
  `&` or `nohup` from a session shell (a session fork killed a run on 2026-10-29). Tell Allen to keep the Mac on AC
  with the lid open; sleep stretches jobs (a 2h budget once took a day).
- **Local CPU only.** No cloud or paid compute.
- `.venv/bin/python` for everything. Full test suite must pass (`.venv/bin/python -m pytest -q`, 175 tests now)
  before every commit. Add tests for new code.
- Commit code, not model files: don't commit new `.npz` models until Allen adopts them.
- Don't stage `AGENTS.md`, `HANDOFF.md`, `output/match/*` scratch, `output/trap_study*`. Add dated bullets to
  `HANDOFF.md` (uncommitted) as you go.
- Never read or print `output/pokernow/.token`. Hand data under `output/pokernow/` is private; don't commit it.
- Public text (commit messages, docs): plain, no em-dashes.

## How the current postflop system works (read these first)

- `scripts/match_strategy.py`
  - `pot_ranges()`: entering ranges per pot type.
  - `train()`: trains one pot with `preflop.fullgame.learn(..., stack=STACK)`. `STACK = 100`; the `postflop-fine`
    command uses `OPTS_FINE`.
  - Saves `sigma_*`, `strat_*`, `reach_*` and `opts_json` into `<pot>.npz`.
- `scripts/open_size_study.py`: `OPTS_MATCH`, `OPTS_FINE` (the size menus).
  - v7's menus (`OPTS_FINE`):
    - Flop: bet 1/3, 2/3 or pot; lead 1/3.
    - Turn: bet 1/3, 1/2, 3/4, pot or 1.5x; lead 1/3, 1/2, 3/4 or pot.
    - River: bet 1/3 up to 2x pot; lead 1/3 up to 1.5x.
    - Raises: `raise_pot=(0.5,)`, one size.
- `preflop/fullgame.py`: the whole-hand CFR+ game.
  - Cells are (texture, line, hand bucket[, size index[, raise index]]).
  - `line_index()` sets the SPR bands.
  - `node_shape()` already supports several raise sizes (`btn_vs_xr` is shaped `(n_bet, n_raise)`).
- `bot/engine.py` `_postflop()` reads the tables, in this order:
  - `pot_type()` picks the pot.
  - `opts(pot)` reads the menu from the npz.
  - `size_weights()` maps an off-menu facing size.
  - The exp-40 deep substitution handles untrained cells.
  - `_short()` / `_low_spr` rules take over at <= 12bb behind or SPR <= 1.2 (keep them as they are).
- **Known engine bug if you add raise sizes:**
  - `raise_to` always uses `raises[0]`, so every "raise r" label gets the same amount.
  - Facing a raise reads cell `(..., k, 0)`, so the raise index is always 0.
  - Fix both (raise amount per size, and map a faced raise to its raise index) before any multi-raise model plays.
- Labels are formatted `f"{f:.2g}"`: 1.25 prints as "1.2". Make sure no two menu sizes collide after formatting.
- Preflop is already depth-aware:
  - `output/match/policy/preflop.json` has trees at 100bb and `by_depth` 60/40/25/15/10/6.
  - Node names are the same, with per-hand action probabilities.
  - `bot/depth.py` blends adjacent depths in log space (exp 49).
  - Use those depth trees for the entering ranges of each depth's postflop models.

## Pilot A: postflop models at 40bb (single-raised pot)

1. Add a stack parameter to the training path. For example, `train(pot, s, btn, bb, fine=True, stack=40,
   folder=POLICY_DEPTH/"40")` and a CLI such as `postflop-depth 40 srp`.
   - Entering ranges: the same construction as `pot_ranges()`, but from `preflop.json["by_depth"]["40"]` nodes.
   - The open size at 40bb is the same 2.25bb.
   - Check `match_strategy._solve` for how 3-bet and iso sizes shrink at short depths (`t3`, `r`), and use the same
     sizes the 40bb preflop tree was solved with.
   - Same menu as v7 (`OPTS_FINE`) so the only change is depth. Same batches as the fine 100bb srp model
     (`BATCHES * 1.6`, `HANDS_PER_BATCH`).
   - Expected time: roughly 2-3h. The 100bb fine srp took about 2.5h, and shallower trees should be faster.
2. Engine, opt-in only:
   - `Engine(..., postflop_depths=None)` by default. Given a list of depths with models, pick the model by effective
     stack.
   - Final version: blend the two bracketing depths' action probabilities per cell, log-linear in stack, exactly as
     `bot/depth.py` does preflop.
   - The pilot only has 40 and 100, so blend between them for stacks in [40, 100] and use 40 below 40.
   - Pots with no depth model fall back to the 100bb model as today.
   - Record in `d.diagnostics` which depth model(s) were used.
3. A/B (see protocol) at STACKS=40,60,30: v7 vs the candidate with `postflop_depths=[40, 100]`.

## Pilot B: wider bet-size menu at 100bb (single-raised pot)

1. New opts `OPTS_WIDE` in `scripts/open_size_study.py`, recorded in the npz via `opts_json` as today. Suggested:
   - Flop: bet 1/4, 1/3, 1/2, 2/3, 3/4, pot, 1.25x; lead 1/4, 1/3, 1/2.
   - Turn: bet 1/4, 1/3, 1/2, 3/4, pot, 1.5x; lead 1/4, 1/3, 1/2, 3/4, pot.
   - River: bet 1/4, 1/3, 1/2, 3/4, pot, 1.5x, 2x; lead 1/4, 1/3, 1/2, 3/4, pot, 1.5x.
   - Raises: `raise_pot=(0.5, 1.0)`.
   - Fix the engine raise bug above first, with tests.
2. Train `srp` at 100bb into `output/match/policy_menu/srp.npz`.
   - Scale batches by the growth in action count vs `OPTS_FINE` (more actions need more samples).
   - Check convergence: train to half the batches, snapshot, finish, and report how much average strategies moved.
     If they still move a lot, say so instead of hiding it.
   - Expected time: roughly 5-6h. Report the real time.
3. Engine, opt-in: `Engine(..., policy_overrides={"srp": path})`, or similar, loads that pot from the new file. The
   menu comes from the file's `opts_json`.
4. A/B at STACKS=100,60. The opponent that matters most is `geo`.

## A/B protocol (pre-registered: use exactly this, don't change it after seeing results)

Harness: `scripts/version_match.py` (committed). Baseline build = the `v7` tag. The candidate is the working tree
with the new option switched on. The harness calls `Engine(exact_required=False, seed=1, pot_odds=True)`, so add a
small env switch or a wrapper so the candidate run enables the pilot option. Keep the baseline exactly v7.

```bash
B=/tmp/v7base && mkdir -p $B && git archive v7 bot preflop | tar -x -C $B && ln -s "$PWD/output" $B/output
PYTHONHASHSEED=0 SEED=70303 STACKS=40,60,30 .venv/bin/python scripts/version_match.py play $B v7 6000 output/match/depth_ab
PYTHONHASHSEED=0 SEED=70303 STACKS=40,60,30 <enable pilot> .venv/bin/python scripts/version_match.py play . cand 6000 output/match/depth_ab
.venv/bin/python scripts/version_match.py compare output/match/depth_ab/v7_70303.npz output/match/depth_ab/cand_70303.npz
```

- **Always `PYTHONHASHSEED=0`.** With it, two runs of the same build match deal for deal. Run the A/A check once
  (v7 vs v7 must print "identical") before trusting a comparison.
- **Two seeds, 6000 deals each:** 70303 and 70404. Not 70101 or 70202: those were used for v7 itself.
- **Gate to scale a pilot up:**
  - Pooled B-A over both seeds is > 0 at every tested stack.
  - Combined z >= 2 at the pilot's main stack (40bb for A, 100bb for B).
  - No stack has combined z <= -2.
- **Report every cell**, including bad ones, with the exact commands. A pilot that fails the gate is a result; report
  it and stop that line.

## Scale-up (only for pilots that pass)

- Depths 25, 40 and 60 for all five pots (`srp`, `3bet`, `limp`, `iso`, `srp_alt`).
  - Skip a pot/depth whose flop SPR is <= 1.2: the short-stack rule already plays those. For example, a 3-bet pot at
    25bb.
- If B passed too, train the depth models with `OPTS_WIDE`, and the 100bb set too.
- Rough cost: 30-60 CPU hours, which is several overnight runs. Run them one at a time.
- Then the same A/B, with STACKS=100,60,40,25,20 on seeds 70505 and 70606.
- Engine stays opt-in. Write up the results for Allen. He decides whether it becomes v8.

## What to hand back to Allen

A short report in `HANDOFF.md` and in your final message, covering:

- What was trained: paths, menus, depths, batches and wall time.
- The A/B tables: per cell and pooled per stack, z, both seeds.
- Whether each gate passed.
- Convergence notes, and any engine bugs found or fixed.
- The commits. Don't push.
