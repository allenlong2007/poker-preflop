"""Experiment 7: the Big Blind's strategy -- preflop and postflop -- vs different Button raise sizes.

For each BTN open size:
  1. Realization at THIS size: start from the pot-odds simulation fit, then loop
     solve preflop -> simulate postflop (pot-odds rules) -> refit alpha / r_oop,
     so a bigger pot (lower stack-to-pot ratio) gets its own realization.
  2. BB preflop: fold / call / 3-bet ranges with its best 3-bet size, compared to
     the minimum defense frequency (MDF) -- how often the BB must continue so a
     BTN raise with any two cards can't auto-profit.
  3. BB postflop, from the simulated hands: how often it leads, folds to a bet,
     gets its bluffs through, how hands end, and which calls make money.

Usage:  python scripts/bb_study.py      (~10 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, range_grid, small_multiples, stacked_bars
from preflop.postflop import (BET_CALLED, BET_FOLD, BLUFF_CALLED, BLUFF_FOLD, CHECK_CALL, CHECK_CHECK,
                              CHECK_FOLD, NOT_REACHED, simulate)
from preflop.sizing import best_threebet
from preflop.solver import solve
from simulate_realization import fit

OPENS = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
ROUNDS = 2
N_HANDS = 600_000
LEARN_HANDS = 200_000
GRID_OPENS = [2.25, 3.0]          # sizes that get 13x13 charts
MIN_SAMPLES = 1500

OUT = ROOT / "output" / "bb_study"
OUT.mkdir(parents=True, exist_ok=True)
START = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
STREETS = ["flop", "turn", "river"]


def solve_at(E, W, s, alpha, r_oop):
    """Solve at open s with the BB's best 3-bet size. Returns (result, 3-bet size)."""
    ts = np.arange(2.5, 6.001, 0.25) * s
    evs = [-solve(E, W, s, t, iters=2000, alpha=alpha, r_oop=r_oop).btn_ev() for t in ts]
    t, _, on_edge = best_threebet(ts, np.array(evs))
    if on_edge:
        print(f"  WARNING: best 3-bet vs {s}bb is on the edge of the size grid")
    return solve(E, W, s, t, iters=5000, alpha=alpha, r_oop=r_oop), t


def postflop_stats(sim, s):
    codes, share = sim["codes"], sim["share"]
    row = {}
    for k, st in enumerate(STREETS):
        c = codes[:, k]
        reached = c != NOT_REACHED
        faced = np.isin(c, [CHECK_CALL, CHECK_FOLD])
        led = np.isin(c, [BET_CALLED, BET_FOLD, BLUFF_CALLED, BLUFF_FOLD])
        row[f"{st}_reached_pct"] = reached.mean()
        row[f"{st}_bb_leads_pct"] = led[reached].mean()
        row[f"{st}_btn_bets_after_check_pct"] = faced[reached & ~led].mean()
        row[f"{st}_bb_folds_to_bet_pct"] = (c == CHECK_FOLD)[faced].mean() if faced.any() else np.nan
    bluffs = np.isin(codes, [BLUFF_CALLED, BLUFF_FOLD])
    row["bb_bluff_success_pct"] = (codes == BLUFF_FOLD).sum() / max(bluffs.sum(), 1)
    bb_folded = (codes == CHECK_FOLD).any(1)
    btn_folded = np.isin(codes, [BET_FOLD, BLUFF_FOLD]).any(1)
    row["ends_bb_folds_pct"] = bb_folded.mean()
    row["ends_btn_folds_pct"] = btn_folded.mean()
    row["ends_showdown_pct"] = 1 - bb_folded.mean() - btn_folded.mean()
    bb_net = (1 - share) * 2 * s - s                        # BB's net result from calling
    row["bb_postflop_net_bb"] = bb_net.mean()
    row["bb_call_vs_fold_bb"] = bb_net.mean() + 1           # vs folding preflop (losing 1bb)
    return row


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    combos = np.array([n_combos(h) for h in HANDS])
    pct = lambda p: float((p * combos).sum() / combos.sum())
    f = pd.read_csv(START, header=None, index_col=0)[1]
    alpha0, r0 = float(f["alpha"]), float(f["r_oop"])

    rows, per_hand = [], []
    for s in OPENS:
        alpha, r_oop = alpha0, r0
        for rnd in range(ROUNDS):
            res, t = solve_at(E, W, s, alpha, r_oop)
            sim = simulate(res.btn_open[:, 1], res.bb_vs_open[:, 1], n=N_HANDS, s=s, seed=rnd + 1,
                           learn_hands=LEARN_HANDS)
            eq = E[sim["btn_class"], sim["bb_class"]]
            alpha, r_oop, _, _ = fit(eq, sim["share"])
            print(f"open {s:.2f} round {rnd + 1}: fit alpha {alpha:.2f}, r_oop {r_oop:.3f}")
        res, t = solve_at(E, W, s, alpha, r_oop)
        sim = simulate(res.btn_open[:, 1], res.bb_vs_open[:, 1], n=N_HANDS, s=s, seed=99,
                       learn_hands=LEARN_HANDS)
        eq = E[sim["btn_class"], sim["bb_class"]]
        fold, call, three = res.bb_vs_open.T
        mdf = 1.5 / (s + 1)          # BTN risks s - 0.5 to win 1.5: BB must continue 1.5 / (s + 1)
        row = {"open_bb": s, "alpha": alpha, "r_oop": r_oop, "bb_3bet_to_bb": round(t, 2),
               "bb_3bet_x_open": round(t / s, 2), "btn_raise_pct": pct(res.btn_open[:, 1]),
               "bb_fold_pct": pct(fold), "bb_call_pct": pct(call), "bb_3bet_pct": pct(three),
               "bb_defend_pct": 1 - pct(fold), "mdf_pct": mdf, "btn_ev_bb": res.btn_ev(),
               "bb_realization": (1 - sim["share"]).mean() / (1 - eq).mean()}
        row.update(postflop_stats(sim, s))
        rows.append(row)

        # Per-hand BB numbers: preflop action + postflop result when it calls.
        j = sim["bb_class"]
        n = np.bincount(j, minlength=169)
        net = np.bincount(j, (1 - sim["share"]) * 2 * s - s, 169) / np.maximum(n, 1)
        real = np.bincount(j, 1 - sim["share"], 169) / np.maximum(np.bincount(j, 1 - eq, 169), 1e-9)
        net[n < MIN_SAMPLES] = np.nan
        real[n < MIN_SAMPLES] = np.nan
        for k, h in enumerate(HANDS):
            per_hand.append({"open_bb": s, "hand": h, "fold": fold[k], "call": call[k], "3bet": three[k],
                             "hands_simulated": int(n[k]), "call_net_bb": net[k],
                             "call_vs_fold_bb": net[k] + 1, "realization": real[k]})
        if s in GRID_OPENS:
            action_grid({"fold": fold, "call": call, "3-bet": three},
                        f"BB vs {s:g}bb open: 3-bet to {t:.1f}bb", OUT / f"bb_preflop_{s:g}.png",
                        subtitle=f"Fold {pct(fold):.0%}   Call {pct(call):.0%}   3-bet {pct(three):.0%}   "
                                 f"(must defend {mdf:.0%})")
            m = np.nanmax(np.abs(net + 1))
            range_grid(net + 1, f"BB calls vs {s:g}bb: profit vs folding (bb per hand, simulated)",
                       OUT / f"bb_call_profit_{s:g}.png", fmt="signed", vmin=-m, vmax=m,
                       note="Blue = calling beats folding, red = folding is better (given the postflop "
                            "rules). Blank = not a calling hand.")
        print(f"open {s:.2f}: BB fold {row['bb_fold_pct']:.0%} call {row['bb_call_pct']:.0%} "
              f"3bet {row['bb_3bet_pct']:.0%} to {t:.1f}  | flop fold-to-bet "
              f"{row['flop_bb_folds_to_bet_pct']:.0%}, realization {row['bb_realization']:.2f}")

    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_hand).round(4).to_csv(OUT / "bb_by_hand.csv", index=False)

    labels = [f"{s:g}bb" for s in OPENS]
    stacked_bars(labels, {"fold": df.bb_fold_pct, "call": df.bb_call_pct, "3-bet": df.bb_3bet_pct},
                 "BB preflop response by BTN open size", "Share of BB hands", OUT / "bb_preflop_by_open.png",
                 colors={"fold": "#e4e3df", "call": "#2a78d6", "3-bet": "#eb6834"},
                 markers=list(1 - df.mdf_pct), marker_label="fold limit (MDF)")
    small_multiples(
        {"BB folds when the BTN bets": {st: (OPENS, list(df[f"{st}_bb_folds_to_bet_pct"] * 100)) for st in STREETS},
         "BB leads (bets first)": {st: (OPENS, list(df[f"{st}_bb_leads_pct"] * 100)) for st in STREETS},
         "BTN bets when the BB checks": {st: (OPENS, list(df[f"{st}_btn_bets_after_check_pct"] * 100))
                                         for st in STREETS}},
        "BB postflop by BTN open size (% of times the street is reached)", "BTN open size (bb)",
        OUT / "bb_postflop_by_open.png", legend_title="Street")
    stacked_bars(labels, {"BB folds": df.ends_bb_folds_pct, "BTN folds": df.ends_btn_folds_pct,
                          "showdown": df.ends_showdown_pct},
                 "How BB-called pots end, by BTN open size", "Share of called pots",
                 OUT / "bb_pot_endings_by_open.png")
    show = ["open_bb", "alpha", "r_oop", "bb_fold_pct", "mdf_pct", "bb_call_pct", "bb_3bet_pct", "bb_3bet_to_bb",
            "bb_realization", "flop_bb_folds_to_bet_pct", "ends_showdown_pct", "bb_call_vs_fold_bb", "btn_ev_bb"]
    print(df[show].round(3).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
