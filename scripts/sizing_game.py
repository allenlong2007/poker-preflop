"""Experiment 4: the Button picks its open size, knowing the BB will pick its best 3-bet size.

For each open size s (2.0-4.0bb) the BB tries 3-bets from 2.5x to 6x the open and
keeps the one that earns it the most. The Button's value for s is its EV against
that best 3-bet size. The Button then picks the s with the highest EV.

Each player uses ONE size for all hands (no size mixing by hand) -- a simplification.
Run for several BB out-of-position penalties r_oop (1.0 = no penalty).

Usage:  python scripts/sizing_game.py      (~6 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, multi_line, range_grid
from preflop.sizing import bb_ev_table, best_threebet
from preflop.solver import solve

R_OOPS = [1.0, 0.9, 0.8, 0.75]
OPENS = np.round(np.arange(2.0, 4.001, 0.1), 2)
MULTIPLES = np.arange(2.5, 6.001, 0.125)      # BB 3-bet = multiple x open

OUT = ROOT / "output" / "sizing_game"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    combos = np.array([n_combos(h) for h in HANDS])
    pct = lambda p: float((p * combos).sum() / combos.sum())

    cache = OUT / "raw_grid.pkl"
    if cache.exists():
        table = pd.read_pickle(cache)
    else:
        table = bb_ev_table(OPENS, lambda s: MULTIPLES * s, R_OOPS)
        pd.to_pickle(table, cache)        # delete this file to recompute
    curve = []
    for (r, s), (ts, evs) in table.items():
        t, bb_ev, on_edge = best_threebet(ts, evs)
        curve.append({"r_oop": r, "open_bb": s, "bb_best_3bet_to_bb": t,
                      "bb_best_3bet_x_open": round(t / s, 3), "btn_ev_bb_per_hand": -bb_ev,
                      "size_on_grid_edge": on_edge})
    curve = pd.DataFrame(curve).sort_values(["r_oop", "open_bb"], ascending=[False, True])
    curve.to_csv(OUT / "btn_ev_by_open_size.csv", index=False)

    multi_line({f"r_oop {r:g}": (list(g.open_bb), list(g.btn_ev_bb_per_hand * 1000))
                for r, g in curve.groupby("r_oop", sort=False)},
               "BTN profit by open size, when the BB picks its best 3-bet size",
               "BTN open size (bb)", "BTN EV (milli-bb per hand)", OUT / "btn_ev_by_open_size.png",
               legend_title="BB realizes")

    summary = []
    for r, g in curve.groupby("r_oop", sort=False):
        best = g.loc[g.btn_ev_bb_per_hand.idxmax()]
        s, t = float(best.open_bb), float(best.bb_best_3bet_to_bb)
        res = solve(E, W, s, t, iters=5000, r_oop=r)
        fold, call, three = res.bb_vs_open.T
        summary.append({
            "r_oop": r, "btn_best_open_bb": s, "bb_3bet_to_bb": t, "bb_3bet_x_open": round(t / s, 2),
            "btn_ev_bb_per_hand": round(best.btn_ev_bb_per_hand, 4),
            "good_open_range_bb": "{}-{}".format(
                *g[g.btn_ev_bb_per_hand >= best.btn_ev_bb_per_hand - 0.002].open_bb.agg(["min", "max"])),
            "btn_raise_pct": round(pct(res.btn_open[:, 1]), 3),
            "bb_fold_pct": round(pct(fold), 3), "bb_call_pct": round(pct(call), 3),
            "bb_3bet_pct": round(pct(three), 3),
            "exploitability_mbb": round(res.exploitability() * 1000, 3),
        })
        tag = f"r{r:g}"
        range_grid(res.btn_open[:, 1], f"BTN raise range — open to {s:g}bb (r_oop {r:g}, "
                   f"{pct(res.btn_open[:, 1]):.0%} of hands)", OUT / f"btn_open_{tag}.png")
        action_grid({"fold": fold, "call": call, "3-bet": three},
                    f"BB vs {s:g}bb open: 3-bet to {t:g}bb (r_oop {r:g})", OUT / f"bb_response_{tag}.png",
                    subtitle=f"Fold {pct(fold):.0%}   Call {pct(call):.0%}   3-bet {pct(three):.0%}")
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.drop(columns="exploitability_mbb").to_string(index=False))
    if curve.size_on_grid_edge.any():
        print("WARNING: some BB best sizes sit on the edge of the grid; widen MULTIPLES")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
