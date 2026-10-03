"""Experiment 3: how much does the Big Blind's out-of-position penalty change its defense?

Sweeps r_oop (the share of its equity the BB realizes in pots that see a flop)
from 0.75 to 1.0, with the Button fixed at r_ip = 1.0. Only the ratio r_ip / r_oop
matters, so the 1.10 / 0.85 "position model" from experiment 2 equals r_oop ~0.77 here.

For each r_oop and each open size (2.25-2.50bb) we find the BB's best 3-bet size,
then record its fold / call / 3-bet frequencies.

Usage:  python scripts/realization_sweep.py      (~2 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import small_multiples
from preflop.sizing import bb_ev_table, best_threebet
from preflop.solver import solve

R_OOPS = np.round(np.arange(0.75, 1.0001, 0.025), 3)
OPENS = [2.25, 2.30, 2.35, 2.40, 2.45, 2.50]
THREEBETS = np.arange(6.0, 16.01, 0.25)

OUT = ROOT / "output" / "realization_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    combos = np.array([n_combos(h) for h in HANDS])
    pct = lambda p: float((p * combos).sum() / combos.sum())

    table = bb_ev_table(OPENS, lambda s: THREEBETS, R_OOPS)
    rows = []
    for (r, s), (ts, evs) in table.items():
        t, ev, on_edge = best_threebet(ts, evs)
        res = solve(E, W, s, t, iters=4000, r_oop=r)
        fold, call, three = res.bb_vs_open.T
        rows.append({"r_oop": r, "open_bb": s, "best_3bet_to_bb": t,
                     "best_3bet_x_open": round(t / s, 2), "bb_ev_bb_per_hand": round(ev, 4),
                     "bb_fold_pct": round(pct(fold), 4), "bb_call_pct": round(pct(call), 4),
                     "bb_3bet_pct": round(pct(three), 4), "btn_raise_pct": round(pct(res.btn_open[:, 1]), 4),
                     "size_on_grid_edge": on_edge})
    df = pd.DataFrame(rows).sort_values(["open_bb", "r_oop"])
    df.to_csv(OUT / "summary.csv", index=False)

    panels = {}
    for col, label in [("best_3bet_to_bb", "Best 3-bet size (bb)"),
                       ("bb_fold_pct", "BB fold % vs the open"),
                       ("bb_3bet_pct", "BB 3-bet % vs the open")]:
        scale = 100 if col.endswith("pct") else 1
        panels[label] = {f"{s:.2f}bb": (list(g.r_oop), list(g[col] * scale))
                         for s, g in df.groupby("open_bb")}
    small_multiples(panels, "How the BB's out-of-position penalty changes its defense",
                    "r_oop (share of equity the BB realizes; 1.0 = no penalty)",
                    OUT / "realization_sweep.png", legend_title="BTN open")

    show = df[df.open_bb.isin([2.25, 2.50])][["open_bb", "r_oop", "best_3bet_to_bb", "best_3bet_x_open",
                                              "bb_fold_pct", "bb_call_pct", "bb_3bet_pct"]]
    print(show.to_string(index=False))
    if df.size_on_grid_edge.any():
        print("WARNING: some best sizes sit on the edge of the 3-bet grid; widen THREEBETS")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
