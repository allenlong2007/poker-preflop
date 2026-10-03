"""Experiment 6a: calibrate the realization model against published solver frequencies.

Targets (BB vs a ~2.5bb open, heads-up 100bb), from the Preflop Wizard heads-up guide,
https://www.preflopwizard.app/blog/poker-heads-up-strategy -- an aggregate of
unnamed solver runs that include limping, so treat it as a rough target:

    BB folds 30-35%, 3-bets 15-20%, 3-bet size ~9-10bb

We grid-search the strength effect alpha and the BB's position penalty r_oop,
score each pair by how far the BB's equilibrium response is from the target
ranges, then check which BTN open size the best-fitting model prefers. The open
size is NOT a target, so it is an out-of-sample check.

Usage:  python scripts/calibrate.py      (~6 min on a multi-core machine)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import heatmap, line
from preflop.sizing import bb_ev_grid, best_threebet
from preflop.solver import solve

OPEN = 2.5
TARGETS = {  # name: (low, high)
    "bb_fold_pct": (0.30, 0.35),
    "bb_3bet_pct": (0.15, 0.20),
    "bb_3bet_to_bb": (9.0, 10.0),
}
ALPHAS = np.round(np.arange(1.0, 2.001, 0.1), 2)
R_OOPS = np.round(np.arange(0.70, 1.001, 0.05), 2)
THREEBETS = np.arange(6.0, 13.01, 0.25)
OPENS = np.round(np.arange(1.75, 3.501, 0.05), 2)

OUT = ROOT / "output" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)


def score(row):
    """Sum of squared misses, each measured in units of its target's half-width (0 = inside every range)."""
    total = 0.0
    for k, (lo, hi) in TARGETS.items():
        miss = max(lo - row[k], 0, row[k] - hi)
        total += (miss / ((hi - lo) / 2)) ** 2
    return total


def bb_response(params):
    """BB's best 3-bet size vs the open, and its fold / call / 3-bet frequencies there."""
    alpha, r_oop = params
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    combos = np.array([n_combos(h) for h in HANDS])
    pct = lambda p: float((p * combos).sum() / combos.sum())
    evs = [-solve(E, W, OPEN, t, iters=2000, alpha=alpha, r_oop=r_oop).btn_ev() for t in THREEBETS]
    t, _, on_edge = best_threebet(THREEBETS, np.array(evs))
    res = solve(E, W, OPEN, t, iters=4000, alpha=alpha, r_oop=r_oop)
    fold, call, three = res.bb_vs_open.T
    return {"alpha": alpha, "r_oop": r_oop, "bb_3bet_to_bb": t, "bb_fold_pct": pct(fold),
            "bb_call_pct": pct(call), "bb_3bet_pct": pct(three),
            "btn_raise_pct": pct(res.btn_open[:, 1]), "size_on_grid_edge": on_edge}


def main():
    params = [(a, r) for a in ALPHAS for r in R_OOPS]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = list(pool.map(bb_response, params))
    grid = pd.DataFrame(rows)
    grid["score"] = grid.apply(score, axis=1)
    grid = grid.sort_values("score")
    grid.round(4).to_csv(OUT / "calibration_grid.csv", index=False)

    heatmap(grid.pivot(index="r_oop", columns="alpha", values="score").sort_index(ascending=False),
            "Calibration error (0 = matches every solver target)", "alpha (strength effect)",
            "r_oop (BB position penalty)", OUT / "calibration_error.png")

    best = grid.iloc[0]
    print("Best fits:")
    print(grid.head(8).round(3).to_string(index=False))

    # Out-of-sample check: which open size does the calibrated model prefer?
    model = {"alpha": float(best.alpha), "r_oop": float(best.r_oop)}
    table = bb_ev_grid(OPENS, lambda s: np.arange(2.5, 6.001, 0.25) * s, {"fit": model})
    btn_ev = np.array([-best_threebet(*table[("fit", s)])[1] for s in OPENS])
    best_open = float(OPENS[btn_ev.argmax()])
    good = OPENS[btn_ev >= btn_ev.max() - 0.001]
    line(list(OPENS), list(btn_ev * 1000),
         f"BTN profit by open size, calibrated model (alpha {best.alpha:g}, r_oop {best.r_oop:g})",
         "BTN open size (bb)", "BTN EV (milli-bb per hand)", OUT / "calibrated_open_size.png",
         highlight=(best_open, float(btn_ev.max() * 1000)))
    pd.DataFrame({"open_bb": OPENS, "btn_ev_bb_per_hand": btn_ev}).to_csv(OUT / "calibrated_open_size.csv",
                                                                         index=False)
    pd.Series({"alpha": best.alpha, "r_oop": best.r_oop, "score": best.score, "best_open_bb": best_open,
               "good_open_range_bb": f"{good.min():g}-{good.max():g}",
               "cost_of_2.25_bb_per_100": (btn_ev.max() - btn_ev[np.isclose(OPENS, 2.25)][0]) * 100}
              ).to_csv(OUT / "calibrated_fit.csv", header=False)
    print(f"\nCalibrated alpha {best.alpha:g}, r_oop {best.r_oop:g} -> best BTN open {best_open:g}bb "
          f"(within 1 mbb: {good.min():g}-{good.max():g}bb)")


if __name__ == "__main__":
    main()
