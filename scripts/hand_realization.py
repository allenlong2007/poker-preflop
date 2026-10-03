"""Experiment 5: does hand-by-hand equity realization make a 2.25bb open the best size?

Two kinds of per-hand realization, both on top of the BB's position penalty
(r_oop = 0.85):

  playability (k)  -- suited / connected / broadway hands realize more
                      (preflop/realization.py; k scales the bonuses)
  strength (alpha) -- strong hands realize more than their equity, weak hands
                      less, because after the flop they win bigger pots / fold
                      more often. alpha = 1 means no effect.

For each model the Button picks its open (1.75-3.5bb, every 0.05bb) knowing the
BB answers with its best 3-bet size (2.5x-6x the open).

Usage:  python scripts/hand_realization.py      (~8 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, line, multi_line, range_grid
from preflop.realization import playability
from preflop.sizing import bb_ev_grid, best_threebet
from preflop.solver import solve

R_OOP = 0.85
K = 2.0          # playability strength used in the playability models
ALPHAS = [1.25, 1.4, 1.5, 1.6, 1.75, 2.0]
OPENS = np.round(np.arange(1.75, 3.501, 0.05), 2)
MULTIPLES = np.arange(2.5, 6.001, 0.25)
GOOD_MBB = 1.0   # opens within this many milli-bb of the best count as "about as good"

OUT = ROOT / "output" / "hand_realization"
OUT.mkdir(parents=True, exist_ok=True)


def model(alpha=1.0, k=0.0):
    p = playability(k)
    return {"r_ip": p, "r_oop": R_OOP * p, "alpha": alpha}


MODELS = {"position only": model(),
          f"playability k={K:g}": model(k=K)}
MODELS.update({f"strength a={a:g}": model(alpha=a) for a in ALPHAS})
MODELS[f"both a=1.5 k={K:g}"] = model(alpha=1.5, k=K)


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    combos = np.array([n_combos(h) for h in HANDS])
    pct = lambda p: float((p * combos).sum() / combos.sum())

    # Local cache this script writes itself (gitignored) -- never load a .pkl from elsewhere.
    cache = OUT / "raw_grid.pkl"
    if cache.exists():
        table = pd.read_pickle(cache)
    else:
        table = bb_ev_grid(OPENS, lambda s: MULTIPLES * s, MODELS)
        pd.to_pickle(table, cache)        # delete this file to recompute

    curve = []
    for (label, s), (ts, evs) in table.items():
        t, bb_ev, on_edge = best_threebet(ts, evs)
        curve.append({"model": label, "open_bb": s, "bb_best_3bet_to_bb": t,
                      "btn_ev_bb_per_hand": -bb_ev, "size_on_grid_edge": on_edge})
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / "btn_ev_by_open_size.csv", index=False)

    summary = []
    for label in MODELS:
        g = curve[curve.model == label].set_index("open_bb")
        best_s = float(g.btn_ev_bb_per_hand.idxmax())
        best_ev = g.btn_ev_bb_per_hand.max()
        good = g.index[g.btn_ev_bb_per_hand >= best_ev - GOOD_MBB / 1000]
        t = float(g.loc[best_s, "bb_best_3bet_to_bb"])
        res = solve(E, W, best_s, t, iters=5000, **MODELS[label])
        fold, call, three = res.bb_vs_open.T
        summary.append({
            "model": label, "btn_best_open_bb": best_s,
            "good_open_range_bb": f"{good.min():g}-{good.max():g}",
            "cost_of_2.25_bb_per_100": round((best_ev - g.loc[2.25, "btn_ev_bb_per_hand"]) * 100, 2),
            "bb_3bet_to_bb": round(t, 2), "bb_3bet_vs_2.25_bb": round(float(g.loc[2.25, "bb_best_3bet_to_bb"]), 2),
            "btn_ev_bb_per_hand": round(best_ev, 4), "btn_raise_pct": round(pct(res.btn_open[:, 1]), 3),
            "bb_fold_pct": round(pct(fold), 3), "bb_call_pct": round(pct(call), 3),
            "bb_3bet_pct": round(pct(three), 3),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    # Charts: EV curves by alpha, best open vs alpha, and the ranges at alpha = 1.5.
    strength = ["position only"] + [f"strength a={a:g}" for a in ALPHAS]
    multi_line({("a=1 (none)" if m == "position only" else m.split()[1]):
                (list(curve[curve.model == m].open_bb), list(curve[curve.model == m].btn_ev_bb_per_hand * 1000))
                for m in strength},
               "BTN profit by open size, with strength-based realization",
               "BTN open size (bb)", "BTN EV (milli-bb per hand)", OUT / "btn_ev_by_open_alpha.png",
               legend_title="Strength effect")
    best = summary.set_index("model").loc[strength, "btn_best_open_bb"]
    line([1.0] + ALPHAS, list(best), "Best BTN open size vs strength effect (alpha)",
         "alpha (1 = no strength effect)", "Best BTN open (bb)", OUT / "best_open_vs_alpha.png",
         highlight=(1.5, float(best.loc["strength a=1.5"])))

    m = "strength a=1.5"
    r = summary.set_index("model").loc[m]
    res = solve(E, W, 2.25, r["bb_3bet_vs_2.25_bb"], iters=5000, **MODELS[m])
    fold, call, three = res.bb_vs_open.T
    range_grid(res.btn_open[:, 1], f"BTN raise range — open to 2.25bb (alpha 1.5, "
               f"{pct(res.btn_open[:, 1]):.0%} of hands)", OUT / "btn_open_2.25_alpha1.5.png")
    action_grid({"fold": fold, "call": call, "3-bet": three},
                f"BB vs 2.25bb open: 3-bet to {r['bb_3bet_vs_2.25_bb']:.1f}bb (alpha 1.5)",
                OUT / "bb_response_2.25_alpha1.5.png",
                subtitle=f"Fold {pct(fold):.0%}   Call {pct(call):.0%}   3-bet {pct(three):.0%}")
    pd.DataFrame({"hand": HANDS, "btn_raise": res.btn_open[:, 1], "bb_fold": fold, "bb_call": call,
                  "bb_3bet": three}).round(3).to_csv(OUT / "strategy_2.25_alpha1.5.csv", index=False)

    print(summary.to_string(index=False))
    if curve.size_on_grid_edge.any():
        print("WARNING: some BB best sizes sit on the edge of the grid; widen MULTIPLES")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
