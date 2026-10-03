"""Experiment 2: how should the Big Blind defend vs a 2.25bb-2.50bb open?

For each BTN open size (every 0.05bb) we try every BB 3-bet size, solve the
game, and keep the 3-bet size that earns the BB the most. Then we save the BB's
fold / call / 3-bet range at that size.

Two models are run:
  equity    -- raw all-in equity (position is worth nothing)
  position  -- BTN realizes 110% of its equity, BB 85%, in pots that see a flop

Usage:  python scripts/bb_defense.py          (~4 min)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, multi_line
from preflop.solver import solve

MODELS = {"equity": (1.0, 1.0), "position": (1.10, 0.85)}
OPENS = [2.25, 2.30, 2.35, 2.40, 2.45, 2.50]
THREEBETS = np.arange(6.0, 14.01, 0.25)
FLAT_MBB = 1.0   # 3-bet sizes within this many milli-bb of the best count as "about as good"

OUT = ROOT / "output" / "bb_defense"
OUT.mkdir(parents=True, exist_ok=True)
d = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = d["E"], d["W"]
combos = np.array([n_combos(h) for h in HANDS])


def pct(p):
    return float((p * combos).sum() / combos.sum())


def hand_list(p, threshold=0.5):
    """Hands that take an action more often than `threshold`, strongest-looking first."""
    return " ".join(h for h, v in zip(HANDS, p) if v > threshold)


grid_rows, summary, ranges = [], [], []
for model, (r_ip, r_oop) in MODELS.items():
    for s in OPENS:
        evs = []
        for t in THREEBETS:
            r = solve(E, W, s, t, iters=3000, r_ip=r_ip, r_oop=r_oop)
            evs.append(-r.btn_ev())                    # BB's EV = minus BTN's
            grid_rows.append({"model": model, "open_bb": s, "threebet_bb": t, "bb_ev_bb": evs[-1]})
        evs = np.array(evs)
        best_t = float(THREEBETS[evs.argmax()])
        flat = THREEBETS[evs >= evs.max() - FLAT_MBB / 1000]

        r = solve(E, W, s, best_t, iters=6000, r_ip=r_ip, r_oop=r_oop)
        fold, call, three = r.bb_vs_open.T
        summary.append({
            "model": model, "open_bb": s, "best_3bet_to_bb": best_t,
            "best_3bet_x_open": round(best_t / s, 2),
            "good_3bet_range_bb": f"{flat.min():g}-{flat.max():g}",
            "bb_ev_bb_per_hand": round(evs.max(), 4),
            "bb_fold_pct": round(pct(fold), 3), "bb_call_pct": round(pct(call), 3),
            "bb_3bet_pct": round(pct(three), 3),
            "btn_fold_vs_3bet_pct": round(pct(r.btn_vs_3bet[:, 0] * r.btn_open[:, 1])
                                          / pct(r.btn_open[:, 1]), 3),
            "exploitability_mbb": round(r.exploitability() * 1000, 3),
            "bb_3bet_hands": hand_list(three), "bb_fold_hands": hand_list(fold),
        })
        for h, f, c, b in zip(HANDS, fold, call, three):
            ranges.append({"model": model, "open_bb": s, "hand": h,
                           "fold": round(f, 3), "call": round(c, 3), "3bet": round(b, 3)})
        action_grid({"fold": fold, "call": call, "3-bet": three},
                    f"BB vs {s:.2f}bb open: 3-bet to {best_t:g}bb ({model} model)",
                    OUT / f"bb_range_{model}_{s:.2f}.png",
                    subtitle=f"Fold {pct(fold):.0%}   Call {pct(call):.0%}   3-bet {pct(three):.0%}")
        print(f"{model:8s} open {s:.2f}: 3-bet to {best_t:5.2f} ({best_t / s:.2f}x)  "
              f"fold {pct(fold):5.1%} call {pct(call):5.1%} 3bet {pct(three):5.1%}  "
              f"BB EV {evs.max():+.4f}")

    sub = pd.DataFrame(grid_rows).query("model == @model")
    multi_line({f"{s:.2f}bb": (list(g.threebet_bb), list(g.bb_ev_bb * 1000))
                for s, g in sub.groupby("open_bb")},
               f"BB profit by 3-bet size ({model} model)", "BB 3-bet to (bb)",
               "BB EV (milli-bb per hand)", OUT / f"bb_ev_by_3bet_size_{model}.png")

pd.DataFrame(grid_rows).to_csv(OUT / "ev_by_3bet_size.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(ranges).to_csv(OUT / "bb_ranges.csv", index=False)
print(f"\nSaved to {OUT}")
