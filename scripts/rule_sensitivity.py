"""Experiment 6c: how much do the simulated results depend on the postflop rule choices?

Re-runs the simulation (ranges fixed at the pot-odds simulated fit) with different
bet sizes and bluffing / calling rules, refits alpha / r_oop, and finds the BTN's
best open size for each.

Usage:  python scripts/rule_sensitivity.py      (~8 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.postflop import simulate
from preflop.sizing import bb_ev_grid, best_threebet
from simulate_realization import OPEN, OPENS, fit, ranges

VARIANTS = {  # label: (bluffing, bet size as a fraction of the pot, calling rule)
    "baseline (balanced, 2/3 pot, pot odds)": ("balanced", 2 / 3, "pot_odds"),
    "bet 1/2 pot": ("balanced", 1 / 2, "pot_odds"),
    "bet full pot": ("balanced", 1.0, "pot_odds"),
    "fixed bluff 1/3 (pot odds)": (1 / 3, 2 / 3, "pot_odds"),
    "old rules: bluff 1/3, call any pair": (1 / 3, 2 / 3, "fixed"),
}
OUT = ROOT / "output" / "simulated_realization_pot_odds"


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    f = pd.read_csv(OUT / "simulated_fit.csv", header=None, index_col=0)[1]
    btn_r, bb_c, _ = ranges(E, W, float(f["alpha"]), float(f["r_oop"]))
    rows, models = [], {}
    for label, (bluff, bet, calling) in VARIANTS.items():
        sim = simulate(btn_r, bb_c, n=800_000, s=OPEN, seed=7, bluff=bluff, bet_fraction=bet, calling=calling)
        bi, bj, share = sim["btn_class"], sim["bb_class"], sim["share"]
        eq = E[bi, bj]
        a, r, _, _ = fit(eq, share)
        models[label] = {"alpha": a, "r_oop": r}
        rows.append({"rules": label, "btn_equity": eq.mean(), "btn_realized": share.mean(),
                     "alpha_fit": a, "r_oop_fit": r})
        print(f"{label:36s} a={a:.2f} r_oop={r:.3f} realized {share.mean():.3f} vs equity {eq.mean():.3f}")
    table = bb_ev_grid(OPENS, lambda s: np.arange(2.5, 6.001, 0.25) * s, models)
    for row in rows:
        ev = np.array([-best_threebet(*table[(row["rules"], s)])[1] for s in OPENS])
        good = OPENS[ev >= ev.max() - 0.001]
        row["best_open_bb"] = float(OPENS[ev.argmax()])
        row["good_open_range_bb"] = f"{good.min():g}-{good.max():g}"
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "rule_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
