"""Step 2: solve BTN vs BB at 100bb, sweep open sizes, save ranges + charts -> output/.

Usage:  python scripts/solve.py            (open 2.25bb, 3-bet to 4x)
        python scripts/solve.py --open 2.5 --threebet 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import line, range_grid
from preflop.solver import solve

ap = argparse.ArgumentParser()
ap.add_argument("--open", type=float, default=2.25)
ap.add_argument("--threebet", type=float, default=None, help="default: 4x the open")
ap.add_argument("--stack", type=float, default=100)
args = ap.parse_args()

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
d = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = d["E"], d["W"]
combos = np.array([n_combos(h) for h in HANDS])


def pct_of_hands(p):
    """Share of all 1,326 combos taking an action."""
    return float((p * combos).sum() / combos.sum())


# 1. Equity vs a random hand.
eq = pd.read_csv(ROOT / "data" / "equity_vs_random.csv").set_index("hand").loc[HANDS, "equity_vs_random"]
range_grid(eq.values, "Equity vs a random hand", OUT / "equity_vs_random.png",
           fmt="eq", vmin=0.3, vmax=0.86)

# 2. Raise-size sweep: which open size earns the BTN the most?
rows = []
for s in [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]:
    r = solve(E, W, s, threebet_size=None if args.threebet is None else args.threebet, stack=args.stack)
    rows.append({"open_bb": s, "btn_ev_bb_per_hand": r.btn_ev(),
                 "btn_raise_pct": pct_of_hands(r.btn_open[:, 1]),
                 "bb_fold_pct": pct_of_hands(r.bb_vs_open[:, 0]),
                 "bb_call_pct": pct_of_hands(r.bb_vs_open[:, 1]),
                 "bb_3bet_pct": pct_of_hands(r.bb_vs_open[:, 2]),
                 "exploitability_bb": r.exploitability()})
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "open_size_sweep.csv", index=False)
here = sweep.loc[(sweep.open_bb - args.open).abs().idxmin()]
line(sweep.open_bb, sweep.btn_ev_bb_per_hand, "BTN profit by open size (equity-only model)",
     "BTN open size (bb)", "BTN EV (bb per hand)", OUT / "open_size_sweep.png",
     highlight=(here.open_bb, here.btn_ev_bb_per_hand))

# 3. Full strategy at the chosen size.
r = solve(E, W, args.open, threebet_size=args.threebet, stack=args.stack, iters=5000)
tag = f"{args.open:g}bb"
t = r.P.t
strat = pd.DataFrame({
    "hand": HANDS, "combos": combos,
    "btn_raise": r.btn_open[:, 1],
    "bb_fold": r.bb_vs_open[:, 0], "bb_call": r.bb_vs_open[:, 1], "bb_3bet": r.bb_vs_open[:, 2],
    "btn_vs_3bet_fold": r.btn_vs_3bet[:, 0], "btn_vs_3bet_call": r.btn_vs_3bet[:, 1],
    "btn_vs_3bet_jam": r.btn_vs_3bet[:, 2],
    "bb_vs_jam_call": r.bb_vs_jam[:, 1],
}).round(3)
strat.to_csv(OUT / f"strategy_{tag}.csv", index=False)

range_grid(r.btn_open[:, 1], f"BTN raise range — open to {tag} ({pct_of_hands(r.btn_open[:, 1]):.0%} of hands)",
           OUT / f"btn_open_{tag}.png")
range_grid(r.bb_vs_open[:, 2], f"BB 3-bet range vs {tag} open — 3-bet to {t:g}bb "
           f"({pct_of_hands(r.bb_vs_open[:, 2]):.0%} of hands)", OUT / f"bb_3bet_{tag}.png")
range_grid(r.bb_vs_open[:, 0], f"BB fold range vs {tag} open ({pct_of_hands(r.bb_vs_open[:, 0]):.0%} of hands)",
           OUT / f"bb_fold_{tag}.png")
range_grid(r.btn_vs_3bet[:, 2] * r.btn_open[:, 1], f"BTN 4-bet all-in vs {t:g}bb 3-bet",
           OUT / f"btn_jam_vs_3bet_{tag}.png")

print(sweep.round(4).to_string(index=False))
print(f"\nAt {tag}: BTN EV {r.btn_ev():+.4f} bb/hand, exploitability {r.exploitability() * 1000:.3f} mbb/hand")
print(f"Charts and CSVs in {OUT}")
