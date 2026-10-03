"""Step 1: compute equity vs a random hand and the 169x169 equity matrix -> data/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflop.cards import HANDS, n_combos
from preflop.equity import build_matrix, vs_random

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

print("Equity vs a random hand...")
eq = [vs_random(h) for h in HANDS]
df = pd.DataFrame({"hand": HANDS, "combos": [n_combos(h) for h in HANDS], "equity_vs_random": eq})
df.sort_values("equity_vs_random", ascending=False).to_csv(DATA / "equity_vs_random.csv", index=False)

print("169x169 equity matrix (~2 min)...")
E, W = build_matrix()
np.savez_compressed(DATA / "equity_matrix.npz", E=E, W=W, hands=np.array(HANDS))
print("Saved to data/")
