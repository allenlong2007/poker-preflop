"""The short-stack tree must match solver.py when limps and open-jams are switched off."""
from pathlib import Path

import numpy as np

from preflop.shortstack import solve_short
from preflop.solver import solve

_D = np.load(Path(__file__).resolve().parents[1] / "data" / "equity_matrix.npz")


def test_matches_solver_without_limps_or_jams():
    E, W = _D["E"], _D["W"]
    a = solve(E, W, 2.0, 20, stack=20, iters=1500, alpha=1.25, r_oop=1.0)    # 3-bet = all-in
    b = solve_short(E, W, 2.0, 3.0, 20, iters=1500, alpha=1.25, r_oop=1.0, limp=False, jam=False)
    assert abs(a.btn_ev() - b.btn_ev()) < 1e-4


def test_short_tree_converges():
    E, W = _D["E"], _D["W"]
    r = solve_short(E, W, 2.0, 3.5, 10, iters=1500, alpha=1.25, r_oop=1.0)
    assert r.exploitability() < 1e-3
