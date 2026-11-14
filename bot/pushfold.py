"""Heads-up push/fold Nash ranges for stacks below the smallest solved preflop tree (6bb).

The game: the button (small blind, 0.5bb posted) shoves all-in or folds; the big blind (1bb posted) calls or folds.
Payoffs use the 169-class all-in equity matrix in data/equity_matrix.npz, with combo weights for card removal.
Solved by fictitious play (average strategies converge to the equilibrium of this two-action game); cached per
stack rounded to 0.1bb.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _matrix():
    from bot.engine import HANDS, INDEX
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    idx = [INDEX[h] for h in d["hands"]]
    E, W = np.zeros((169, 169)), np.zeros((169, 169))
    E[np.ix_(idx, idx)], W[np.ix_(idx, idx)] = d["E"], d["W"]
    combos = np.array([6 if h[0] == h[1] else 4 if h.endswith("s") else 12 for h in HANDS], float)
    return E, W, combos


@lru_cache(maxsize=256)
def _solve(stack10, iters=400):
    """(push probability, call probability) per hand class at stack10 / 10 bb effective."""
    S = stack10 / 10.0
    E, W, combos = _matrix()
    push, call = np.full(169, 0.5), np.full(169, 0.5)
    avg_push, avg_call = np.zeros(169), np.zeros(169)
    for t in range(1, iters + 1):
        # Button: EV(push) - EV(fold), in bb from its starting stack (fold loses the 0.5 posted).
        wc = W * call[None, :]                                    # weight of BB hands that call
        n_all, n_call = W.sum(1), wc.sum(1)
        eq_called = (E * wc).sum(1) / np.maximum(n_call, 1e-12)
        p_call = n_call / n_all
        ev_push = (1 - p_call) * 1.0 + p_call * S * (2 * eq_called - 1)
        br_push = (ev_push > -0.5).astype(float)
        # Big blind: EV(call) - EV(fold) against the pushing range (fold loses the 1 posted).
        wp = W.T * push[None, :]                                  # rows: BB hand, cols: button hands that push
        eq_vs_push = ((1 - E.T) * wp).sum(1) / np.maximum(wp.sum(1), 1e-12)
        br_call = (S * (2 * eq_vs_push - 1) > -1.0).astype(float)
        push += (br_push - push) / (t + 1)
        call += (br_call - call) / (t + 1)
        avg_push += push
        avg_call += call
    return avg_push / iters, avg_call / iters


def ranges(stack):
    """Push and call probabilities (arrays over bot.engine.HANDS order) at this effective stack in bb."""
    return _solve(int(round(max(1.0, float(stack)) * 10)))


def push_share(stack):
    """Fraction of all combos the button shoves at equilibrium."""
    push, _ = ranges(stack)
    _, _, combos = _matrix()
    return float((push * combos).sum() / combos.sum())
