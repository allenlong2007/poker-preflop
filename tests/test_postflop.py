"""Checks for the rule-based postflop simulator."""
import eval7
import numpy as np

from preflop.postflop import MEDIUM, STRONG, WEAK, _worker, strength


def cards(s):
    return [eval7.Card(c) for c in s.split()]


def test_strength_rules():
    cases = [("As Kd", "Ah 7c 2d", False, STRONG),     # top pair, good kicker
             ("Qs Qd", "Jh 7c 2d", False, STRONG),     # overpair
             ("Kh 5s", "Kd 9c 4s", False, MEDIUM),     # top pair, weak kicker
             ("9s 8s", "Ks 7s 2d", False, MEDIUM),     # flush draw
             ("9h 8c", "Ts 7d 2c 3h Kd", True, WEAK),  # missed draw on the river
             ("Ah 3c", "Kd Kc 9s", False, WEAK)]       # board pair only
    for hole, board, river, want in cases:
        assert strength(tuple(cards(hole)), cards(board), river) == want


def test_no_betting_realizes_raw_equity():
    E = np.load("data/equity_matrix.npz")["E"]
    w = np.ones(169)
    i, j, share, _, _, _ = _worker((w, w, 20_000, 2.25, 100, 1, {"passive": True}))
    assert abs(share.mean() - E[i, j].mean()) < 0.01
