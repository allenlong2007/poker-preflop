"""Per-hand playability: how much better or worse than average a hand plays after the flop.

These bonuses are hand-picked assumptions, not measured -- a starting point to
replace later with numbers fitted to solver output.
"""
import numpy as np

from .cards import HANDS, RANKS, n_combos

SUITED = 0.06                                   # flushes and backdoor draws
CONNECTED = {0: 0.04, 1: 0.02, 2: 0.0}          # by gap; 3+ gaps = GAPPED
GAPPED = -0.03
BROADWAY = 0.03                                 # both cards T or higher


def playability_score(hand):
    hi, lo = RANKS.index(hand[0]), RANKS.index(hand[1])
    if hi == lo:
        return 0.0
    score = SUITED if hand[2] == "s" else 0.0
    score += CONNECTED.get(lo - hi - 1, GAPPED)
    score += BROADWAY if lo <= RANKS.index("T") else 0.0
    return score


def playability(k=1.0):
    """169 multipliers centred on 1.0 (combo-weighted average = 1). k scales the effect."""
    p = np.array([playability_score(h) for h in HANDS])
    combos = np.array([n_combos(h) for h in HANDS])
    p -= (p * combos).sum() / combos.sum()
    return 1 + k * p
