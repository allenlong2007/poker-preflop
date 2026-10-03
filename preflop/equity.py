"""Preflop all-in equity with eval7 (ties count as half a win)."""
import itertools
import time

import eval7
import numpy as np

from .cards import HANDS, combos, overlaps

ALL_COMBOS = [tuple(eval7.Card(c) for c in pair)
              for pair in itertools.combinations([r + s for r in "AKQJT98765432" for s in "shdc"], 2)]


def _cards(combo):
    return [eval7.Card(c) for c in combo]


def class_vs_class(a, b, iters=200_000):
    """Equity of hand class `a` vs hand class `b`, plus the number of combo pairs.

    We only need ONE combo of `a`: every combo of `a` is a suit-relabelling of every
    other, and relabelling suits leaves class `b` unchanged, so they all have the
    same equity. eval7 drops villain combos that share a card with hero (card removal).
    """
    hero = combos(a)[0]
    villain = [(tuple(_cards(c)), 1.0) for c in combos(b) if not overlaps(hero, c)]
    eq = eval7.py_hand_vs_range_monte_carlo(_cards(hero), villain, [], iters)
    return eq, len(combos(a)) * len(villain)


def vs_random(name, iters=1_000_000):
    """Equity of a hand class against one uniformly random hand."""
    hero = combos(name)[0]
    villain = [(c, 1.0) for c in ALL_COMBOS]
    return eval7.py_hand_vs_range_monte_carlo(_cards(hero), villain, [], iters)


def build_matrix(iters=200_000, verbose=True):
    """169x169 equity matrix E and combo-count weight matrix W.

    E[i, j] = equity of HANDS[i] vs HANDS[j]; W[i, j] = number of non-overlapping
    combo pairs (so blockers / card removal are baked in). Monte Carlo standard
    error per cell is about 0.5 / sqrt(iters), i.e. ~0.1% at 200k.
    """
    n = len(HANDS)
    E = np.full((n, n), 0.5)
    W = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # A class vs itself is 50% by symmetry; only the weight is needed.
                _, W[i, i] = class_vs_class(HANDS[i], HANDS[i], iters=1)
                continue
            E[i, j], W[i, j] = class_vs_class(HANDS[i], HANDS[j], iters)
            E[j, i], W[j, i] = 1 - E[i, j], W[i, j]
        if verbose and i % 13 == 12:
            print(f"  row {i + 1}/{n}  ({time.time() - t0:.0f}s)")
    return E, W
