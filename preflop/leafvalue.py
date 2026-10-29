"""Fast flop-leaf values for depth-limited preflop training: a fitted model of the legacy engine's postflop play.

Why (2026-10-29): one legacy rollout per flop leaf costs 25-40ms, so the full preflop game (1.4M infosets at 20bb,
79M at 100bb) would take months. Here the leaf value is predicted from the two hands' classes and the leaf's public
state, by a least-squares model fitted to recorded legacy rollouts (scripts/build_leaf_model.py).

    g = BTN's final result / pot at the leaf        (so the BTN's leaf value is g * pot, in bb)

Features: preflop all-in equity of the BTN class vs the BB class (class_equity table), hand-shape flags for both
seats, crossed with log stack-to-pot ratio; one model per preflop aggressor (none / btn / bb). The model is an
approximation of "both players continue with the legacy postflop policy": it is fitted to, and validated against,
held-out rollouts of the same engine, and recorded (with the continuation fingerprint it was fitted under) in the
candidate that uses it.
"""
import hashlib
import json
from itertools import combinations

import numpy as np

RANKS = "23456789TJQKA"
AGGRESSORS = ("none", "btn", "bb")


def hand_class(hole):
    """0..168: pairs, suited and offsuit classes by rank (suits only matter as suited / not)."""
    a, b = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    hi, lo = RANKS.index(a[0]), RANKS.index(b[0])
    if hi == lo:
        return hi
    index = 13 + (hi * (hi - 1)) // 2 + lo                      # 78 rank pairs with hi > lo
    return index if a[1] == b[1] else index + 78


def class_name(index):
    if index < 13:
        return RANKS[index] * 2
    suited, index = index < 91, (index - 13) % 78
    hi = next(h for h in range(1, 13) if (h * (h + 1)) // 2 > index)
    return RANKS[hi] + RANKS[index - (hi * (hi - 1)) // 2] + ("s" if suited else "o")


def class_combos(index):
    name = class_name(index)
    if len(name) == 2:
        return [(name[0] + s, name[1] + t) for s, t in combinations("cdhs", 2)]
    if name[2] == "s":
        return [(name[0] + s, name[1] + s) for s in "cdhs"]
    return [(name[0] + s, name[1] + t) for s in "cdhs" for t in "cdhs" if s != t]


def shape(index):
    """(pair, suited, connectedness 0..1) for a class."""
    name = class_name(index)
    if len(name) == 2:
        return 1.0, 0.0, 1.0
    gap = RANKS.index(name[0]) - RANKS.index(name[1])
    return 0.0, float(name[2] == "s"), max(0.0, 1 - (gap - 1) / 4)


SHAPES = np.array([shape(i) for i in range(169)])


def features(btn_class, bb_class, equity, spr):
    """Design rows: equity polynomial and hand shapes, each crossed with (1, log spr, log spr^2)."""
    btn_class, bb_class = np.atleast_1d(btn_class), np.atleast_1d(bb_class)
    e = np.atleast_1d(equity) - 0.5
    base = np.column_stack([np.ones_like(e), e, e ** 2, e ** 3, SHAPES[btn_class], SHAPES[bb_class]])
    s = np.log(np.atleast_1d(spr))
    return np.hstack([base, base * s[:, None], base * (s ** 2)[:, None]])


class LeafModel:
    def __init__(self, path):
        record = json.loads(open(path).read())
        self.record = record
        self.equity = np.asarray(record["class_equity"])
        self.coef = {k: np.asarray(v) for k, v in record["coefficients"].items()}
        self.spr_range = tuple(record["spr_range"])
        self.sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()

    def g(self, btn_class, bb_class, spr, aggressor):
        spr = min(max(spr, self.spr_range[0]), self.spr_range[1])           # no extrapolation past the data
        x = features(btn_class, bb_class, self.equity[btn_class, bb_class], spr)
        return float(x[0] @ self.coef[aggressor])

    def value(self, state, updater):
        """Updater's leaf value in bb at a flop leaf (no postflop action yet)."""
        pot = state.pot * state.config.chip_bb
        spr = (state.stack - state.spent[0]) / state.pot
        btn = self.g(hand_class(state.holes[0]), hand_class(state.holes[1]), spr, aggressor(state)) * pot
        return btn if updater == 0 else -btn


def aggressor(state):
    """Who made the last preflop raise ("none" for a limped pot)."""
    raises = [p for p, kind, _ in state.history[0] if kind == "raise"]
    return AGGRESSORS[1 + raises[-1]] if raises else "none"


def fit(rows):
    """rows: dicts with btn, bb, equity, spr, aggressor, g. Least squares per aggressor."""
    coefficients = {}
    for name in AGGRESSORS:
        part = [r for r in rows if r["aggressor"] == name]
        x = features([r["btn"] for r in part], [r["bb"] for r in part], [r["equity"] for r in part],
                     [r["spr"] for r in part])
        y = np.array([r["g"] for r in part])
        coefficients[name] = np.linalg.lstsq(x, y, rcond=None)[0].tolist()
    return coefficients
