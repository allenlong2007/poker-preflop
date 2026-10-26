"""Experiment 49: stack-depth-consistent preflop policies.

The preflop strategy was solved at 100bb (the main tree) and at 60/40/25/15/10/6bb (`by_depth`). Instead of
snapping to the nearest solved depth, a stack between two solved depths gets a log-linear blend of the two trees'
per-hand action probabilities. The engine plays from this blend and the postflop range estimator (bot/potodds.py)
reads the opponent's preflop range from the same blend, so the bot's own play and its picture of the opponent's
range always assume the same depth: a 12bb opponent's open or shove range is a 12bb range, not a 100bb one.

Actions a tree doesn't have (the 40/60bb trees have no shoves, the 6/10bb trees no 3-bet) count as probability 0
there, so each blended row still sums to 1. A node one tree lacks is taken, for that tree, from the 100bb tree and
then from the deepest shorter tree that has it (as before).
"""
import math

import numpy as np

STACK = 100.0


def depths(pre):
    """{depth: nodes} for every solved tree, including the 100bb main tree."""
    out = {float(k): v for k, v in pre.get("by_depth", {}).items()}
    out[STACK] = pre["nodes"]
    return dict(sorted(out.items()))


def bracket(pre, stack, nearest=False):
    """(lo, hi, weight of hi) for a stack: the two solved depths around it, weighted in log space. nearest=True
    reproduces the pre-experiment-49 behaviour (the single nearest solved depth) for comparisons."""
    ds = list(depths(pre))
    s = max(float(stack), 1e-9)
    if nearest:
        d = min(ds, key=lambda d: abs(math.log(d / max(s, 1.0))))
        return d, d, 1.0
    if s <= ds[0]:
        return ds[0], ds[0], 1.0
    if s >= ds[-1]:
        return ds[-1], ds[-1], 1.0
    lo = max(d for d in ds if d <= s)
    hi = min(d for d in ds if d >= s)
    if lo == hi:
        return lo, hi, 1.0
    return lo, hi, (math.log(s) - math.log(lo)) / (math.log(hi) - math.log(lo))


class BlendedNodes:
    """Dict-like view of the preflop nodes at an exact stack depth (see the module docstring)."""

    def __init__(self, pre, stack, nearest=False):
        self.trees = depths(pre)
        self.lo, self.hi, self.w = bracket(pre, stack, nearest)
        self.stack = float(stack)
        self.borrowed, self.missing = [], []

    def _tree_node(self, depth, name):
        tree = self.trees[depth]
        if name in tree:
            return tree[name]
        for d in [STACK] + sorted((d for d in self.trees if d != STACK), reverse=True):
            if name in self.trees[d]:
                self.borrowed.append((depth, d, name))
                return self.trees[d][name]
        return None

    def blend(self, name, hi_override=None, lo_override=None):
        """Blended {action: per-hand probabilities}; *_override replaces one side's node (per-size 100bb spots)."""
        lo = lo_override if lo_override is not None else self._tree_node(self.lo, name)
        hi = hi_override if hi_override is not None else self._tree_node(self.hi, name)
        if lo is None and hi is None:
            self.missing.append(name)
            raise KeyError(name)
        if lo is None or self.w >= 1.0:
            return {a: np.asarray(v, float) for a, v in (hi or lo).items()}
        if hi is None or self.w <= 0.0:
            return {a: np.asarray(v, float) for a, v in lo.items()}
        n = len(next(iter(hi.values())))
        zero = np.zeros(n)
        return {a: (1 - self.w) * np.asarray(lo.get(a, zero), float) + self.w * np.asarray(hi.get(a, zero), float)
                for a in dict.fromkeys(list(lo) + list(hi))}

    def action(self, name, act):
        """Per-hand probability of one action at this depth. When neither side of the blend has the action (a shove
        at 40/60bb, where those trees never shove), use the nearest solved depth (log distance) that has it."""
        node = self.get(name) or {}
        if act in node and np.asarray(node[act]).max() > 0:
            return node[act]
        for d in sorted(self.trees, key=lambda d: abs(math.log(d / max(self.stack, 1e-9)))):
            pol = self.trees[d].get(name, {}).get(act)
            if pol is not None and np.asarray(pol, float).max() > 0:
                self.borrowed.append((self.stack, d, f"{name}:{act}"))
                return np.asarray(pol, float)
        return None

    def __getitem__(self, name):
        return self.blend(name)

    def get(self, name, default=None):
        try:
            return self.blend(name)
        except KeyError:
            return default

    def __contains__(self, name):
        return any(name in t for t in self.trees.values())

    @property
    def exact(self):
        """True when the stack is a solved depth (no blending)."""
        return self.lo == self.hi or self.w in (0.0, 1.0)

    @property
    def weight_100(self):
        """Weight of the 100bb tree in this blend (the only depth with per-size answers and adaptation)."""
        return self.w if self.hi == STACK else 0.0

    def describe(self):
        return {"depth_bb": self.hi if self.w >= 0.5 else self.lo, "depth_lo": self.lo, "depth_hi": self.hi,
                "weight_hi": round(self.w, 4), "stack_bb": round(self.stack, 3)}
