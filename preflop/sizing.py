"""Bet-size searches built on the solver, run in parallel across CPU cores."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .solver import solve

_DATA = Path(__file__).resolve().parents[1] / "data" / "equity_matrix.npz"
_E = _W = None


def _load():
    global _E, _W
    if _E is None:
        d = np.load(_DATA)
        _E, _W = d["E"], d["W"]
    return _E, _W


def _bb_ev(job):
    """BB's equilibrium EV (bb/hand) for one (open, 3-bet, r_oop) combination."""
    s, t, r_oop, iters = job
    E, W = _load()
    return -solve(E, W, s, t, iters=iters, r_oop=r_oop).btn_ev()


def bb_ev_table(opens, threebets_for, r_oops, iters=2000):
    """BB EV for every (r_oop, open, 3-bet) in the grid.

    threebets_for(s) gives the 3-bet sizes to try against an open of s.
    Returns {(r_oop, s): (threebet sizes array, BB EV array)}.
    """
    keys, jobs = [], []
    for r in r_oops:
        for s in opens:
            ts = np.asarray(threebets_for(s))
            keys.append((r, s, ts))
            jobs += [(s, t, r, iters) for t in ts]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        evs = list(pool.map(_bb_ev, jobs, chunksize=8))
    out, k = {}, 0
    for r, s, ts in keys:
        out[(r, s)] = (ts, np.array(evs[k:k + len(ts)]))
        k += len(ts)
    return out


def best_threebet(ts, evs):
    """3-bet size with the highest BB EV, and whether it sits on the edge of the grid."""
    i = int(np.argmax(evs))
    return float(ts[i]), float(evs[i]), i in (0, len(ts) - 1)
