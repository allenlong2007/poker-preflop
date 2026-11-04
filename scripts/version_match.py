"""Version A/B on identical duplicate deals against the frozen held-out bots (bot/heldout.py).

The harness behind the v6 -> v7 check (2026-11-02). It plays one bot build (any checkout or `git archive` of
bot/ + preflop/, with output/ reachable from it) against geo, trapper, limper, threebet and adaptive, both seats per
deal, and saves bb/100 per deal so two builds can be compared deal by deal.

    PYTHONHASHSEED=0 SEED=70202 STACKS=100,40 python scripts/version_match.py play <build root> <label> <deals> <out dir>
    python scripts/version_match.py compare <out dir>/<label A>_<seed>.npz <out dir>/<label B>_<seed>.npz

  play     -> <out dir>/<label>_<seed>.npz, one array per "stack|opponent" (bb/100 per deal, mean of both seats)
  compare  -> B minus A per cell, pooled per stack, with standard errors and z (paired by deal)

A baseline build of a tag:  git archive v7 bot preflop | tar -x -C DIR && ln -s <repo>/output DIR/output
It then reads the repo's output/ (policies), so new policy files must go to NEW directories, never over existing ones.
Always set PYTHONHASHSEED=0: then two runs of the same build match deal for deal (A/A diff exactly 0, checked
2026-11-04); without it set iteration order varies and an A/A run differs by about 1.5 bb/100 at 20bb x 6000 deals.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OPPS = tuple(os.environ.get("OPPS", "geo,trapper,limper,threebet,adaptive").split(","))
STACKS = tuple(float(x) for x in os.environ.get("STACKS", "100,20").split(","))
SEED = int(os.environ.get("SEED", 70_101))
ROOT = None


def engine():
    from bot.engine import Engine
    if os.environ.get("POSTFLOP_DEPTHS"):
        depths = [float(s) for s in os.environ["POSTFLOP_DEPTHS"].split(",")]
        return Engine(exact_required=False, seed=1, pot_odds=True, postflop_depths=depths)
    try:
        return Engine(exact_required=False, seed=1, pot_odds=True)      # v7+: legacy serving, as the live server
    except TypeError:                                                   # v6 and older: no exact mode
        return Engine(seed=1, pot_odds=True)


def run(job):
    root, stack, opp, deals = job
    sys.path.insert(0, str(root))
    from bot.heldout import make
    from bot.selfplay import DECK, play_hand
    e = engine()

    class Hero:
        def act(self, seat, hole, board, history, facing):
            d = e.decide(seat, hole, board, history, stack=stack)
            return d.action, d.to

    hero, villain = Hero(), make(opp)
    villain.stack = stack
    rng = np.random.default_rng(SEED + int(stack))
    out = np.zeros(deals)
    for k in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        holes, board = {"btn": c[:2], "bb": c[2:4]}, c[4:]
        for seat, sign in (("btn", 1), ("bb", -1)):
            agents = {"btn": hero, "bb": villain} if seat == "btn" else {"btn": villain, "bb": hero}
            log = []
            out[k] += sign * play_hand(agents, holes, board, stack=stack, log=log)
            if hasattr(villain, "observe"):
                villain.observe("bb" if seat == "btn" else "btn", log, holes, board)
    return f"{stack:g}|{opp}", out / 2 * 100


def play(root, label, deals, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("WORKERS", 10))) as pool:
        res = dict(pool.map(run, [(Path(root).resolve(), s, o, deals) for s in STACKS for o in OPPS]))
    np.savez_compressed(out_dir / f"{label}_{SEED}.npz", **res)
    for k, r in res.items():
        print(label, k, f"{r.mean():+.1f} ± {r.std(ddof=1) / np.sqrt(len(r)):.1f}")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    by_stack = {}
    for k in sorted(set(a.files) & set(b.files), key=lambda k: (-float(k.split("|")[0]), k)):
        d = b[k] - a[k]
        by_stack.setdefault(k.split("|")[0], []).append(d)
        print(f"{k:16} B-A {d.mean():+6.1f} ± {d.std(ddof=1) / np.sqrt(len(d)):4.1f}")
    for stack, ds in by_stack.items():
        p = np.concatenate(ds)
        se = p.std(ddof=1) / np.sqrt(len(p))
        z = f"z {p.mean() / se:+.1f}" if se > 0 else "identical"
        print(f"{stack:>5}bb pooled B-A {p.mean():+6.1f} ± {se:4.1f}  {z}")


if __name__ == "__main__":
    if sys.argv[1] == "play":
        play(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        compare(sys.argv[2], sys.argv[3])
