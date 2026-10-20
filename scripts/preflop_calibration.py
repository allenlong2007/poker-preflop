"""Experiment 33: make the preflop model value each pot the way the bot actually plays it postflop.

The match's preflop tree (experiment 26) valued pots that see a flop with a realization formula plus
experiment 24's per-hand shape. Checked against the whole-hand model that the bot plays postflop, it gave
the BTN far less of every pot:

    pot            preflop model   whole-hand model
    srp (3bb open)     +7.7%            +17.3%      (BTN net, share of the pot)
    srp (2.25bb)      +12.5%            +23.7%
    limped            +12.2%            +23.4%
    iso-raised         -6.4%             -1.9%
    3-bet              -6.4%             -5.7%

and the BTN's overall EV came out negative (-7.7 bb/100), which no heads-up equilibrium does. A BTN that
undervalues its position folds and limps too often and opens big to avoid postflop play.

Here each pot type gets a level correction (an extra BTN share of the pot, preflop/shortstack.py `levels`),
fitted so the preflop model's BTN net in that pot equals the whole-hand model's, and the open-size /
3-bet-size / iso-size grid is solved again with it.

Usage:
    python scripts/preflop_calibration.py          (~10 min; writes output/match/calibration.json)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import match_strategy as ms
from preflop.shortstack import solve_short

OUT = ROOT / "output" / "match"
POL = ms.POLICY_FINE
ITERS = 1500
# One step: the correction is the gap measured on the current ranges. Iterating to hit the targets doesn't
# settle: a richer BTN pot makes the BB defend tighter, shrinking the share again, while the targets were
# measured on the old ranges (experiment 24 saw the same). A true fixed point needs the postflop model
# retrained on the new ranges, then measured again.
DAMP = 1.0
ROUNDS = 1


def target_shares():
    """The whole-hand model's BTN net per pot (share of the pot) for each pot type, from the trained policies."""
    t = {}
    for pot in ("srp", "srp_alt", "3bet", "limp", "iso"):
        d = np.load(POL / f"{pot}.npz")
        s = float(d["pot_to"])
        t[pot] = ((float(d["value_btn"]) - s) / (2 * s), s)
    return t


def srp_target(s, t):
    """BTN net share in a single-raised pot opened to s: linear between the two trained open sizes."""
    (a, sa), (b, sb) = t["srp_alt"], t["srp"]
    return a + (s - sa) * (b - a) / (sb - sa)


def pot_shares(res, s, t3, r):
    """The preflop model's BTN net share of the pot in each pot type, under its own strategy."""
    root, W = res.root, res.W
    A = lambda name, a: res.actions(name)[a]
    def avg(mat, pb, po, to):
        M = W * pb[:, None] * po[None, :]
        return float((M * mat).sum() / max(M.sum(), 1e-12)) / (2 * to)
    ch = lambda *p: _child(root, *p)
    raise_, limp = A("btn_root", "raise"), A("btn_root", "limp")
    return {
        "srp": avg(ch("raise", "call"), raise_, A("bb_vs_raise", "call"), s),
        "3bet": avg(ch("raise", "3-bet", "call"), raise_ * A("btn_vs_3bet_small", "call"), A("bb_vs_raise", "3-bet"), t3),
        "limp": avg(ch("limp", "check"), limp, A("bb_vs_limp", "check"), 1.0),
        "iso": avg(ch("limp", "raise", "call"), limp * A("btn_vs_iso", "call"), A("bb_vs_limp", "raise"), r),
    }


def _child(node, *path):
    for p in path:
        node = node.children[p]
    return node


def _solve(job):
    s, t3, r, levels, iters = job
    return solve_short(ms.E, ms.W, s, r, ms.STACK, iters, r_oop=ms.R_OOP, alpha=ms.ALPHA, t3=t3,
                       adj_ip=ms.A_IP, adj_oop=ms.A_OOP, levels=levels)


def _ev(job):
    return _solve(job).btn_ev()


def calibrate(s, t3, r, targets):
    """The level corrections for one (open, 3-bet, iso) spot: the whole-hand minus the preflop share."""
    want = {"srp": srp_target(s, targets), "3bet": targets["3bet"][0], "limp": targets["limp"][0],
            "iso": targets["iso"][0]}
    levels = {k: 0.0 for k in want}
    for _ in range(ROUNDS):
        have = pot_shares(_solve((s, t3, r, levels, ITERS)), s, t3, r)
        levels = {k: levels[k] + DAMP * (want[k] - have[k]) for k in want}
    have = pot_shares(_solve((s, t3, r, levels, ITERS)), s, t3, r)
    return levels, {k: (want[k], have[k]) for k in want}


def main():
    t0 = time.time()
    targets = target_shares()
    old = json.load(open(ms.CACHE / "preflop.json"))
    rows, fits = [], {}
    # Calibrate at each open's current best BB response, then let the BB re-pick its sizes with the levels on.
    with Pool(12) as pool:
        cal = pool.starmap(calibrate, [(s, old["best"][str(s)]["t"], old["best"][str(s)]["r"], targets)
                                       for s in ms.OPENS])
    for s, (levels, check) in zip(ms.OPENS, cal):
        fits[s] = levels
        print(f"open {s}: levels " + ", ".join(f"{k} {v:+.3f}" for k, v in levels.items()) + " | fit " +
              ", ".join(f"{k} {w:+.1%}->{h:+.1%}" for k, (w, h) in check.items()), flush=True)
    grid = [(s, round(m * s, 4), r) for s in ms.OPENS for m in ms.T_MULT for r in ms.ISO]
    with Pool(12) as pool:
        evs = dict(zip(grid, pool.map(_ev, [(s, t, r, fits[s], ITERS) for s, t, r in grid])))
    best = {}
    for s in ms.OPENS:
        cand = {(t, r): v for (s2, t, r), v in evs.items() if s2 == s}
        t, r = min(cand, key=cand.get)
        best[s] = {"t": t, "r": r, "btn_ev": cand[(t, r)], "levels": fits[s]}
    s_main = max(best, key=lambda s: best[s]["btn_ev"])
    final = _solve((s_main, best[s_main]["t"], best[s_main]["r"], fits[s_main], 5000))
    w = np.array([ms.n_combos(h) for h in ms.HANDS], float)
    freq = {a: float(np.dot(w, v) / w.sum()) for a, v in final.actions("btn_root").items()}
    for s in ms.OPENS:
        b = best[s]
        print(f"open {s:>4}: BB 3-bet to {b['t']:>5}, iso {b['r']}, BTN EV {b['btn_ev'] * 100:+.2f} bb/100"
              f"{'   <- best' if s == s_main else ''}")
    print("BTN first-in at the best open:", {k: f"{v:.1%}" for k, v in freq.items()})
    json.dump({"s_main": s_main, "best": {str(k): v for k, v in best.items()}, "btn_root": freq,
               "targets": {k: v[0] for k, v in targets.items()},
               "old": {"s_main": old["s_main"], "best": old["best"]}},
              open(OUT / "calibration.json", "w"), indent=1)
    print(f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
