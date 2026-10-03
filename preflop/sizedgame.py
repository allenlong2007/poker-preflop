"""Flop, turn and river with MULTIPLE bet sizes, all decisions learned (experiment 13).

The line studied is the same as experiments 8, 11 and 12: BTN opens, BB calls;
BB checks the flop and the BTN c-bets or checks; if the BB calls, the turn and
river are played. Each street is one learned game:

    BB:   check | lead (size k)                      (no lead on the flop)
      after a check:  BTN  check | bet (size k)
                      BB   fold | call | check-raise to 3x  (separately for each k)
                      BTN  fold | call
      after a lead:   BTN  fold | call | raise to 3x
                      BB   fold | call

Streets are learned in order. The flop is learned with the rules playing the
turn and river; the turn is learned given the flop strategy (rules play the
river); the river is learned given both. Later streets remember the sizes
chosen earlier: the turn plays a separate game for each flop c-bet size, and
the river for each (flop size, turn size) pair, since the pot is different.
Turn and river use the line where the BB called the c-bet, then check-called
the turn bet (as in experiments 11 and 12).

Learning is CFR+ per (texture, earlier sizes, hand bucket), with Monte Carlo
rollouts for the streets that follow (exact showdowns on the river).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import eval7
import numpy as np

from .cards import HANDS, n_combos
from .flopgame import BUCKETS, TEXTURES as FLOP_TEXTURES, _regret_match, bucket, texture
from .postflop import COMBOS, DECK, play_streets
from .rivergame import BUCKETS as RIVER_BUCKETS, TEXTURES as RIVER_TEXTURES, river_bucket, river_texture
from .turngame import TEXTURES as TURN_TEXTURES, turn_texture

SIZES = {"flop": (1 / 3, 3 / 4), "turn": (1 / 2, 1.0), "river": (1 / 2, 1.0, 3 / 2)}
RAISE_X = 3.0
N_TEX = {"flop": len(FLOP_TEXTURES), "turn": len(TURN_TEXTURES), "river": len(RIVER_TEXTURES)}
NB = len(BUCKETS)
NODES = ["bb_first", "btn_vs_check", "bb_vs_bet", "btn_vs_xr", "btn_vs_lead", "bb_vs_raise"]
WHO = ["bb", "btn", "bb", "btn", "btn", "bb"]                 # whose hand bucket each node uses


def n_lines(street, sizes):
    if street == "flop":
        return 1
    if street == "turn":
        return len(sizes["flop"])
    return len(sizes["flop"]) * len(sizes["turn"])


def node_shapes(street, sizes):
    K, T = len(sizes[street]), N_TEX[street] * n_lines(street, sizes)
    return [(T, NB, 1 + K), (T, NB, 1 + K), (T, NB, K, 3), (T, NB, K, 2), (T, NB, K, 3), (T, NB, K, 2)]


def tree(v, p):
    """Values of every decision in the multi-size tree.

    v: leaf values in BTN share units. Scalars: "cc" (check-check), "lf" (BTN folds to a
       lead). Arrays over sizes k: "bf" (BB folds to bet k), "bc" (bet / lead k called),
       "xf" (BTN folds to a check-raise of bet k), "xc" (raise of size k called),
       "rf" (BB folds to the BTN's raise of lead k).
    p: node policies for this hand: p0 (1+K), p1 (1+K), p2 (K,3), p3 (K,2), p4 (K,3), p5 (K,2).
    Returns (action values per node -- BB nodes negated --, hand value, reach per node).
    """
    K = len(v["bc"])
    ev3 = np.stack([v["xf"], v["xc"]], -1)
    v3 = (p[3] * ev3).sum(-1)
    ev2 = -np.stack([v["bf"], v["bc"], v3], -1)
    v2 = -(p[2] * ev2).sum(-1)
    ev1 = np.concatenate([[v["cc"]], v2])
    v1 = p[1] @ ev1
    ev5 = -np.stack([v["rf"], v["xc"]], -1)
    v5 = -(p[5] * ev5).sum(-1)
    ev4 = np.stack([np.full(K, v["lf"]), v["bc"], v5], -1)
    v4 = (p[4] * ev4).sum(-1)
    ev0 = -np.concatenate([[v1], v4])
    value = -(p[0] @ ev0)
    r2 = p[0][0] * p[1][1:]
    r4 = p[0][1:]
    reach = (1.0, p[0][0], r2, r2 * p[2][:, 2], r4, r4 * p[4][:, 2])
    return (ev0, ev1, ev2, ev3, ev4, ev5), value, reach


def _leaves(P, pin, left, fracs, u, cont):
    """Leaf values for a street entered with pot P, BTN postflop chips pin, stacks `left`."""
    b = np.array([min(f * P, left) for f in fracs])
    R = np.minimum(RAISE_X * b, left)
    return {"cc": cont(P, pin, left),
            "bc": np.array([cont(P + 2 * bk, pin + bk, left - bk) for bk in b]),
            "xc": np.array([cont(P + 2 * Rk, pin + Rk, left - Rk) for Rk in R]),
            "bf": (P - pin) / u * np.ones(len(b)), "xf": (-pin - b) / u,
            "lf": -pin / u, "rf": (P + b - pin) / u}


def _policies_at(pol, t, tb, bk):
    return [pol[0][t, bk], pol[1][t, tb], pol[2][t, bk], pol[3][t, tb], pol[4][t, tb], pol[5][t, bk]]


def _worker(job):
    street, btn_w, bb_w, n, s, stack, seed, sizes, prior, pol, rules, record = job
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    shapes = node_shapes(street, sizes)
    acc = [np.zeros(sh) for sh in shapes]
    wts = [np.zeros(sh[:-1]) for sh in shapes]
    rec = []
    u, rest = 2 * s, stack - s
    fl, tu = sizes["flop"], sizes["turn"]
    for i, j in zip(bi, bj):
        btn = COMBOS[i][rng.integers(len(COMBOS[i]))]
        bb = COMBOS[j][rng.integers(len(COMBOS[j]))]
        if set(btn) & set(bb):
            continue
        dead = set(btn) | set(bb)
        live = [c for c in DECK if c not in dead]
        board = [live[k] for k in rng.choice(len(live), size=5, replace=False)]
        holes = {"btn": btn, "bb": bb}
        flop = board[:3]
        ft, fb_btn, fb_bb = texture(flop), bucket(btn, flop), bucket(bb, flop)
        hseed = int(rng.integers(1 << 31))

        # Entry states for this street: (line index, reach weight, pot, BTN chips in, stacks left).
        if street == "flop":
            entries = [(0, 1.0, 2 * s, 0.0, rest)]
        else:
            fpol = prior["flop"]
            entries = []
            for kf, f in enumerate(fl):
                c = min(f * 2 * s, rest)
                w = fpol[1][ft, fb_btn][1 + kf] * fpol[2][ft, fb_bb][kf, 1]     # c-bet size kf, BB calls
                if street == "turn":
                    entries.append((kf, w, 2 * s + 2 * c, c, rest - c))
                    continue
                tpol = prior["turn"]
                turn4 = board[:4]
                tt = turn_texture(flop, board[3]) * len(fl) + kf
                tb_btn, tb_bb = bucket(btn, turn4), bucket(bb, turn4)
                P1 = 2 * s + 2 * c
                for kt, g in enumerate(tu):
                    tbet = min(g * P1, rest - c)
                    w2 = w * tpol[0][tt, tb_bb][0] * tpol[1][tt, tb_btn][1 + kt] * tpol[2][tt, tb_bb][kt, 1]
                    entries.append((kf * len(tu) + kt, w2, P1 + 2 * tbet, c + tbet, rest - c - tbet))

        if street == "flop":
            tex, tb, bk = ft, fb_btn, fb_bb
            nxt = (1, 2)
        elif street == "turn":
            tex, tb, bk = turn_texture(flop, board[3]), bucket(btn, board[:4]), bucket(bb, board[:4])
            nxt = (2,)
        else:
            tex, tb, bk = river_texture(board[:4], board[4]), river_bucket(btn, board), river_bucket(bb, board)
            v_b, v_o = eval7.evaluate(list(btn) + board), eval7.evaluate(list(bb) + board)
            r = 1.0 if v_b > v_o else 0.5 if v_b == v_o else 0.0
        if street != "river":
            r = np.nan

        nl = n_lines(street, sizes)
        for line, w, P, pin, left in entries:
            if w < 1e-4:
                continue
            if street == "river":
                cont = lambda pot, pin_t, lft: (r * pot - pin_t) / u
            else:
                def cont(pot, pin_t, lft):
                    state = {"pot": pot, "left": {"btn": lft, "bb": lft}, "put_in": {"btn": pin_t, "bb": pin_t}}
                    return play_streets(holes, board, s, state, nxt, np.random.default_rng(hseed), **rules)[0]
            v = _leaves(P, pin, left, sizes[street], u, cont)
            t = tex * nl + line
            p = _policies_at(pol, t, tb, bk)
            evs, value, reach = tree(v, p)
            for m in range(6):
                cell = (t, bk) if WHO[m] == "bb" else (t, tb)
                rch = np.asarray(reach[m])
                acc[m][cell] += w * (rch[..., None] * evs[m] if rch.ndim else rch * evs[m])
                wts[m][cell] += w * rch
            if record:
                rec.append((i, j, t, tb, bk, w, value, r, ft))
    return acc, wts, np.array(rec).reshape(-1, 9)


COLUMNS = ["btn_class", "bb_class", "texture", "btn_bucket", "bb_bucket", "weight", "share", "btn_wins",
           "flop_texture"]


def _run(street, btn_range, bb_range, n, s, stack, seed, sizes, prior, pol, rules, record):
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(street, cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, sizes, prior, pol,
             rules, record) for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_worker, jobs))
    acc = [sum(p[0][m] for p in parts) for m in range(6)]
    wts = [sum(p[1][m] for p in parts) for m in range(6)]
    return acc, wts, np.concatenate([p[2] for p in parts])


def learn_street(street, btn_range, bb_range, rules, prior=None, sizes=SIZES, s=2.25, stack=100,
                 rounds=50, hands_per_round=60_000, n=600_000, seed=0, verbose=True):
    """Learn one street with CFR+, then record n hands. Returns {policy, hands, history}."""
    shapes = node_shapes(street, sizes)
    regret = [np.zeros(sh) for sh in shapes]
    avg = [np.zeros(sh) for sh in shapes]
    check_only = np.zeros(shapes[0])
    check_only[..., 0] = 1
    history = []
    for k in range(1, rounds + 1):
        cur = [_regret_match(r) for r in regret]
        if street == "flop":
            cur[0] = check_only                          # the BB never leads the flop
        acc, wts, _ = _run(street, btn_range, bb_range, hands_per_round, s, stack, seed * 100 + k, sizes,
                           prior, cur, rules, False)
        for m in range(6):
            ev = acc[m] / np.maximum(wts[m], 1e-9)[..., None]
            gain = ev - (cur[m] * ev).sum(-1, keepdims=True)
            regret[m] = np.maximum(regret[m] + wts[m][..., None] * gain, 0)
            avg[m] += k * cur[m]
        bet = (wts[1][..., None] * cur[1][..., 1:]).sum((0, 1)) / max(wts[1].sum(), 1e-9)
        history.append({"round": k, **{f"btn_bets_size{q}": b for q, b in enumerate(bet)}})
        if verbose and (k % 10 == 0 or k == 1):
            print(f"  {street} round {k}: BTN bets (by size, when checked to) "
                  + " / ".join(f"{b:.0%}" for b in bet))
    final = [a / a.sum(-1, keepdims=True) for a in avg]
    if street == "flop":
        final[0] = check_only
    _, _, rec = _run(street, btn_range, bb_range, n, s, stack, seed + 999, sizes, prior, final, rules, True)
    return {"policy": final, "hands": rec, "history": history}
