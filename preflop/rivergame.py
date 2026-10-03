"""River game after the BB check-calls a turn bet, with LEARNED decisions.

The line so far: BTN c-bets the flop, BB calls; on the turn the BB checks, the BTN
bets 2/3 pot, the BB calls. River (same tree as the turn game):

    BB:   check | lead 2/3 pot
      after a check:  BTN  check | bet 2/3 pot
                      BB   fold | call | check-raise to 3x     (after a bet)
                      BTN  fold | call                         (after a check-raise)
      after a lead:   BTN  fold | call | raise to 3x
                      BB   fold | call                         (after a raise)

Every line ends in a fold or a showdown, so each hand's values are exact (no
rollouts). Hands are weighted by how likely they are to reach this river under
the flop policy (c-bet, call) and the turn policy (check, bet, call).

River texture = what the river card did: blank, overcard, pairs the board,
flush card (puts a third card of a suit on board), straight card.
River hand buckets are the flop buckets, with "draw" becoming "missed draw":
a hand that had a flush draw or open-ender on the turn and made nothing better.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import eval7
import numpy as np

from .cards import HANDS, n_combos
from .flopgame import BUCKETS as FLOP_BUCKETS, CBET, bucket, texture
from .postflop import COMBOS, DECK
from .turngame import (BET as TURN_BET, NODES, _straight_possible, cfr_learn, line_values,
                       turn_texture)

BET = 2 / 3
RAISE_X = 3.0
TEXTURES = ["blank", "overcard", "pairs the board", "flush card", "straight card"]
BUCKETS = [b if b != "draw" else "missed draw" for b in FLOP_BUCKETS]
NT, NB = len(TEXTURES), len(BUCKETS)
DRAW = FLOP_BUCKETS.index("draw")

COLUMNS = ["btn_class", "bb_class", "texture", "btn_bucket", "bb_bucket", "weight", "bb_lead",
           "btn_bet", "bb_fold", "bb_call", "bb_xr", "btn_call_vs_xr", "btn_fold_vs_lead",
           "btn_call_vs_lead", "btn_raise_vs_lead", "bb_call_vs_raise", "share", "btn_wins"]


def river_texture(turn_board, river):
    ranks = [c.rank for c in turn_board]
    if river.rank in ranks:
        return TEXTURES.index("pairs the board")
    if sum(c.suit == river.suit for c in turn_board) >= 2:
        return TEXTURES.index("flush card")
    if _straight_possible(ranks + [river.rank]) and not _straight_possible(ranks):
        return TEXTURES.index("straight card")
    if river.rank > max(ranks):
        return TEXTURES.index("overcard")
    return TEXTURES.index("blank")


def river_bucket(hole, board):
    """Flop-style bucket on the full board; a turn draw that made nothing becomes 'missed draw'."""
    b = bucket(hole, board)
    turn_b = bucket(hole, board[:4])
    if b in (DRAW, 6, 7):                        # nothing made on the river (6 overcards, 7 air)
        return DRAW if turn_b == DRAW else (b if b != DRAW else 7)
    return b


def _worker(job):
    btn_w, bb_w, n, s, stack, seed, flop_pol, turn_pol, pol, record = job
    f_cbet, f_bb = flop_pol
    t_first, t_btn, t_bb = turn_pol
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    acc = [np.zeros((NT, NB, len(a))) for _, _, a in NODES]
    wts = [np.zeros((NT, NB)) for _ in NODES]
    rec = []
    for i, j in zip(bi, bj):
        btn = COMBOS[i][rng.integers(len(COMBOS[i]))]
        bb = COMBOS[j][rng.integers(len(COMBOS[j]))]
        if set(btn) & set(bb):
            continue
        dead = set(btn) | set(bb)
        live = [c for c in DECK if c not in dead]
        board = [live[k] for k in rng.choice(len(live), size=5, replace=False)]
        flop, turn4 = board[:3], board[:4]
        ft = texture(flop)
        w = f_cbet[ft, bucket(btn, flop)] * f_bb[ft, bucket(bb, flop), 1]          # flop: c-bet, call
        if w < 1e-4:
            continue
        tt, ttb, tbk = turn_texture(flop, board[3]), bucket(btn, turn4), bucket(bb, turn4)
        w *= t_first[tt, tbk, 0] * t_btn[tt, ttb, 1] * t_bb[tt, tbk, 1]           # turn: check, bet, call
        if w < 1e-4:
            continue
        t, tb, bk = river_texture(turn4, board[4]), river_bucket(btn, board), river_bucket(bb, board)
        v_b, v_o = eval7.evaluate(list(btn) + board), eval7.evaluate(list(bb) + board)
        r = 1.0 if v_b > v_o else 0.5 if v_b == v_o else 0.0                    # BTN's showdown result
        rest = stack - s
        c = min(CBET * 2 * s, rest)
        P1 = 2 * s + 2 * c
        tbet = min(TURN_BET * P1, rest - c)
        P2, pin, left = P1 + 2 * tbet, c + tbet, rest - c - tbet                 # river pot, BTN chips in
        B = min(BET * P2, left)
        R = min(RAISE_X * B, left)
        u = 2 * s
        leaves = {"check_check": (r * P2 - pin) / u, "bb_folds_to_bet": (P2 - pin) / u,
                  "bet_called": (r * (P2 + 2 * B) - pin - B) / u, "btn_folds_to_xr": (-pin - B) / u,
                  "raise_called": (r * (P2 + 2 * R) - pin - R) / u, "btn_folds_to_lead": -pin / u,
                  "bb_folds_to_raise": (P2 + B - pin) / u}
        p = [pol[0][t, bk], pol[1][t, tb], pol[2][t, bk], pol[3][t, tb], pol[4][t, tb], pol[5][t, bk]]
        evs, value, reach = line_values(leaves, p)
        cells = [bk, tb, bk, tb, tb, bk]
        for k, ev in enumerate(evs):
            acc[k][t, cells[k]] += w * reach[k] * ev
            wts[k][t, cells[k]] += w * reach[k]
        if record:
            rec.append((i, j, t, tb, bk, w, p[0][1], p[1][1], p[2][0], p[2][1], p[2][2], p[3][1],
                        p[4][0], p[4][1], p[4][2], p[5][1], value, r))
    return acc, wts, np.array(rec).reshape(-1, len(COLUMNS))


def _run(btn_range, bb_range, n, s, stack, seed, flop_pol, turn_pol, pol, record):
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, flop_pol, turn_pol, pol,
             record) for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, jobs))
    acc = [sum(p[0][k] for p in parts) for k in range(len(NODES))]
    wts = [sum(p[1][k] for p in parts) for k in range(len(NODES))]
    return acc, wts, np.concatenate([p[2] for p in parts])


def learn_and_play(btn_range, bb_range, flop_pol, turn_pol, s=2.25, stack=100, rounds=60,
                   hands_per_round=150_000, n=1_500_000, seed=0, verbose=True):
    """Learn the river policies with CFR+, then play n hands. Returns {policy, hands, history}.

    turn_pol: (BB first [NT,NB,2], BTN vs check [NT,NB,2], BB vs bet [NT,NB,3]) from turngame.
    """
    def run_round(cur, k):
        acc, wts, _ = _run(btn_range, bb_range, hands_per_round, s, stack, seed * 100 + k,
                           flop_pol, turn_pol, cur, False)
        return acc, wts
    run_round.nt = NT
    final, history = cfr_learn(run_round, rounds, "river", verbose)
    _, _, rec = _run(btn_range, bb_range, n, s, stack, seed + 999, flop_pol, turn_pol, final, True)
    return {"policy": final, "hands": rec, "history": history}
