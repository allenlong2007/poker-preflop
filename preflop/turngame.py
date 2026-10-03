"""Turn game after the BTN c-bets the flop and the BB calls, with LEARNED decisions.

Turn (pot = preflop pot + both flop c-bet amounts):

    BB:   check | lead 2/3 pot
      after a check:  BTN  check | bet 2/3 pot
                      BB   fold | call | check-raise to 3x     (after a bet)
                      BTN  fold | call                         (after a check-raise)
      after a lead:   BTN  fold | call | raise to 3x
                      BB   fold | call                         (after a raise)

The river uses the pot-odds / balanced rules (postflop.py).

Which hands get here comes from a flop policy (flopgame.py): every simulated hand
is weighted by P(BTN c-bets) x P(BB calls) for its flop texture and buckets.
Decisions are learned per (turn texture, hand bucket) with CFR+ and Monte Carlo
rollouts of the river, exactly like the flop game.

Turn texture = what the turn card did x whether the flop was dry or wet:
    turn card: blank, overcard, pairs the board, flush card, straight card
    flop:      dry (rainbow, unpaired, disconnected) or wet (anything else)
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .cards import HANDS, n_combos
from .flopgame import BUCKETS, CBET, TEXTURES as FLOP_TEXTURES, _regret_match, bucket, texture
from .postflop import COMBOS, DECK, play_streets

BET = 2 / 3
RAISE_X = 3.0
TURN_CARDS = ["blank", "overcard", "pairs the board", "flush card", "straight card"]
FLOPS = ["dry flop", "wet flop"]
TEXTURES = [(tc, fl) for tc in TURN_CARDS for fl in FLOPS]
NT, NB = len(TEXTURES), len(BUCKETS)

# Decision nodes: (name, who decides, actions)
NODES = [("bb_first", "bb", ["check", "lead"]),
         ("btn_vs_check", "btn", ["check", "bet"]),
         ("bb_vs_bet", "bb", ["fold", "call", "check-raise"]),
         ("btn_vs_xr", "btn", ["fold", "call"]),
         ("btn_vs_lead", "btn", ["fold", "call", "raise"]),
         ("bb_vs_raise", "bb", ["fold", "call"])]

COLUMNS = ["btn_class", "bb_class", "texture", "btn_bucket", "bb_bucket", "weight", "bb_lead",
           "btn_bet", "bb_fold", "bb_call", "bb_xr", "btn_call_vs_xr", "btn_fold_vs_lead",
           "btn_call_vs_lead", "btn_raise_vs_lead", "bb_call_vs_raise", "share", "flop_texture"]


def _straight_possible(ranks):
    ranks = set(ranks)
    if 12 in ranks:
        ranks.add(-1)
    return any(len(ranks & set(range(lo, lo + 5))) >= 3 for lo in range(-1, 9))


def turn_texture(flop, turn):
    ranks = [c.rank for c in flop]
    if turn.rank in ranks:
        kind = "pairs the board"
    elif sum(c.suit == turn.suit for c in flop) >= 2:
        kind = "flush card"
    elif _straight_possible(ranks + [turn.rank]) and not _straight_possible(ranks):
        kind = "straight card"
    elif turn.rank > max(ranks):
        kind = "overcard"
    else:
        kind = "blank"
    _, paired, suits, connected = FLOP_TEXTURES[texture(flop)]
    dry = paired == "unpaired" and suits == "rainbow" and connected == "disconnected"
    return TEXTURES.index((kind, FLOPS[0] if dry else FLOPS[1]))


def _river(holes, board, s, pot, left, put_in, seed, rules):
    rng = np.random.default_rng(seed)          # same seed for every line: common random numbers
    return play_streets(holes, board, s, {"pot": pot, "left": left, "put_in": put_in}, (2,), rng, **rules)[0]


def line_values(v, p):
    """Values of every decision in the check/bet/raise tree, given the leaf values.

    v: leaf values in BTN share units -- check_check, bb_folds_to_bet, bet_called,
       btn_folds_to_xr, raise_called, btn_folds_to_lead, bb_folds_to_raise. (A called
       lead and a called bet put the same chips in, as do a called raise and a called
       check-raise, since both raises are 3x the same bet size.)
    p: the 6 node policies for this hand, in NODES order.
    Returns (action values per node -- BB nodes from the BB's side, i.e. negated --,
    the hand's value for the BTN, each node's reach probability given the hand gets here).
    """
    ev3 = np.array([v["btn_folds_to_xr"], v["raise_called"]]); v3 = p[3] @ ev3
    ev2 = -np.array([v["bb_folds_to_bet"], v["bet_called"], v3]); v2 = -(p[2] @ ev2)
    ev1 = np.array([v["check_check"], v2]); v1 = p[1] @ ev1
    ev5 = -np.array([v["bb_folds_to_raise"], v["raise_called"]]); v5 = -(p[5] @ ev5)
    ev4 = np.array([v["btn_folds_to_lead"], v["bet_called"], v5]); v4 = p[4] @ ev4
    ev0 = -np.array([v1, v4]); value = -(p[0] @ ev0)
    reach = [1.0, p[0][0], p[0][0] * p[1][1], p[0][0] * p[1][1] * p[2][2], p[0][1], p[0][1] * p[4][2]]
    return (ev0, ev1, ev2, ev3, ev4, ev5), value, reach


def _worker(job):
    btn_w, bb_w, n, s, stack, seed, flop_pol, pol, rules, record = job
    f_cbet, f_bb = flop_pol
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
        flop = board[:3]
        ft = texture(flop)
        w = f_cbet[ft, bucket(btn, flop)] * f_bb[ft, bucket(bb, flop), 1]   # c-bet, then call
        if w < 1e-4:
            continue
        turn4 = board[:4]
        t, tb, bk = turn_texture(flop, board[3]), bucket(btn, turn4), bucket(bb, turn4)
        holes = {"btn": btn, "bb": bb}
        rest = stack - s
        c = min(CBET * 2 * s, rest)
        P1, rest1 = 2 * s + 2 * c, rest - c
        b = min(BET * P1, rest1)
        R = min(RAISE_X * b, rest1)
        seed_h = int(rng.integers(1 << 31))
        u = 2 * s
        v_check_check = _river(holes, board, s, P1, {"btn": rest1, "bb": rest1}, {"btn": c, "bb": c}, seed_h, rules)
        v_bet_called = _river(holes, board, s, P1 + 2 * b, {"btn": rest1 - b, "bb": rest1 - b},
                              {"btn": c + b, "bb": c + b}, seed_h, rules)
        v_raise_called = _river(holes, board, s, P1 + 2 * R, {"btn": rest1 - R, "bb": rest1 - R},
                                {"btn": c + R, "bb": c + R}, seed_h, rules)
        v_bb_folds_to_bet = (P1 - c) / u
        v_btn_folds_to_xr = (-c - b) / u
        v_btn_folds_to_lead = -c / u
        v_bb_folds_to_raise = (P1 + b - c) / u

        p = [pol[0][t, bk], pol[1][t, tb], pol[2][t, bk], pol[3][t, tb], pol[4][t, tb], pol[5][t, bk]]
        evs, value, reach = line_values({"check_check": v_check_check, "bb_folds_to_bet": v_bb_folds_to_bet,
                                         "bet_called": v_bet_called, "btn_folds_to_xr": v_btn_folds_to_xr,
                                         "raise_called": v_raise_called, "btn_folds_to_lead": v_btn_folds_to_lead,
                                         "bb_folds_to_raise": v_bb_folds_to_raise}, p)
        reach = [w * r for r in reach]
        cells = [bk, tb, bk, tb, tb, bk]
        for k, ev in enumerate(evs):
            acc[k][t, cells[k]] += reach[k] * ev
            wts[k][t, cells[k]] += reach[k]
        if record:
            rec.append((i, j, t, tb, bk, w, p[0][1], p[1][1], p[2][0], p[2][1], p[2][2], p[3][1],
                        p[4][0], p[4][1], p[4][2], p[5][1], value, ft))
    return acc, wts, np.array(rec).reshape(-1, len(COLUMNS))


def _run(btn_range, bb_range, n, s, stack, seed, flop_pol, pol, rules, record):
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, flop_pol, pol, rules, record)
            for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, jobs))
    acc = [sum(p[0][k] for p in parts) for k in range(len(NODES))]
    wts = [sum(p[1][k] for p in parts) for k in range(len(NODES))]
    return acc, wts, np.concatenate([p[2] for p in parts])


def cfr_learn(run_round, rounds, street, verbose=True):
    """Shared CFR+ loop for the turn and river games.

    run_round(policies, k) plays one round with the current policies and returns
    (action-value sums, reach weights) per node; run_round.nt is the number of
    textures. Returns (average policies, per-round history).
    """
    regret = [np.zeros((run_round.nt, NB, len(a))) for _, _, a in NODES]
    avg = [np.zeros_like(r) for r in regret]
    history = []
    for k in range(1, rounds + 1):
        cur = [_regret_match(r) for r in regret]
        acc, wts = run_round(cur, k)
        for m in range(len(NODES)):
            ev = acc[m] / np.maximum(wts[m], 1e-9)[..., None]
            gain = ev - (cur[m] * ev).sum(-1, keepdims=True)
            regret[m] = np.maximum(regret[m] + wts[m][..., None] * gain, 0)
            avg[m] += k * cur[m]
        lead = (wts[0] * cur[0][..., 1]).sum() / wts[0].sum()
        xr = (wts[2] * cur[2][..., 2]).sum() / max(wts[2].sum(), 1e-9)
        history.append({"round": k, "bb_lead_pct": lead, "bb_xr_vs_bet_pct": xr})
        if verbose and (k % 10 == 0 or k == 1):
            print(f"  {street} round {k}: BB leads {lead:.0%}, check-raises {xr:.0%} of BTN bets")
    return [a / a.sum(-1, keepdims=True) for a in avg], history


def learn_and_play(btn_range, bb_range, flop_pol, rules, s=2.25, stack=100, rounds=40,
                   hands_per_round=80_000, n=800_000, seed=0, verbose=True):
    """Learn the turn policies with CFR+, then play n hands. Returns {policy, hands, history}."""
    def run_round(cur, k):
        acc, wts, _ = _run(btn_range, bb_range, hands_per_round, s, stack, seed * 100 + k,
                           flop_pol, cur, rules, False)
        return acc, wts
    run_round.nt = NT
    final, history = cfr_learn(run_round, rounds, "turn", verbose)
    _, _, rec = _run(btn_range, bb_range, n, s, stack, seed + 999, flop_pol, final, rules, True)
    return {"policy": final, "hands": rec, "history": history}
