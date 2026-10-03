"""Flop c-bet / check-raise game with LEARNED decisions; turn and river use the rules.

Flop, single-raised pot (BTN opened, BB called):

    BB:   always checks
    BTN:  check | c-bet 1/3 pot
    BB:   fold | call | check-raise to 3x the c-bet          (only after a c-bet)
    BTN:  fold | call                                        (only after a check-raise)

Turn and river are played with the pot-odds / balanced rules in postflop.py.

Every flop decision is made per (board texture, hand bucket). The policies are
learned by regret matching (CFR+, as in the preflop solver) with Monte Carlo rollouts:
  * for each simulated hand, every flop line is played out to the end
    (check, c-bet & fold, c-bet & call, c-bet & raise & fold, c-bet & raise & call),
    so the value of each option is known under the current policies;
  * values are averaged per (texture, bucket), weighted by how often the opponent
    lets the player reach that decision;
  * options that would have done better than the current mix gain regret and get
    played more next round; the average policy over rounds is the answer.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import eval7
import numpy as np

from .cards import HANDS, n_combos
from .postflop import (CHECK_CALL, CHECK_CHECK, CHECK_FOLD, COMBOS, DECK, _has_draw, play_streets,
                       simulate as rules_simulate)

CBET = 1 / 3            # flop c-bet, fraction of the pot
RAISE_X = 3.0           # check-raise to this multiple of the c-bet

BUCKETS = ["two pair+", "overpair", "top pair, T+ kicker", "top pair, weak kicker", "weaker pair",
           "draw", "two overcards", "air"]
HIGH = ["A-high", "K-high", "Q/J-high", "T-high or lower"]
SUITS = ["rainbow", "two-tone", "monotone"]
TEXTURES = [(h, p, su, c) for h in HIGH for p in ("unpaired", "paired") for su in SUITS
            for c in ("disconnected", "connected")]
TEX_INDEX = {t: k for k, t in enumerate(TEXTURES)}
NT, NB = len(TEXTURES), len(BUCKETS)

# Extra BB action codes for the flop (turn / river keep postflop.py's codes).
CHECK_RAISE_FOLD, CHECK_RAISE_CALLED = 8, 9


def texture(flop):
    ranks = sorted((c.rank for c in flop), reverse=True)
    top = ranks[0]
    high = HIGH[0] if top == 12 else HIGH[1] if top == 11 else HIGH[2] if top >= 9 else HIGH[3]
    paired = len(set(ranks)) < 3
    suits = SUITS[max(sum(c.suit == su for c in flop) for su in range(4)) - 1]
    low = [r if r != 12 else -1 for r in ranks]           # ace can play low
    connected = not paired and (max(ranks) - min(ranks) <= 4 or max(low) - min(low) <= 4)
    return TEX_INDEX[(high, "paired" if paired else "unpaired", suits,
                      "connected" if connected else "disconnected")]


def bucket(hole, flop):
    cards = list(hole) + flop
    typ = eval7.evaluate(cards) >> 24
    board_typ = eval7.evaluate(flop) >> 24
    ranks = [c.rank for c in flop]
    top = max(ranks)
    h1, h2 = hole[0].rank, hole[1].rank
    if (typ >= 3 and typ > board_typ) or (typ == 2 and board_typ == 0):
        return 0
    if h1 == h2:
        return 1 if h1 > top else 4
    if top in (h1, h2):
        kicker = h2 if h1 == top else h1
        return 2 if kicker >= 8 else 3
    if h1 in ranks or h2 in ranks:
        return 4
    if _has_draw(hole, flop):
        return 5
    if min(h1, h2) > top:
        return 6
    return 7


def _rollout(holes, board, s, pot, left, put_in, seed, rules):
    """Turn + river with the rules from a given flop outcome. Returns (BTN share, codes)."""
    rng = np.random.default_rng(seed)   # same seed for every line: common random numbers
    state = {"pot": pot, "left": left, "put_in": put_in}
    return play_streets(holes, board, s, state, (1, 2), rng, **rules)


def _worker(job):
    btn_w, bb_w, n, s, stack, seed, pol, rules, record = job
    cbet, bb_pol, btn_xr = pol
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    acc_btn = np.zeros((NT, NB, 2)); w_btn = np.zeros((NT, NB))
    acc_bb = np.zeros((NT, NB, 3)); w_bb = np.zeros((NT, NB))
    acc_xr = np.zeros((NT, NB, 2)); w_xr = np.zeros((NT, NB))
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
        t, tb, bbk = texture(flop), bucket(btn, flop), bucket(bb, flop)
        holes = {"btn": btn, "bb": bb}
        P, rest = 2 * s, stack - s
        c = min(CBET * P, rest)
        R = min(RAISE_X * c, rest)
        hs = int(rng.integers(1 << 31))
        v_check, codes_check = _rollout(holes, board, s, P, {"btn": rest, "bb": rest},
                                        {"btn": 0.0, "bb": 0.0}, hs, rules)
        v_bb_folds = P / (2 * s)                                   # BTN takes the pot
        v_call, codes_call = _rollout(holes, board, s, P + 2 * c, {"btn": rest - c, "bb": rest - c},
                                      {"btn": c, "bb": c}, hs, rules)
        v_xr_fold = -c / (2 * s)                                   # BTN gives up its c-bet
        v_xr_call, codes_xr = _rollout(holes, board, s, P + 2 * R, {"btn": rest - R, "bb": rest - R},
                                       {"btn": R, "bb": R}, hs, rules)

        p_bet = cbet[t, tb]
        q = bb_pol[t, bbk]                                          # BB: fold, call, raise
        x = btn_xr[t, tb]                                           # BTN vs raise: fold, call
        ev_xr = np.array([v_xr_fold, v_xr_call])                    # BTN's view
        v_raise = x @ ev_xr
        ev_bb = -np.array([v_bb_folds, v_call, v_raise])            # BB's view (minus BTN share)
        v_bet = -(q @ ev_bb)
        ev_btn = np.array([v_check, v_bet])
        acc_btn[t, tb] += ev_btn; w_btn[t, tb] += 1
        acc_bb[t, bbk] += p_bet * ev_bb; w_bb[t, bbk] += p_bet
        acc_xr[t, tb] += p_bet * q[2] * ev_xr; w_xr[t, tb] += p_bet * q[2]

        if record:
            share = (1 - p_bet) * v_check + p_bet * v_bet          # expected over flop choices
            # Sample one actual flop line for the action codes.
            if rng.random() >= p_bet:
                flop_code, rest_codes = CHECK_CHECK, codes_check
            else:
                a = rng.choice(3, p=q)
                if a == 0:
                    flop_code, rest_codes = CHECK_FOLD, [0, 0, 0]
                elif a == 1:
                    flop_code, rest_codes = CHECK_CALL, codes_call
                else:
                    called = rng.random() < x[1]
                    flop_code = CHECK_RAISE_CALLED if called else CHECK_RAISE_FOLD
                    rest_codes = codes_xr if called else [0, 0, 0]
            rec.append((i, j, t, tb, bbk, p_bet, q[0], q[1], q[2], x[1], share,
                        flop_code, rest_codes[1], rest_codes[2]))
    return acc_btn, w_btn, acc_bb, w_bb, acc_xr, w_xr, np.array(rec).reshape(-1, 14)


def _run(btn_range, bb_range, n, s, stack, seed, pol, rules, record):
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, pol, rules, record)
            for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, jobs))
    sums = [sum(p[k] for p in parts) for k in range(6)]
    return sums, np.concatenate([p[6] for p in parts])


def _regret_match(R):
    pos = np.maximum(R, 0)
    tot = pos.sum(-1, keepdims=True)
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1 / R.shape[-1])


def learn_and_play(btn_range, bb_range, s=2.25, stack=100, rounds=30, hands_per_round=60_000,
                   n=600_000, seed=0, verbose=True):
    """Learn the flop policies with CFR+ (regret matching), then play n hands with them.

    Each round plays hands_per_round hands with the current policies and measures,
    per (texture, bucket), the average value of every flop option. Regret for an
    option = how much better it would have done than the current mix; the next
    policy plays options in proportion to positive regret. The reported policy is
    the average over rounds (later rounds weigh more), which is what converges.

    Returns a dict: policy (cbet [NT,NB] = c-bet probability, bb [NT,NB,3] = fold /
    call / raise, btn_vs_xr [NT,NB,2] = fold / call), hands (per-hand records, see
    COLUMNS), history (per-round summary), rules (turn/river tables used).
    """
    warm = rules_simulate(btn_range, bb_range, n=os.cpu_count(), s=s, stack=stack, seed=seed + 7,
                          learn_hands=150_000)
    rules = {"table": warm["table"], "bluff_probs": warm["bluff_rates"]}
    shapes = ((NT, NB, 2), (NT, NB, 3), (NT, NB, 2))
    regret = [np.zeros(sh) for sh in shapes]
    avg = [np.zeros(sh) for sh in shapes]
    history = []
    for k in range(1, rounds + 1):
        cur = [_regret_match(r) for r in regret]
        pol = (cur[0][..., 1], cur[1], cur[2])
        (ab, wb, ac, wc, ax, wx), _ = _run(btn_range, bb_range, hands_per_round, s, stack,
                                           seed * 100 + k, pol, rules, False)
        for m, (acc, w) in enumerate(((ab, wb), (ac, wc), (ax, wx))):
            ev = acc / np.maximum(w, 1e-9)[..., None]                 # average value per option
            gain = ev - (cur[m] * ev).sum(-1, keepdims=True)
            regret[m] = np.maximum(regret[m] + w[..., None] * gain, 0)   # CFR+
            avg[m] += k * cur[m]
        history.append({"round": k, "btn_cbet_pct": (wb * pol[0]).sum() / wb.sum(),
                        "bb_xr_pct_vs_cbet": (wc * pol[1][..., 2]).sum() / max(wc.sum(), 1e-9)})
        if verbose and (k % 5 == 0 or k == 1):
            print(f"  flop round {k}: BTN c-bets {history[-1]['btn_cbet_pct']:.0%}, "
                  f"BB check-raises {history[-1]['bb_xr_pct_vs_cbet']:.0%} of c-bets")
    final = [a / a.sum(-1, keepdims=True) for a in avg]
    pol = (final[0][..., 1], final[1], final[2])
    _, rec = _run(btn_range, bb_range, n, s, stack, seed + 999, pol, rules, True)
    return {"policy": pol, "hands": rec, "history": history, "rules": rules}


COLUMNS = ["btn_class", "bb_class", "texture", "btn_bucket", "bb_bucket", "p_cbet", "bb_fold",
           "bb_call", "bb_raise", "btn_call_vs_xr", "share", "flop_code", "turn_code", "river_code"]
