"""Experiment 49: low stack-to-pot postflop decisions by range-vs-range expected value.

Replaces the fixed rule "shove with 55% equity, else check; call by price". At a low SPR every option is worked out
in chips against the opponent's actual range (bot/potodds.opponent_range, at the effective stack depth) and how that
range responds to the price we offer:

  check        showdown share of the pot (low SPR: little room for later betting)
  bet ~1/2 pot they fold, call, or jam back their strongest hands; facing the jam we call or fold by its price
  jam          they fold or call; a hand calls when its equity against our jamming range beats its price
  call / fold  by our equity against their betting range

Their response is decided per combo against the range they can put us on (our perceived range: the same estimator
run from their side, with our cards unknown). We jam / bet with the stronger half of it, so they fold the hands
that can't beat that at the price. Equities use eval7 on shared sampled runouts with card-conflict masks (exact on the
river). The chosen action is a soft maximum of the EVs, so near-equal options mix instead of always taking the same
line.
"""
import numpy as np
import eval7

DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
MAX_OPP, MAX_HERO = 260, 120      # combos kept (weighted subsample) for speed
RUNOUTS = {3: 30, 4: 24, 5: 1}     # sampled runouts by board size
BET_FRACTION = 0.5                 # the non-all-in bet at low SPR
RAISE_BACK_EQ = 0.62               # their equity vs our betting range above which they jam over a small bet
TEMPERATURE = 0.015                # softmax temperature as a share of the pot
BET_MIN_SPR = 1.0                  # below this many pots behind, a small bet commits anyway: jam or check
CALL_SLACK = 0.0                   # opponents call this much below their price (calling stations call wider)


def _subsample(rng_, n, g):
    if len(rng_) <= n:
        return rng_
    w = np.array([x[2] for x in rng_], float)
    idx = g.choice(len(rng_), size=n, replace=False, p=w / w.sum())
    return [(rng_[i][0], rng_[i][1], 1.0) for i in idx]


def _runouts(board, dead, g):
    need = 5 - len(board)
    if need == 0:
        return [()]
    rest = [c for c in DECK if c not in dead]
    if need == 1:
        cards = rest if len(rest) <= RUNOUTS[4] else [rest[i] for i in g.choice(len(rest), RUNOUTS[4], replace=False)]
        return [(c,) for c in cards]
    out = []
    for _ in range(RUNOUTS[3]):
        i, j = g.choice(len(rest), size=2, replace=False)
        out.append((rest[i], rest[j]))
    return out


def _values(combos, full):
    """Hand values on one full board; -1 where a combo shares a card with it."""
    used = set(full)
    return np.array([-1 if (a in used or b in used) else eval7.evaluate([a, b] + full) for a, b, _ in combos])


def equities(hole, board, opp, hero_range, seed=0):
    """eq_me[j]: our hand vs opponent combo j; M[j, i]: opponent combo j vs perceived-hero combo i (NaN if the
    combos share a card or never met on a valid runout)."""
    g = np.random.default_rng(seed)
    opp = _subsample(opp, MAX_OPP, g)
    hero_range = _subsample(hero_range, MAX_HERO, g)
    runs = _runouts(list(board), set(hole) | set(board), g)
    n_o, n_h = len(opp), len(hero_range)
    win_me, cnt_me = np.zeros(n_o), np.zeros(n_o)
    win_m, cnt_m = np.zeros((n_o, n_h)), np.zeros((n_o, n_h))
    for run in runs:
        full = list(board) + list(run)
        me = eval7.evaluate(list(hole) + full)
        vo, vh = _values(opp, full), _values(hero_range, full)
        ok = vo >= 0
        win_me += ok * ((me > vo) + 0.5 * (me == vo))
        cnt_me += ok
        pair = ok[:, None] & (vh >= 0)[None, :]
        win_m += pair * ((vo[:, None] > vh[None, :]) + 0.5 * (vo[:, None] == vh[None, :]))
        cnt_m += pair
    clash = np.array([[len({a, b} & {c, d}) > 0 for c, d, _ in hero_range] for a, b, _ in opp]) if n_h else \
        np.zeros((n_o, 0), bool)
    cnt_m[clash] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        eq_me = np.where(cnt_me > 0, win_me / np.maximum(cnt_me, 1), 0.5)
        m = np.where(cnt_m > 0, win_m / np.maximum(cnt_m, 1), np.nan)
    w_o = np.array([x[2] for x in opp], float)
    w_h = np.array([x[2] for x in hero_range], float)
    return eq_me, m, w_o, w_h


def _weighted_vs(m, w_cols, cols):
    """Each row's equity against the weighted subset `cols` of columns (ignoring clashing pairs)."""
    w = np.where(np.isnan(m), 0.0, 1.0) * (w_cols * cols)[None, :]
    num = np.nansum(np.nan_to_num(m) * w, axis=1)
    den = w.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.5)


SOLVE_ITERATIONS = 40


def _solve_jam_range(m, w_o, w_h, pot, facing, add, cost):
    """Per perceived-hero combo, the average probability of jamming in the jam-or-(check/call) game against an
    opponent who calls a jam exactly when their equity against our jamming range beats the price. Hero combo i's
    equity against opponent combo j is 1 - m[j, i] (clashing pairs ignored)."""
    valid = ~np.isnan(m)
    e = np.where(valid, 1 - np.nan_to_num(m), 0.0)               # [opp, hero] hero equity
    wv = valid * w_o[:, None]
    den = np.maximum(wv.sum(axis=0), 1e-12)
    eq_all = (e * wv).sum(axis=0) / den                          # each hero combo vs their whole range
    passive = eq_all * (pot + facing) - facing if facing > 0 else eq_all * pot
    if facing > 0:
        passive = np.maximum(passive, 0.0)                       # facing a bet the passive choice is call or fold
    price = cost / (pot + add + cost)
    jam = np.ones(len(w_h))
    avg = np.zeros(len(w_h))
    for t in range(1, SOLVE_ITERATIONS + 1):
        calls = (_weighted_vs(m, w_h, jam) >= price - CALL_SLACK).astype(float)   # their response to our jam range
        win = (e * wv * calls[:, None]).sum(axis=0)
        called = (wv * calls[:, None]).sum(axis=0) / den
        ev_jam = (1 - called) * pot + win / den * (pot + add + cost) - called * add
        best = (ev_jam > passive).astype(float)                          # our best response
        avg += (best - avg) / t
        jam = np.maximum(avg, 1e-3)
    return jam


def decide(hole, board, pot, facing, hero_left, opp_left, opp, hero_range, can_raise, seed=0):
    """EV of each low-SPR option (chips, relative to folding now) and soft-max probabilities.
    pot: all chips in the middle including bets on this street; facing: what we must add to call."""
    eq_me, m, w_o, w_h = equities(hole, board, opp, hero_range, seed)
    W = w_o.sum()
    if W <= 0:
        eq_me, w_o, W = np.array([0.5]), np.array([1.0]), 1.0
    add = min(hero_left, facing + opp_left)
    cost = add - facing
    # The range we jam with is solved, not assumed: which of the hands they put us on prefer jamming, given how
    # they respond to that range (they call by price against it). Fictitious play to a fixed point.
    jam_range = _solve_jam_range(m, w_o, w_h, pot, facing, add, cost) if m.size and cost > 1e-9 \
        else np.ones(len(w_h))
    their_eq = _weighted_vs(m, w_h, jam_range) if m.size else np.full(len(w_o), 0.5)
    mean = lambda x, mask=None: float((w_o * x * (1 if mask is None else mask)).sum() / W)
    ev = {}
    if facing > 0:
        ev["fold"] = 0.0
        ev["call"] = mean(eq_me) * (pot + facing) - facing
    else:
        ev["check"] = mean(eq_me) * pot
    # Jam: our effective add, their cost to call, their response by price against our solved jam range.
    if can_raise and cost > 1e-9:
        price = cost / (pot + add + cost)
        calls = (their_eq >= price - CALL_SLACK).astype(float)
        ev["all-in"] = mean(np.ones_like(eq_me), 1 - calls) * pot + \
            mean(eq_me * (pot + add + cost) - add, calls)
    # Half-pot bet, only first to act and with enough behind for it to differ from a jam.
    x = BET_FRACTION * pot
    if facing <= 0 and hero_left > x + 1e-9 and hero_left >= BET_MIN_SPR * pot and opp_left > 0:
        price_x = x / (pot + 2 * x)
        back = ((their_eq >= RAISE_BACK_EQ) & (opp_left > x)).astype(float)
        call = ((their_eq >= price_x - CALL_SLACK) & (back == 0)).astype(float)
        fold = 1 - back - call
        e = min(hero_left, opp_left)                         # all-in after their jam back
        # Facing their jam we call when our equity against the jamming hands beats the price, else fold.
        jam_w = (w_o * back).sum()
        eq_vs_jam = float((w_o * back * eq_me).sum() / jam_w) if jam_w > 0 else 0.0
        vs_jam = max(eq_vs_jam * (pot + 2 * e) - e, -x)
        ev[f"bet {BET_FRACTION:g}"] = mean(np.ones_like(eq_me), fold) * pot + \
            mean(eq_me * (pot + 2 * x) - x, call) + (jam_w / W) * vs_jam
    keys = list(ev)
    v = np.array([ev[k] for k in keys])
    t = max(TEMPERATURE * pot, 1e-6)
    p = np.exp((v - v.max()) / t)
    probs = dict(zip(keys, p / p.sum()))
    return probs, {"ev": {k: round(float(val), 3) for k, val in ev.items()}, "equity": round(mean(eq_me), 4),
                   "opp_combos": int(len(w_o)), "hero_range_combos": int(len(w_h))}
