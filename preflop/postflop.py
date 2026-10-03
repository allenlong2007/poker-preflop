"""Rule-based postflop simulator: measure how much of its equity each hand really wins.

After BTN opens and BB calls, both players follow the same simple rules:

  Hand strength on each street (using only the cards they can see):
    STRONG -- two pair or better that improves on the board, an overpair,
              or top pair with a ten-or-better kicker
    MEDIUM -- any other pair that uses a hole card (incl. pocket pairs),
              or a flush draw / open-ended straight draw (flop and turn only)
    WEAK   -- everything else

  Actions (one bet per street at most, no raises, bet = 2/3 pot):
    first to act, or checked to:  STRONG bets, WEAK bluffs BLUFF of the time,
                                  MEDIUM checks
    facing a bet:                 STRONG and MEDIUM call, WEAK folds

The BB acts first on every street (out of position). Nothing else differs
between the players, so any edge the BTN gets comes from acting last.

"Realized share" = the fraction of the flop pot a player ends up with on
average, counting chips won or lost in postflop betting (it can go above 1 or
below 0). With no betting at all it equals raw equity.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import eval7
import numpy as np

from .cards import HANDS, combos

BLUFF = 1 / 3
BET_FRACTION = 2 / 3
STRONG, MEDIUM, WEAK = 2, 1, 0
TEN = 8                                   # eval7 rank index: 2 = 0 ... A = 12

COMBOS = [[tuple(eval7.Card(c) for c in combo) for combo in combos(h)] for h in HANDS]
DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]


def _has_draw(hole, board):
    cards = list(hole) + board
    for suit in range(4):
        if sum(c.suit == suit for c in cards) == 4 and any(c.suit == suit for c in hole):
            return True
    ranks = {c.rank for c in cards}
    if 12 in ranks:
        ranks.add(-1)                     # ace plays low too
    hole_ranks = {c.rank for c in hole} | ({-1} if any(c.rank == 12 for c in hole) else set())
    for lo in range(-1, 9):               # four in a row with room on both ends
        run = set(range(lo, lo + 4))
        if run <= ranks and run & hole_ranks and lo - 1 >= -1 and lo + 4 <= 12:
            return True
    return False


def strength(hole, board, river):
    typ = eval7.evaluate(list(hole) + board) >> 24       # 0 high card, 1 pair, 2 two pair, ...
    board_typ = eval7.evaluate(board) >> 24
    board_ranks = [c.rank for c in board]
    top = max(board_ranks)
    h1, h2 = hole[0].rank, hole[1].rank
    if typ >= 3 and typ > board_typ:
        return STRONG
    if typ == 2 and board_typ == 0:
        return STRONG
    if h1 == h2 and h1 > top:
        return STRONG                     # overpair
    for a, b in ((h1, h2), (h2, h1)):
        if a == top and b >= TEN:
            return STRONG                 # top pair, good kicker
    if h1 == h2 or h1 in board_ranks or h2 in board_ranks:
        return MEDIUM
    if not river and _has_draw(hole, board):
        return MEDIUM
    return WEAK


def play(btn, bb, board, s, stack, rng, passive=False, bluff=BLUFF, bet_fraction=BET_FRACTION):
    """Play one hand from the flop. Returns the BTN's realized share of the flop pot."""
    pot = 2 * s
    left = {"btn": stack - s, "bb": stack - s}
    put_in = {"btn": 0.0, "bb": 0.0}
    holes = {"btn": btn, "bb": bb}
    for n in (3, 4, 5):
        seen = board[:n]
        st = {p: strength(holes[p], seen, n == 5) for p in holes}
        if passive:
            continue
        bettor = None
        for p in ("bb", "btn"):           # BB acts first; BTN acts after a check
            if st[p] == STRONG or (st[p] == WEAK and rng.random() < bluff):
                bettor = p
                break
        if bettor is None:
            continue
        caller = "btn" if bettor == "bb" else "bb"
        bet = min(bet_fraction * pot, left[bettor], left[caller])
        if bet <= 0:
            continue
        if st[caller] == WEAK:            # fold: bettor takes the pot
            won = pot if bettor == "btn" else 0.0
            return (won - put_in["btn"]) / (2 * s)
        for p in (bettor, caller):
            left[p] -= bet
            put_in[p] += bet
        pot += 2 * bet
    v_btn = eval7.evaluate(list(btn) + board)
    v_bb = eval7.evaluate(list(bb) + board)
    won = pot if v_btn > v_bb else pot / 2 if v_btn == v_bb else 0.0
    return (won - put_in["btn"]) / (2 * s)


def _worker(job):
    btn_w, bb_w, n, s, stack, seed, passive, bluff, bet_fraction = job
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    out_i, out_j, share = [], [], []
    for i, j in zip(bi, bj):
        btn = COMBOS[i][rng.integers(len(COMBOS[i]))]
        bb = COMBOS[j][rng.integers(len(COMBOS[j]))]
        if set(btn) & set(bb):
            continue                      # card clash: drop the deal (keeps card removal right)
        dead = set(btn) | set(bb)
        live = [c for c in DECK if c not in dead]
        board = [live[k] for k in rng.choice(len(live), size=5, replace=False)]
        out_i.append(i)
        out_j.append(j)
        share.append(play(btn, bb, board, s, stack, rng, passive, bluff, bet_fraction))
    return np.array(out_i), np.array(out_j), np.array(share)


def simulate(btn_range, bb_range, n=1_000_000, s=2.25, stack=100, seed=0, passive=False,
             bluff=BLUFF, bet_fraction=BET_FRACTION):
    """Deal n hands from the BTN's opening range vs the BB's calling range and play them out.

    btn_range / bb_range: 169 action frequencies (e.g. BTN raise %, BB call %).
    Returns arrays (btn_class, bb_class, btn_realized_share).
    """
    from .cards import n_combos
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, passive,
             bluff, bet_fraction)
            for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, jobs))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
