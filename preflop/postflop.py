"""Rule-based postflop simulator: measure how much of its equity each hand really wins.

After BTN opens and BB calls, both players follow the same simple rules:

  Hand strength on each street (using only the cards they can see):
    STRONG -- two pair or better that improves on the board, an overpair,
              or top pair with a ten-or-better kicker
    MEDIUM -- any other pair that uses a hole card (incl. pocket pairs),
              or a flush draw / open-ended straight draw (flop and turn only)
    WEAK   -- everything else

  Actions (one bet per street at most, no raises, bet = 2/3 pot):
    first to act, or checked to:  STRONG bets, MEDIUM checks, WEAK bluffs some of the time:
        bluff="balanced" (default): just often enough that bluffs make up
            bet / (pot + 2*bet) of all bets in that spot (28.6% for 2/3 pot) -- the
            mix that leaves a caller indifferent. Learned per spot (BB leading /
            BTN after a check, by street) from an earlier batch of hands.
        bluff=<number>: bluff that fraction of WEAK hands (the original rules used 1/3)
    facing a bet (calling="pot_odds", the default):
        STRONG calls. Otherwise call only if your estimated chance of beating the
        bettor is at least the price, bet / (pot + 2*bet) -- 28.6% vs a 2/3-pot bet.
        The chance comes from a learned table: hand strength (equity vs a random
        hand on this board) -> how often hands that strong actually beat a bettor
        on this street, measured in an earlier batch of simulated hands.
    facing a bet (calling="fixed", the original rules):
        STRONG and MEDIUM call, WEAK folds

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

HS_ITERS = 200
HS_BINS = 20

COMBOS = [[tuple(eval7.Card(c) for c in combo) for combo in combos(h)] for h in HANDS]
DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
ALL_HANDS = [(a, b) for k, a in enumerate(DECK) for b in DECK[k + 1:]]

# What the BB did on each street (from the BB's point of view), stored per hand.
NOT_REACHED, CHECK_CHECK, CHECK_CALL, CHECK_FOLD, BET_CALLED, BET_FOLD, BLUFF_CALLED, BLUFF_FOLD = range(8)
CODE_NAMES = {CHECK_CHECK: "check, BTN checks back", CHECK_CALL: "check, BTN bets, BB calls",
              CHECK_FOLD: "check, BTN bets, BB folds", BET_CALLED: "BB value-bets, BTN calls",
              BET_FOLD: "BB value-bets, BTN folds", BLUFF_CALLED: "BB bluffs, BTN calls",
              BLUFF_FOLD: "BB bluffs, BTN folds"}


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


def hand_strength(hole, board):
    """Equity vs one random hand on this board (Monte Carlo; board cards removed from the range)."""
    dead = set(hole) | set(board)
    villain = [(h, 1.0) for h in ALL_HANDS if h[0] not in dead and h[1] not in dead]
    return eval7.py_hand_vs_range_monte_carlo(list(hole), villain, board, HS_ITERS)


def win_chance(table, street, hs):
    """Learned chance of beating a bettor; falls back to raw hand strength where there's no data."""
    if table is None:
        return hs
    v = table[street, min(int(hs * HS_BINS), HS_BINS - 1)]
    return hs if np.isnan(v) else v


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


def play(btn, bb, board, s, stack, rng, passive=False, bluff=BLUFF, bet_fraction=BET_FRACTION,
         calling="pot_odds", table=None, learn=None, bluff_probs=None, spots=None):
    """Play one hand from the flop.

    Returns (BTN realized share of the flop pot, list of 3 BB action codes).
    If `learn` is a list, every pot-odds decision appends (street, hand strength,
    caller's showdown result vs the bettor) to it, for building the win-chance table.
    If `spots` is a list, every betting decision appends (role, street, strength) so
    balanced bluff rates can be learned (role 0 = BB first to act, 1 = BTN after a check).
    bluff_probs: 2 x 3 array of bluff rates by role and street (used instead of `bluff`).
    """
    state = {"pot": 2 * s, "left": {"btn": stack - s, "bb": stack - s}, "put_in": {"btn": 0.0, "bb": 0.0}}
    return play_streets({"btn": btn, "bb": bb}, board, s, state, (0, 1, 2), rng, passive=passive,
                        bluff=bluff, bet_fraction=bet_fraction, calling=calling, table=table,
                        learn=learn, bluff_probs=bluff_probs, spots=spots)


def play_streets(holes, board, s, state, streets, rng, passive=False, bluff=BLUFF,
                 bet_fraction=BET_FRACTION, calling="pot_odds", table=None, learn=None,
                 bluff_probs=None, spots=None):
    """Play the given streets (0 flop, 1 turn, 2 river) with the rules, then showdown.

    state: {"pot", "left": {player: chips}, "put_in": {player: postflop chips}} at the
    start of the first street; it is not modified. Returns (BTN realized share, codes).
    """
    pot = state["pot"]
    left = dict(state["left"])
    put_in = dict(state["put_in"])
    codes = [NOT_REACHED] * 3
    for street in streets:
        seen = board[:3 + street]
        st = {p: strength(holes[p], seen, street == 2) for p in holes}
        if passive:
            codes[street] = CHECK_CHECK
            continue
        bettor, bluffing = None, False
        for role, p in enumerate(("bb", "btn")):   # BB acts first; BTN acts after a check
            if spots is not None:
                spots.append((role, street, st[p]))
            rate = bluff_probs[role, street] if bluff_probs is not None else bluff
            if st[p] == STRONG or (st[p] == WEAK and rng.random() < rate):
                bettor, bluffing = p, st[p] == WEAK
                break
        bet = min(bet_fraction * pot, left[bettor], left["btn" if bettor == "bb" else "bb"]) if bettor else 0
        if bettor is None or bet <= 0:
            codes[street] = CHECK_CHECK
            continue
        caller = "btn" if bettor == "bb" else "bb"
        if calling == "fixed" or st[caller] == STRONG:
            calls = st[caller] != WEAK
        else:
            hs = hand_strength(holes[caller], seen)
            calls = win_chance(table, street, hs) >= bet / (pot + 2 * bet)
            if learn is not None:
                v_c = eval7.evaluate(list(holes[caller]) + board)
                v_b = eval7.evaluate(list(holes[bettor]) + board)
                learn.append((street, hs, 1.0 if v_c > v_b else 0.5 if v_c == v_b else 0.0))
        if bettor == "bb":
            codes[street] = (BLUFF_CALLED if calls else BLUFF_FOLD) if bluffing else \
                            (BET_CALLED if calls else BET_FOLD)
        else:
            codes[street] = CHECK_CALL if calls else CHECK_FOLD
        if not calls:                     # fold: bettor takes the pot
            won = pot if bettor == "btn" else 0.0
            return (won - put_in["btn"]) / (2 * s), codes
        for p in (bettor, caller):
            left[p] -= bet
            put_in[p] += bet
        pot += 2 * bet
    v_btn = eval7.evaluate(list(holes["btn"]) + board)
    v_bb = eval7.evaluate(list(holes["bb"]) + board)
    won = pot if v_btn > v_bb else pot / 2 if v_btn == v_bb else 0.0
    return (won - put_in["btn"]) / (2 * s), codes


def _worker(job):
    btn_w, bb_w, n, s, stack, seed, opts = job
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    out_i, out_j, share, codes, learn, spots = [], [], [], [], [], []
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
        sh, cd = play(btn, bb, board, s, stack, rng, learn=learn, spots=spots, **opts)
        share.append(sh)
        codes.append(cd)
    return (np.array(out_i, dtype=int), np.array(out_j, dtype=int), np.array(share),
            np.array(codes, dtype=np.int8).reshape(-1, 3),
            np.array(learn).reshape(-1, 3), np.array(spots, dtype=np.int8).reshape(-1, 3))


def learn_table(records):
    """Average showdown result vs the bettor, by street and hand-strength bin (NaN = no data)."""
    table = np.full((3, HS_BINS), np.nan)
    if len(records) == 0:
        return table
    street = records[:, 0].astype(int)
    b = np.minimum((records[:, 1] * HS_BINS).astype(int), HS_BINS - 1)
    for st in range(3):
        for k in range(HS_BINS):
            m = (street == st) & (b == k)
            if m.sum() >= 50:
                table[st, k] = records[m, 2].mean()
    return table


def learn_bluff_rates(spots, bet_fraction):
    """Bluff rate per (role, street) so bluffs are bet / (pot + 2*bet) of all bets there."""
    target = bet_fraction / (1 + 2 * bet_fraction)
    rates = np.zeros((2, 3))
    for role in range(2):
        for st in range(3):
            m = (spots[:, 0] == role) & (spots[:, 1] == st)
            strong, weak = (spots[m, 2] == STRONG).sum(), (spots[m, 2] == WEAK).sum()
            if weak:
                rates[role, st] = min(1.0, target / (1 - target) * strong / weak)
    return rates


def _run(btn_range, bb_range, n, s, stack, seed, opts):
    from .cards import n_combos
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, s, stack, seed * 1000 + k, opts)
            for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, jobs))
    keys = ("btn_class", "bb_class", "share", "codes", "learn", "spots")
    return {k: np.concatenate([p[m] for p in parts]) for m, k in enumerate(keys)}


def simulate(btn_range, bb_range, n=1_000_000, s=2.25, stack=100, seed=0, passive=False,
             bluff="balanced", bet_fraction=BET_FRACTION, calling="pot_odds", learn_hands=300_000):
    """Deal n hands from the BTN's opening range vs the BB's calling range and play them out.

    btn_range / bb_range: 169 action frequencies (e.g. BTN raise %, BB call %).
    Learning: two batches of `learn_hands` hands are played first. Each one
    re-learns the win-chance table (pot-odds calling) and the balanced bluff
    rates from the batch before, so the main batch plays with both settled.

    Returns a dict of arrays: btn_class, bb_class, share (BTN realized share),
    codes (n x 3 BB action codes for flop / turn / river), plus "table"
    (win chance vs a bettor) and "bluff_rates" (role x street).
    """
    balanced = bluff == "balanced"
    opts = {"passive": passive, "bluff": BLUFF if balanced else bluff, "bet_fraction": bet_fraction,
            "calling": calling}
    table, rates = None, None
    if not passive and (calling == "pot_odds" or balanced):
        for k in range(2):
            batch = _run(btn_range, bb_range, learn_hands, s, stack, seed + 10_000 * (k + 1),
                         {**opts, "table": table, "bluff_probs": rates})
            if calling == "pot_odds":
                table = learn_table(batch["learn"])
            if balanced:
                rates = learn_bluff_rates(batch["spots"], bet_fraction)
    out = _run(btn_range, bb_range, n, s, stack, seed, {**opts, "table": table, "bluff_probs": rates})
    out["table"], out["bluff_rates"] = table, rates
    return out
