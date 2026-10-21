"""Experiment 39: a pot-odds check for every call, against the opponent's range as their actions shape it.

The whole-hand model calls by hand group and board texture; it never sees the price or how aggressive this
opponent is. Here, facing a bet, the bot estimates the opponent's range and its equity against it:

  preflop   the 169-class weights of the opponent's preflop line (the solved preflop nodes, 100bb), mixed with a
            floor (people deviate, aggressive ones especially)
  postflop  each opponent action narrows it by hand strength on that street's board:
              bet / raise of x pot   the top f(1 - beta) of the range for value plus the weakest f * beta as bluffs
                                     (draws first), f = how often they bet there (their profile), beta = the bluff
                                     share x / (1 + 2x) that makes a call break even, times their aggression
                                     relative to the bot's baseline (and by bet size, once they have shown hands)
              call                   drops most of the bottom (they'd fold it) and some of the top (they'd raise)
              check                  keeps everything, a few strong hands less (slowplays)
  equity    exact over the river cards on the turn, Monte Carlo on the flop, exact on the river; on the flop and
            turn scaled by a realization factor (out of position realizes less)

The price is call / (pot after calling). Clearly better (+4 points): the bot calls (the fold moves to call). On the
river clearly worse (-10 points): the call moves to fold. Raises are left alone.
"""
import numpy as np
import eval7

from preflop.cards import HANDS, INDEX
from preflop import fullgame as fg

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [eval7.Card(r + s) for r in RANKS for s in SUITS]
FLOOR = 0.12                 # share of the preflop weight every class keeps whatever the line says
DEFAULT_BET = {1: 0.40, 2: 0.33, 3: 0.33}       # how often a player bets a street, before data
REALIZE = {"ip": 0.95, "oop": 0.85}             # equity realized before the river
CALL_MARGIN, FOLD_MARGIN = 0.04, 0.10
FLOP_SAMPLES = 1500


def _class(c1, c2):
    r1, r2 = c1.rank, c2.rank
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if c1.suit == c2.suit else "o")


def preflop_weights(pre, acts, opp):
    """169 weights for the opponent's preflop line from the solved nodes (missing nodes leave weights alone)."""
    n = pre["nodes"]
    w = np.ones(len(HANDS))
    seen = []
    for a in acts:
        if a.player == opp and a.kind != "fold":
            node, act = _node(seen, a, opp)
            pol = n.get(node, {}).get(act) if node else None
            if pol is not None:
                w = w * np.asarray(pol, float)
        seen.append((a.player, a.kind, a.to))
    if w.max() <= 0:
        w = np.ones(len(HANDS))
    w = w / w.max()
    return (1 - FLOOR) * w + FLOOR * 0.25          # a floor: a quarter-weight for everything


def _node(seen, a, opp):
    kinds = [(p, k) for p, k, _ in seen if k != "post"]
    jam = a.to is not None and a.to >= 60
    if opp == "btn" and not kinds:
        return "btn_root", {"call": "limp", "raise": "all-in" if jam else "raise"}.get(a.kind)
    if opp == "bb" and kinds == [("btn", "call")]:
        return "bb_vs_limp", {"check": "check", "raise": "all-in" if jam else "raise"}.get(a.kind)
    if opp == "bb" and kinds == [("btn", "raise")]:
        return "bb_vs_raise", {"call": "call", "raise": "all-in" if jam else "3-bet"}.get(a.kind)
    if opp == "btn" and kinds == [("btn", "call"), ("bb", "raise")]:
        return "btn_vs_iso", {"call": "call", "raise": "all-in" if jam else "re-raise"}.get(a.kind)
    if opp == "btn" and kinds == [("btn", "raise"), ("bb", "raise")]:
        return "btn_vs_3bet_small", {"call": "call", "raise": "all-in" if jam else "4-bet"}.get(a.kind)
    if opp == "bb" and kinds == [("btn", "raise"), ("bb", "raise"), ("btn", "raise")]:
        return "bb_vs_4bet_small", {"call": "call", "raise": "all-in"}.get(a.kind)
    if opp == "btn" and len(kinds) >= 3 and a.kind == "call":
        return "btn_vs_5bet", "call"
    return None, None


def combos(weights, dead):
    out = []
    for i, c1 in enumerate(DECK):
        if c1 in dead:
            continue
        for c2 in DECK[i + 1:]:
            if c2 in dead:
                continue
            w = weights[INDEX[_class(c1, c2)]]
            if w > 1e-4:
                out.append([c1, c2, float(w)])
    return out


def _strength(rng, board):
    """Percentile (0 weakest .. 1 strongest) of each combo's made hand on this board, by range weight;
    a flush draw or open-ended straight draw before the river counts as at least the 55th percentile."""
    vals = np.array([eval7.evaluate([c1, c2] + board) for c1, c2, _ in rng], float)
    w = np.array([x[2] for x in rng])
    order = np.argsort(vals, kind="stable")
    cum = np.empty(len(vals))
    cum[order] = (np.cumsum(w[order]) - w[order] / 2) / max(w.sum(), 1e-12)
    if len(board) < 5:
        for k, (c1, c2, _) in enumerate(rng):
            if fg._has_draw((c1, c2), board):
                cum[k] = max(cum[k], 0.55)
    return cum


def narrow(rng, board, kind, frac, f_bet, aggression):
    """Reweight the range for one opponent action on this board."""
    if kind not in ("bet", "raise", "call", "check"):
        return rng
    pct = _strength(rng, board)
    w = np.array([x[2] for x in rng])
    tot = w.sum()
    if kind in ("bet", "raise"):
        beta = float(np.clip(frac / (1 + 2 * frac) * aggression * (1.3 if len(board) < 5 else 1.0), 0.05, 0.6))
        f = float(np.clip(f_bet, 0.05, 0.95))
        v_cut = 1 - f * (1 - beta)                 # value: everything above this percentile
        b_cut = f * beta                           # bluffs: the weakest b_cut (by weight), draws count as weak here
        draw = np.array([len(board) < 5 and fg._has_draw((c1, c2), board) for c1, c2, _ in rng])
        bluff_pct = np.where(draw, 0.0, pct)       # draws are the first bluffs
        keep = np.clip((pct - v_cut) / 0.08 + 0.5, 0, 1) + np.clip((b_cut - bluff_pct) / 0.08 + 0.5, 0, 1)
        new = w * np.clip(keep, 0.03, 1)
    elif kind == "call":
        new = w * np.where(pct < 0.3, 0.25, np.where(pct > 0.93, 0.6, 1.0))
    else:
        new = w * np.where(pct > 0.9, 0.5, 1.0)
    new = new * tot / max(new.sum(), 1e-12)
    return [[c1, c2, float(x)] for (c1, c2, _), x in zip(rng, new) if x > 1e-5]


def equity(hole, board, rng, rng_seed=0):
    """Our equity (ties half) against a weighted range, over the cards still to come."""
    dead = set(hole) | set(board)
    rng = [x for x in rng if x[0] not in dead and x[1] not in dead]
    if not rng:
        return 0.5
    if len(board) == 5:
        mine = eval7.evaluate(list(hole) + board)
        win = tot = 0.0
        for c1, c2, w in rng:
            v = eval7.evaluate([c1, c2] + board)
            win += w * (1.0 if mine > v else 0.5 if mine == v else 0.0)
            tot += w
        return win / tot
    if len(board) == 4:
        rest = [c for c in DECK if c not in dead]
        win = tot = 0.0
        for c1, c2, w in rng:
            for r in rest:
                if r is c1 or r is c2 or r == c1 or r == c2:
                    continue
                b = board + [r]
                m, v = eval7.evaluate(list(hole) + b), eval7.evaluate([c1, c2] + b)
                win += w * (1.0 if m > v else 0.5 if m == v else 0.0)
                tot += w
        return win / tot
    # Flop: sample opponent combos by weight and two board cards.
    g = np.random.default_rng(rng_seed)
    w = np.array([x[2] for x in rng])
    picks = g.choice(len(rng), size=FLOP_SAMPLES, p=w / w.sum())
    rest = [c for c in DECK if c not in dead]
    win = 0.0
    for k in picks:
        c1, c2, _ = rng[k]
        left = [c for c in rest if c != c1 and c != c2]
        i, j = g.choice(len(left), size=2, replace=False)
        b = board + [left[i], left[j]]
        m, v = eval7.evaluate(list(hole) + b), eval7.evaluate([c1, c2] + b)
        win += 1.0 if m > v else 0.5 if m == v else 0.0
    return win / FLOP_SAMPLES


def opponent_range(pre, st, hero, board, hole, prof=None, aggression=1.0, size_value=None):
    """The opponent's weighted combos given everything they did in the hand so far."""
    opp = "bb" if hero == "btn" else "btn"
    dead = set(hole) | set(board)
    rng = combos(preflop_weights(pre, st.history[0], opp), dead)
    nodes = (prof or {}).get("nodes", prof or {})
    for s in range(1, st.street + 1):
        b = board[:2 + s]
        pot = st.street_start[s][0]
        for a in st.history[s]:
            if a.player != opp:
                continue
            frac = (a.to / pot) if a.kind in ("bet", "raise") and pot > 0 else 0.0
            name = ["", "flop", "turn", "river"][s]
            f_bet = _bet_rate(nodes, name, DEFAULT_BET[s])
            agg = aggression
            if size_value is not None and a.kind in ("bet", "raise"):
                agg *= size_value(frac)
            rng = narrow(rng, b, a.kind, frac, f_bet, agg)
    return rng


def _bet_rate(nodes, street, default, shrink=10):
    keys = [f"{street}_bet_when_checked_to", f"{street}_lead", f"{street}_cbet"]
    n = sum(nodes.get(k, {}).get("n", 0) for k in keys)
    hits = sum(nodes.get(k, {}).get("n", 0) * nodes.get(k, {}).get("bet", 0) for k in keys)
    return (hits + shrink * default) / (n + shrink)


def check(probs, hole, board, st, hero, pre, prof, aggression, notes, size_value=None):
    """Adjust a facing-a-bet decision by pot odds; returns probs."""
    call_amt = st.facing(hero)
    if call_amt <= 1e-9 or "call" not in probs or "fold" not in probs:
        return probs
    pot_after = st.pot() + call_amt
    need = call_amt / pot_after
    rng = opponent_range(pre, st, hero, board, hole, prof, aggression, size_value)
    eq = equity(list(hole), list(board), rng)
    real = 1.0 if len(board) == 5 else REALIZE["oop" if hero == "bb" else "ip"]
    eff = eq * real
    out = dict(probs)
    if eff >= need + CALL_MARGIN and out["fold"] > 0:
        out["call"] += out["fold"]
        out["fold"] = 0.0
        notes.append(f"pot odds: equity {eq:.0%} (x{real:.2f}) vs price {need:.0%}: call")
    elif len(board) == 5 and eff <= need - FOLD_MARGIN and out["call"] > 0:
        out["fold"] += out["call"]
        out["call"] = 0.0
        notes.append(f"pot odds: equity {eq:.0%} vs price {need:.0%}: fold")
    else:
        notes.append(f"pot odds: equity {eq:.0%} (x{real:.2f}) vs price {need:.0%}")
    return out
