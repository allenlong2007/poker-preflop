"""Experiment 46: rival bots -- the kinds of heads-up bots people write with Codex or Claude, to train against.

Each is a self-play agent (act(seat, hole, board, history, facing) -> (kind, to)) for bot/selfplay.py:
    EquityBot     Monte Carlo equity vs a random hand; raises / bets 2/3 pot above thresholds, calls when that
                  equity beats the pot odds. The most common "simple solver" (it ignores ranges, so it calls too much).
    ChartBot      tight-aggressive rules: a preflop chart (open 45%, 3-bet 10%), postflop by made-hand category
                  (two pair+ raises, top pair bets 2/3 and calls one bet, draws semi-bluff, air c-bets the flop once).
    PushFold      shoves its top 35% preflop, folds the rest, calls shoves with the top 20%.
    MinRaiser     a prober: min-bets / min-raises everything decent, min-raises the river with anything, checks back air.
    Overbettor    polar 2x-pot bets: with two pair+ and with its weakest 25%, checks / calls the middle.
    Nit           the cautious "LLM-style" bot: plays its top 25%, bets 1/2 pot only with top pair+, folds to any
                  bet over 1/2 pot without two pair.
    Exploiter     a counter-bot: the engine's strategy, but bets whenever checked to and raises small bets 40% of
                  the time (targets a bot that gives up after checking).
"""
import json
from functools import lru_cache
from pathlib import Path

import eval7
import numpy as np

from preflop.cards import HANDS
from bot.engine import STACK, EPS, State, hand_class

ROOT = Path(__file__).resolve().parents[1]
EQ_CACHE = ROOT / "output" / "match" / "equity_vs_random.json"
_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]


@lru_cache(None)
def preflop_equity():
    """Equity of each of the 169 hands vs a random hand (cached to disk)."""
    if EQ_CACHE.exists():
        return json.loads(EQ_CACHE.read_text())
    from preflop.cards import combos
    rng = np.random.default_rng(0)
    out = {}
    for h in HANDS:
        c1, c2 = [eval7.Card(x) for x in combos(h)[0]]
        rest = [c for c in _DECK if c not in (c1, c2)]
        win = 0.0
        for _ in range(1500):
            idx = rng.choice(len(rest), 7, replace=False)
            o, b = [rest[i] for i in idx[:2]], [rest[i] for i in idx[2:]]
            m, v = eval7.evaluate([c1, c2] + b), eval7.evaluate(o + b)
            win += 1.0 if m > v else 0.5 if m == v else 0.0
        out[h] = win / 1500
    EQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EQ_CACHE.write_text(json.dumps(out))
    return out


def rank_pct(hole):
    """Share of the 169 hands (by combos) this hand beats on equity vs random: 1.0 = AA."""
    eq = preflop_equity()
    order = sorted(eq, key=eq.get)
    from preflop.cards import n_combos
    tot = sum(n_combos(h) for h in order)
    h = hand_class(hole)
    below = sum(n_combos(x) for x in order[:order.index(h)])
    return below / tot


def mc_equity(hole, board, rng, n=120):
    """Equity vs a random hand on this board."""
    dead = set(hole) | set(board)
    rest = [c for c in _DECK if c not in dead]
    need = 5 - len(board)
    win = 0.0
    for _ in range(n):
        idx = rng.choice(len(rest), 2 + need, replace=False)
        o = [rest[i] for i in idx[:2]]
        b = board + [rest[i] for i in idx[2:]]
        m, v = eval7.evaluate(list(hole) + b), eval7.evaluate(o + b)
        win += 1.0 if m > v else 0.5 if m == v else 0.0
    return win / n


def made(hole, board):
    """Hand category beyond the board: 0 nothing, 1 pair, 2 two pair, 3 trips..."""
    return (eval7.evaluate(list(hole) + board) >> 24) - (eval7.evaluate(board) >> 24)


def top_pair_plus(hole, board):
    if made(hole, board) >= 2:
        return True
    top = max(c.rank for c in board)
    return any(c.rank == top for c in hole) or (hole[0].rank == hole[1].rank and hole[0].rank > top)


def has_draw(hole, board):
    cards = list(hole) + board
    suits = [c.suit for c in cards]
    if len(board) < 5 and any(suits.count(s) == 4 for s in set(suits)) and any(suits.count(c.suit) == 4 for c in hole):
        return True
    r = sorted({c.rank for c in cards} | ({-1} if any(c.rank == 12 for c in cards) else set()))
    return len(board) < 5 and any(sum(a <= x <= a + 4 for x in r) >= 4 for a in r)


class _Base:
    name = "rival"

    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)

    def ctx(self, history, facing):
        st = State(history)
        pot = st.pot()
        top = max(st.inp.values())
        return st, pot, top

    def bet(self, st, frac, facing):
        """Bet / raise to frac of the pot (after calling), capped by the stack."""
        top = max(st.inp.values())
        to = top + (st.pot() + facing) * frac
        return ("raise", to)

    def passive(self, facing):
        return ("call", 0) if facing > EPS else ("check", 0)


class EquityBot(_Base):
    name = "equity bot (MC equity vs random, pot-odds calls)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            eq = preflop_equity()[hand_class(hole)]
            if facing > 5:                                            # facing a big raise / shove
                return ("call", 0) if eq > 0.6 else ("fold", 0)
            if eq > 0.62 and top < 10:
                return ("raise", top * 3)
            if eq > 0.47 or (facing > EPS and eq > facing / (pot + facing) + 0.05):
                return ("call", 0) if facing > EPS else ("check", 0)
            return ("fold", 0) if facing > EPS else ("check", 0)
        eq = mc_equity(hole, board, self.rng)
        if eq > 0.85:
            return self.bet(st, 1.0, facing)
        if facing > EPS:
            return ("call", 0) if eq > facing / (pot + facing) else ("fold", 0)
        if eq > 0.62 or self.rng.random() < 0.1:
            return self.bet(st, 0.66, facing)
        return ("check", 0)


class ChartBot(_Base):
    name = "chart bot (TAG rules)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 10:
                return ("call", 0) if r > 0.92 else ("fold", 0)
            if r > 0.9 and top < 8:
                return ("raise", top * 3)
            if seat == "btn" and top <= 1.0:
                return ("raise", 2.5) if r > 0.55 else ("fold", 0)
            if facing > EPS:
                return ("call", 0) if r > 0.6 else ("fold", 0)
            return ("check", 0)
        m = made(hole, board)
        tp = top_pair_plus(hole, board)
        raiser = next((a.player for a in reversed(st.history[0]) if a.kind == "raise"), None)
        if m >= 2:
            return self.bet(st, 1.0 if facing > EPS else 0.75, facing)
        if facing > EPS:
            bets_this_street = sum(a.kind in ("bet", "raise") for a in st.history[st.street])
            if tp and bets_this_street <= 1:
                return ("call", 0)
            if has_draw(hole, board) and facing <= 0.75 * pot:
                return ("call", 0)
            return ("fold", 0)
        if tp:
            return self.bet(st, 0.66, facing)
        if has_draw(hole, board) and self.rng.random() < 0.5:
            return self.bet(st, 0.66, facing)
        if st.street == 1 and raiser == seat and self.rng.random() < 0.6:
            return self.bet(st, 0.5, facing)
        return ("check", 0)


class PushFold(_Base):
    name = "push / fold bot"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 5:
                return ("call", 0) if r > 0.8 else ("fold", 0)
            if r > 0.65:
                return ("raise", STACK)
            return ("fold", 0) if facing > EPS else ("check", 0)
        return self.passive(facing)


class MinRaiser(_Base):
    name = "min-raiser (probes with minimum sizes)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 6:
                return ("call", 0) if r > 0.75 else ("fold", 0)
            if r > 0.35:
                return ("raise", top + max(facing, 1.0))
            return ("fold", 0) if facing > EPS else ("check", 0)
        eq = mc_equity(hole, board, self.rng, 80)
        if st.street == 3 and facing > EPS and facing < 0.5 * pot:
            return ("raise", top + facing)                          # min-raise the river with anything
        if eq > 0.55:
            return ("raise", top + max(facing, 1.0))
        if facing > EPS:
            return ("call", 0) if eq > facing / (pot + facing) else ("fold", 0)
        return ("check", 0)


class Overbettor(_Base):
    name = "overbettor (polar 2x pot)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 10:
                return ("call", 0) if r > 0.85 else ("fold", 0)
            if seat == "btn" and top <= 1.0:
                return ("raise", 3.0) if r > 0.4 else ("fold", 0)
            if facing > EPS:
                return ("call", 0) if r > 0.45 else ("fold", 0)
            return ("check", 0)
        eq = mc_equity(hole, board, self.rng, 80)
        strong = made(hole, board) >= 2 or eq > 0.8
        if strong or (eq < 0.3 and self.rng.random() < 0.5):
            return self.bet(st, 2.0, facing) if facing <= 0.5 * pot or strong else ("fold", 0)
        if facing > EPS:
            return ("call", 0) if eq > 0.5 and facing <= pot else ("fold", 0)
        return ("check", 0)


class Nit(_Base):
    name = "nit (cautious LLM-style bot)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 6:
                return ("call", 0) if r > 0.9 else ("fold", 0)
            if r > 0.75:
                return ("raise", top * 3 if top > 1 else 2.5)
            if facing > EPS:
                return ("fold", 0) if facing > 0.5 else ("call", 0) if r > 0.5 else ("fold", 0)
            return ("check", 0)
        tp = top_pair_plus(hole, board)
        if facing > EPS:
            if made(hole, board) >= 2:
                return ("call", 0)
            return ("call", 0) if tp and facing <= 0.5 * pot else ("fold", 0)
        return self.bet(st, 0.5, facing) if tp else ("check", 0)


class Exploiter(_Base):
    name = "exploiter (stabs when checked to, raises small bets)"

    def __init__(self, engine, seed=11):
        super().__init__(seed)
        self.engine = engine

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        d = self.engine.decide(seat, hole, board, history)
        if st.street == 0:
            return d.action, d.to
        if facing <= EPS and st.history[st.street] and d.action == "check":
            return self.bet(st, 0.5, facing)
        if facing > EPS and facing <= 0.4 * pot and self.rng.random() < 0.4:
            return self.bet(st, 0.7, facing)
        return d.action, d.to


def make(name, seed=11):
    from bot.engine import Engine
    return {"equity": lambda: EquityBot(seed), "chart": lambda: ChartBot(seed), "pushfold": lambda: PushFold(seed),
            "minraise": lambda: MinRaiser(seed), "overbet": lambda: Overbettor(seed), "nit": lambda: Nit(seed),
            "exploiter": lambda: Exploiter(Engine(exact_required=False, seed=2), seed)}[name]()


RIVALS = ("equity", "chart", "pushfold", "minraise", "overbet", "nit", "exploiter")
