"""Experiment 50: held-out opponents for ablation matches (scripts/ablation.py).

These bots are frozen and must never be used to tune a parameter: they exist to check, after tuning, whether each
adjustment on top of the base policy still helps against styles it was not fitted to. Tuning against them would
turn them into training opponents and the check would mean nothing. Add a new held-out bot instead of editing one.

    geo        "solver-like" script: geometric sizing, value:bluff by the bet's price, MDF defence by equity rank
    trapper    slow-plays two pair+ until the river, then check-raises / jams; stabs weak hands when checked to
    limper     limps and calls wide preflop, calls any pair or draw, bets only two pair+ on the river
    threebet   hyper-aggressive preflop (3-bets 30%, 4-bet jams 12%), small c-bets, barrels by equity
    adaptive   learns the bot's fold rates and river bluff share during the match and best-responds to them
    base       the bot's own base policy with every adjustment switched off (a pure-policy mirror)
"""
import eval7
import numpy as np

from .engine import STACK, Engine, State
from .rivals import _Base as _RivalBase, has_draw, made, mc_equity, rank_pct

EPS = 1e-6


class _Base(_RivalBase):
    def ctx(self, history, facing):
        st = State(history, stack=getattr(self, "stack", STACK))     # the harness sets the effective stack
        return st, st.pot(), max(st.inp.values())


def _equity_rank(eq):
    """Rough share of random hands an equity vs random beats (eq vs random spreads roughly over 0.2 - 0.9)."""
    return float(np.clip((eq - 0.2) / 0.7, 0.0, 1.0))


class Geo(_Base):
    name = "geo (geometric sizing, price-balanced bluffs, MDF defence)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            n_raises = sum(a.kind == "raise" for a in st.history[0])
            if n_raises >= 2:
                return ("raise", st.stack) if r > 0.95 else ("call", 0) if r > 0.88 else ("fold", 0)
            if n_raises == 1 and seat == "bb":
                return ("raise", top * 3.5) if r > 0.86 or r < 0.06 else ("call", 0) if r > 0.35 else ("fold", 0)
            if seat == "btn" and top <= 1.0:
                return ("raise", 2.5) if r > 0.2 else ("fold", 0)
            return ("call", 0) if facing > EPS and r > 0.55 else self.passive(facing) if facing <= EPS else ("fold", 0)
        eq = mc_equity(hole, board, self.rng, 100)
        rank = _equity_rank(eq)
        streets_left = 4 - st.street
        behind = min(st.left("btn"), st.left("bb"))
        # Geometric fraction: the same pot share each street puts the stacks in by the river.
        geo = ((1 + 2 * behind / max(pot, EPS)) ** (1 / streets_left) - 1) / 2
        frac = float(np.clip(geo, 0.33, 1.5))
        if facing > EPS:
            mdf = pot / (pot + facing)                              # pot before their bet over pot after it
            keep = 1 - mdf
            if rank > 0.9:
                return self.bet(st, frac, facing)
            return ("call", 0) if rank >= keep else ("fold", 0)
        bluff_share = frac / (1 + 2 * frac)
        if rank > 0.75 or (rank < 0.15 and self.rng.random() < bluff_share / 0.15 * 0.25 / (1 - bluff_share)):
            return self.bet(st, frac, facing)
        return ("check", 0)


class Trapper(_Base):
    name = "trapper (slow-plays two pair+, river check-raises)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 8:
                return ("call", 0) if r > 0.85 else ("fold", 0)
            if seat == "btn" and top <= 1.0:
                return ("call", 0) if r > 0.75 else ("raise", 2.5) if r > 0.3 else ("fold", 0)   # limps its best
            if facing > EPS:
                return ("call", 0) if r > 0.4 else ("fold", 0)
            return ("check", 0)
        strong = made(hole, board) >= 2
        if strong:
            if st.street < 3:
                return self.passive(facing)
            return ("raise", st.inp[seat] + st.left(seat)) if facing > EPS else ("check", 0) \
                if seat == "bb" else self.bet(st, 1.0, facing)
        tp = made(hole, board) >= 1
        if facing > EPS:
            return ("call", 0) if (tp and facing <= pot) or (has_draw(hole, board) and facing <= 0.5 * pot) \
                else ("fold", 0)
        if st.history[st.street] and self.rng.random() < 0.45:
            return self.bet(st, 0.5, facing)
        return ("check", 0)


class Limper(_Base):
    name = "limper (wide passive, bets two pair+ on the river)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 6:
                return ("call", 0) if r > 0.8 else ("fold", 0)
            if facing > EPS:
                return ("call", 0) if r > 0.15 else ("fold", 0)
            return ("check", 0)
        m = made(hole, board)
        if facing > EPS:
            return ("call", 0) if m >= 1 or (has_draw(hole, board) and st.street < 3) else ("fold", 0)
        if m >= 2 and st.street == 3:
            return self.bet(st, 0.75, facing)
        return ("check", 0)


class ThreeBet(_Base):
    name = "threebet (hyper-aggressive preflop, small c-bets)"

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        if st.street == 0:
            r = rank_pct(hole)
            n_raises = sum(a.kind == "raise" for a in st.history[0])
            if n_raises >= 2:
                if r > 0.88 or (r < 0.1 and self.rng.random() < 0.5):
                    return ("raise", st.stack)                     # 4-bet jam ~12%
                return ("call", 0) if r > 0.7 and facing < 0.3 * st.stack else ("fold", 0)
            if n_raises == 1 and seat == "bb":
                return ("raise", top * 3.2) if r > 0.7 else ("call", 0) if r > 0.3 else ("fold", 0)
            if seat == "btn" and top <= 1.0:
                return ("raise", 2.2) if r > 0.15 else ("fold", 0)
            return self.passive(facing)
        eq = mc_equity(hole, board, self.rng, 80)
        last = max((a.player for a in st.history[st.street - 1] if a.kind in ("bet", "raise")), default=None)
        if facing > EPS:
            return ("call", 0) if eq > facing / (pot + facing) + 0.05 else ("fold", 0)
        need = {1: 0.0, 2: 0.4, 3: 0.6}[st.street]
        if (st.street == 1 and (last == seat or next((a.player for a in reversed(st.history[0])
                                                       if a.kind == "raise"), None) == seat)) or eq > need + 0.1:
            return self.bet(st, 0.33 if st.street == 1 else 0.66, facing)
        return ("check", 0)


class Adaptive(_Base):
    """Starts as an equity player and best-responds to what it has seen of the bot this match: bluffs a street when
    the bot folds to bets there more than the bluff's price, and calls river bets with bluff-catchers when the bot's
    shown river bets were bluffs more often than the call's price."""
    name = "adaptive (online best response to observed fold / bluff rates)"
    BLUFF_SIZE = 0.75
    PRIOR = 10                     # pseudo-observations at the breakeven rates

    def __init__(self, seed=11):
        super().__init__(seed)
        price = self.BLUFF_SIZE / (1 + self.BLUFF_SIZE)
        self.folds = {s: [price * self.PRIOR, self.PRIOR] for s in (1, 2, 3)}     # [folds, bets faced]
        self.river_bets = [0.3 * self.PRIOR, self.PRIOR]                           # [bluffs shown, shown]
        self.my_seat = None

    def act(self, seat, hole, board, history, facing):
        st, pot, top = self.ctx(history, facing)
        self.my_seat = seat
        if st.street == 0:
            r = rank_pct(hole)
            if facing > 8:
                return ("call", 0) if r > 0.85 else ("fold", 0)
            if seat == "btn" and top <= 1.0:
                return ("raise", 2.5) if r > 0.3 else ("fold", 0)
            if facing > EPS:
                return ("raise", top * 3) if r > 0.88 else ("call", 0) if r > 0.4 else ("fold", 0)
            return ("check", 0)
        eq = mc_equity(hole, board, self.rng, 80)
        if facing > EPS:
            price = facing / (pot + facing)
            if st.street == 3 and 0.25 < eq < 0.7:
                bluffs, shown = self.river_bets
                return ("call", 0) if bluffs / shown > price else ("fold", 0)
            return ("call", 0) if eq > price else ("fold", 0)
        if eq > 0.7:
            return self.bet(st, 0.66, facing)
        f, n = self.folds[st.street]
        if eq < 0.4 and f / n > self.BLUFF_SIZE / (1 + self.BLUFF_SIZE):
            return self.bet(st, self.BLUFF_SIZE, facing)
        return ("check", 0)

    def observe(self, seat, log, holes, board):
        """After each hand: the action log [(street, actor, kind, to)] and, at showdown, the bot's cards."""
        other = "bb" if seat == "btn" else "btn"
        for i, (street, actor, kind, _) in enumerate(log):
            if actor == seat and kind in ("bet", "raise") and street > 0:
                nxt = next((k for s, a, k, _ in log[i + 1:] if a == other), None)
                if nxt is not None:
                    self.folds[street][0] += nxt == "fold"
                    self.folds[street][1] += 1
        if any(k == "fold" for *_, k, _ in log):
            return
        if any(s == 3 and a == other and k in ("bet", "raise") for s, a, k, _ in log):
            shown = list(holes[other])
            # A river bet with less than one pair beyond the board counts as a bluff.
            bluff = (eval7.evaluate(shown + board) >> 24) <= (eval7.evaluate(list(board)) >> 24)
            self.river_bets[0] += bluff
            self.river_bets[1] += 1


class BaseMirror:
    name = "base (the bot's base policy, every adjustment off)"
    STAGES = ("vs_shove", "small_raise", "reraise_war", "monster", "trap", "river_jam", "short_stack", "pot_odds",
              "oop_correction", "small_leads", "polar_river")

    def __init__(self, seed=23):
        self.engine = Engine(exact_required=False, seed=seed, pot_odds=False)
        self.engine.off = set(self.STAGES)

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history, stack=getattr(self, "stack", STACK))
        return d.action, d.to


def make(name, seed=11):
    return {"geo": lambda: Geo(seed), "trapper": lambda: Trapper(seed), "limper": lambda: Limper(seed),
            "threebet": lambda: ThreeBet(seed), "adaptive": lambda: Adaptive(seed),
            "base": lambda: BaseMirror(seed + 12)}[name]()


HELDOUT = ("geo", "trapper", "limper", "threebet", "adaptive", "base")
