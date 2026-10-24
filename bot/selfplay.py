"""Heads-up 100bb dealer for testing the match bot offline (experiment 29).

Plays full hands between two agents with standard no-limit rules (blinds 0.5 / 1, min-raise, all-ins run
out to showdown) and the engine's history format. Matches are "duplicate": every deal is played twice
with the seats swapped, so card luck mostly cancels.

Usage:
    python -m bot.selfplay [hands] [opponent]      opponent: station | maniac | model | self | aggro   (default 2000, station)
"""
import sys
import time

import eval7
import numpy as np

from .engine import STACK, Act, Engine, State

EPS = 1e-6
DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs"]


class EngineAgent:
    def __init__(self, engine, name="engine"):
        self.engine, self.name = engine, name
        self.sources = {}

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history)
        self.sources[d.source] = self.sources.get(d.source, 0) + 1
        return d.action, d.to


class Station:
    """Never folds, never bets: checks or calls everything."""
    name = "calling station"

    def act(self, seat, hole, board, history, facing):
        return ("call", 0) if facing > EPS else ("check", 0)


class Maniac:
    """Always bets or raises pot (an all-in once the stack runs out), never folds."""
    name = "maniac"

    def act(self, seat, hole, board, history, facing):
        st = State(history)
        return "raise", max(st.inp.values()) + st.pot() + facing


class Aggro:
    """A loose-aggressive player: the engine's strategy, but 40% of its checks and calls postflop become pot-sized
    bets / raises, and 25% of preflop calls become 3x raises (experiment 39)."""
    name = "aggressive (engine + 40% extra pot bets / raises)"

    def __init__(self, engine, rate=0.4, seed=7):
        self.engine, self.rate, self.rng = engine, rate, np.random.default_rng(seed)

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history)
        st = State(history)
        rate = self.rate if st.street > 0 else 0.25
        if d.action in ("check", "call") and self.rng.random() < rate:
            return "raise", max(st.inp.values()) + st.pot() + facing if st.street > 0 else max(st.inp.values()) * 3
        return d.action, d.to


class ValueHeavy:
    """A value-heavy player (experiment 42): the engine's strategy, but postflop it only bets or raises with two pair
    or better (pot-sized) and never bluffs; with less, its bets become checks and its raises calls."""
    name = "value-heavy (bets / raises only two pair+, pot-sized)"

    def __init__(self, engine):
        self.engine = engine

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history)
        st = State(history)
        if st.street == 0:
            return d.action, d.to
        made = (eval7.evaluate(list(hole) + board) >> 24) - (eval7.evaluate(board) >> 24)
        if made >= 2:
            return "raise", max(st.inp.values()) + st.pot() + facing
        if d.action in ("bet", "raise"):
            return ("call", 0) if facing > EPS else ("check", 0)
        return d.action, d.to


def play_hand(agents, holes, board, stack=STACK, log=None):
    """One hand; returns the BTN's net result in bb. agents / holes: {"btn": ..., "bb": ...}."""
    total = {"btn": 0.0, "bb": 0.0}
    hist = []
    for street in range(4):
        inp = {"btn": 0.5, "bb": 1.0} if street == 0 else {"btn": 0.0, "bb": 0.0}
        acts = []
        hist.append(acts)
        all_in = any(stack - total[p] <= EPS for p in total)
        actor = "btn" if street == 0 else "bb"
        last_inc = 1.0
        n_board = (0, 3, 4, 5)[street]
        while not all_in:
            other = "bb" if actor == "btn" else "btn"
            facing = inp[other] - inp[actor]
            cap = stack - total[actor]                       # the actor's whole stack on this street
            kind, to = agents[actor].act(actor, holes[actor], board[:n_board], hist, facing)
            if kind == "check" and facing > EPS:
                kind = "call"
            if kind == "call" and facing <= EPS:
                kind = "check"
            if kind in ("bet", "raise") and (stack - total[other] - inp[other] <= EPS or cap <= inp[other] + EPS):
                kind = "call"                                # nobody left to raise / can't raise
            if log is not None:
                log.append((street, actor, kind, to))
            if kind == "fold":
                acts.append(Act(actor, "fold"))
                return total["bb"] + inp["bb"] if actor == "bb" else -(total["btn"] + inp["btn"])
            if kind == "check":
                acts.append(Act(actor, "check"))
                if street == 0 or len(acts) >= 2:
                    break
            elif kind == "call":
                inp[actor] = min(inp[other], cap)
                acts.append(Act(actor, "call"))
                if not (street == 0 and len(acts) == 1):    # a limp gives the BB its option
                    break
            else:
                to = min(max(to, inp[other] + last_inc), cap)
                last_inc = max(last_inc, to - inp[other])
                inp[actor] = to
                acts.append(Act(actor, "bet" if inp[other] <= EPS else "raise", to))
            actor = other
        for p in total:
            total[p] += inp[p]
    v_btn = eval7.evaluate(list(holes["btn"]) + board)
    v_bb = eval7.evaluate(list(holes["bb"]) + board)
    pot = total["btn"] + total["bb"]
    share = 1.0 if v_btn > v_bb else 0.5 if v_btn == v_bb else 0.0
    return share * pot - total["btn"]


def match(a, b, hands, seed=0):
    """Duplicate match: a's net bb per deal (both seats), returns an array of per-deal results."""
    rng = np.random.default_rng(seed)
    out = np.zeros(hands)
    for k in range(hands):
        cards = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        h1, h2, board = cards[:2], cards[2:4], cards[4:]
        out[k] = play_hand({"btn": a, "bb": b}, {"btn": h1, "bb": h2}, board) \
            - play_hand({"btn": b, "bb": a}, {"btn": h1, "bb": h2}, board)
    return out


def main():
    hands = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    opp = sys.argv[2] if len(sys.argv) > 2 else "station"
    a = EngineAgent(Engine(exact_required=False, seed=1), "engine (solver flops)")
    b = {"station": Station(), "maniac": Maniac(), "model": EngineAgent(Engine(exact_required=False, use_solver_flops=False, seed=2), "engine (model flops)"),
         "self": EngineAgent(Engine(exact_required=False, seed=2), "engine (solver flops)"), "aggro": Aggro(Engine(exact_required=False, seed=2))}[opp]
    t0 = time.time()
    r = match(a, b, hands)
    per100 = r.mean() / 2 * 100                              # two hands per deal
    se = r.std(ddof=1) / np.sqrt(hands) / 2 * 100
    print(f"{a.name} vs {b.name}: {per100:+.1f} ± {se:.1f} bb/100 over {2 * hands} hands "
          f"({time.time() - t0:.0f}s); decisions by source: {a.sources}")


if __name__ == "__main__":
    main()
