"""Experiment 42: train how strict the bot is against big bets and raises, from opponent data.

Strictness s (0 loose .. 1 strict) blends the value-heavy priors (the learned bluff model's share, the two-pair
floor for value raises, fewer bluffs in raises) with a loose read (the break-even bluff share scaled by the
opponent's aggression). Per opponent s = clip(S0 - K * ln(aggression), 0, 1), aggression = their postflop raise
and bet rates relative to the bot's own self-play.

Training: for each opponent of a population (itself, two loose-aggressive players, a maniac, a value-heavy
player) and two amounts of history (an early read from 20 deals, a settled one from 150), the bot is adapted to
that read and plays a duplicate match at each of several fixed strictness values (common random cards). (S0, K)
are then chosen to maximise the average result over the population, with s read off each opponent's profile.

Usage:
    python scripts/strictness_train.py [deals per match]      (default 800; ~15 min on 10 cores)
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "match" / "strictness_train.json"
S_GRID = (0.0, 1.0, "evidence")
HISTORY = (20, 150)
OPPONENTS = ("itself", "aggro 0.2", "aggro 0.4", "maniac", "value-heavy")


def make(name):
    from bot.engine import Engine
    from bot.selfplay import Aggro, EngineAgent, Maniac, ValueHeavy
    return {"itself": lambda: EngineAgent(Engine(exact_required=False, seed=2)), "aggro 0.2": lambda: Aggro(Engine(exact_required=False, seed=2), rate=0.2),
            "aggro 0.4": lambda: Aggro(Engine(exact_required=False, seed=2), rate=0.4), "maniac": Maniac,
            "value-heavy": lambda: ValueHeavy(Engine(exact_required=False, seed=2))}[name]()


def profile_of(name, deals, seed=5):
    from bot.engine import Engine
    from bot.opponent import profile
    from bot.selfplay import DECK, EngineAgent, play_hand
    rng = np.random.default_rng(seed)
    eng, opp = EngineAgent(Engine(exact_required=False, seed=9)), make(name)

    class Rec:
        def __init__(self, a):
            self.a, self.h = a, None

        def act(self, seat, hole, board, history, facing):
            self.h = history
            return self.a.act(seat, hole, board, history, facing)

    E1, M1 = Rec(eng), Rec(opp)
    recs = []
    for k in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        for o in ("btn", "bb"):
            E1.h = M1.h = None
            play_hand({"btn": M1, "bb": E1} if o == "btn" else {"btn": E1, "bb": M1}, {"btn": c[:2], "bb": c[2:4]}, c[4:])
            h = [[[x.player, x.kind, x.to] for x in st] for st in (E1.h if E1.h is not None else M1.h)]
            showdown = not any(k == "fold" for st in h for _, k, _ in st)
            opp_hole = c[:2] if o == "btn" else c[2:4]
            recs.append({"number": k, "opp_pos": o, "hero_pos": "bb" if o == "btn" else "btn", "history": h,
                         "stacks": {"hero": 100, "opp": 100},
                         "shown": {"opp": "".join(str(x) for x in opp_hole)} if showdown else {},
                         "board": "".join(str(x) for x in c[4:]),
                         # Identities the read gates (bot/reads.py) require; without them every read failed closed
                         # and these offline reads silently did nothing (found in experiment 50).
                         "id": f"sim-{seed}-{k}-{o}", "opp_id": f"sim-{name}", "hero_id": "bot"})
    prof = profile(recs)
    prof["records"] = recs
    return prof


def read(name, deals):
    """The engine's opponent state after `deals` deals of their play, through the live adapter's read path
    (bot.adapt.read_state: evidence gates, preflop solve, per-street postflop scales, strictness, bluff model)."""
    from bot.adapt import BASELINE, Preflop, read_state
    state, _, _ = read_state(profile_of(name, deals)["records"], json.load(open(BASELINE)), Preflop())
    return state


def setup(job):
    """The read for one (opponent, history): shared by every strictness value. Sent between processes, so the
    size scale (a local function) is rebuilt in apply()."""
    name, hist = job
    state = read(name, hist)
    state.pop("size_value")
    return (name, hist), state


def apply(e, state, s="evidence"):
    """Load a read into the engine as bot.adapt.Adapter.apply does; s overrides the read's strictness."""
    from bot.adapt import size_value
    e.set_adaptation(state["nodes"], state["lam"], state["post"])
    e.opp_profile, e.size_value = state["profile"], size_value(state["profile"])[0]
    e.bluff_model = state["bluff_model"]
    e.strict = state["strict"] if s == "evidence" else s
    e.read_evidence = state["evidence"]


def play(job):
    (name, hist), s, state, deals = job
    from bot.engine import Engine
    from bot.selfplay import EngineAgent, match
    e = Engine(exact_required=False, seed=1, pot_odds=True)
    apply(e, state, s)
    r = match(EngineAgent(e), make(name), deals, seed=3)
    return (name, hist, s), float(r.mean() / 2 * 100), float(r.std(ddof=1) / np.sqrt(len(r)) / 2 * 100)


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    with Pool(10) as pool:
        ctx = dict(pool.map(setup, list(itertools.product(OPPONENTS, HISTORY))))
        jobs = [(key, s, ctx[key], deals) for key in ctx for s in S_GRID]
        res = {k: (m, se) for k, m, se in pool.map(play, jobs)}
    print(f"bb/100 ({deals} deals each, duplicate):  always loose (s=0) | always strict (s=1) | evidence-based")
    tot = {x: [] for x in S_GRID}
    for (name, hist), c in ctx.items():
        cells = []
        for x in S_GRID:
            m, se = res[(name, hist, x)]
            tot[x].append(m)
            cells.append(f"{m:+6.0f} ± {se:3.0f}")
        sv = c["strict"]
        print(f"   {name:12} history {hist:3d}: " + " | ".join(cells) + f"   (evidence s = {sv:.2f})")
    for x in S_GRID:
        v = tot[x]
        print(f"   average {str(x):9}: all {np.mean(v):+.1f}   without the maniac "
              f"{np.mean([m for (n, h), m in zip(ctx, v) if n != 'maniac']):+.1f}")
    OUT.write_text(json.dumps({"deals": deals, "results": {f"{n}|{h}|{x}": v for (n, h, x), v in res.items()}}, indent=1))


if __name__ == "__main__":
    main()
