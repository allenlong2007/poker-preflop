"""Chipzen-format freezeouts offline: 10000 chips, blinds up every 20 hands, duplicate seats.

Heroes:   v8 (micro_pushfold on, the live Chipzen config), v8w (+ wide_3bet), v8_off (micro_pushfold off)
Opponents: v8_off, shover (PluriBot-like: shoves a wide range preflop, calls shoves light, 3-bets to 3x),
           minraiser (the shover with min-raise 3-bets), nit (pebble/fledge-like)
Prints match win rate +- SE and mean hands per match, per (hero, opponent). Results -> output/chipzen_sim/.

Usage: python scripts/chipzen_freezeout.py [pairs per matchup, default 40] [workers, default 6] [first seed, default 1000]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "chipzen_sim"
MATCHUPS = tuple((h, o) for h in ("v8wr", "v8wr20") for o in ("small3", "nitsmall", "shover", "minraiser", "nit", "v8"))
SERIES = 1          # matches per seed and seat order


def _strength():
    """Hand class -> percentile (0 = best) by all-in equity vs a random hand, combo weighted."""
    from bot.pushfold import _matrix
    E, W, combos = _matrix()
    eq = (E * W).sum(1) / W.sum(1)
    order = np.argsort(-eq)
    pct = np.empty(169)
    pct[order] = np.cumsum(combos[order]) / combos.sum()
    return pct


class Rule:
    """A simple scripted bot. shove: open-shove range share, call: share that calls an all-in, open: raise share
    (2.5x) at deep stacks, cbet: bets pot postflop with a pair or better, else check/fold."""

    def __init__(self, shove, call, open_, deep_bb, seed, min3=False, x3=3.0):
        from bot.engine import INDEX, hand_class, parse_cards as parse
        self.pct, self.INDEX, self.hand_class, self.parse = _strength(), INDEX, hand_class, parse
        self.shove, self.call, self.open, self.deep, self.min3, self.x3 = shove, call, open_, deep_bb, min3, x3
        self.rng = np.random.default_rng(seed)

    def __call__(self, s):
        from bot.chipzen.adapter import big_blind
        import eval7
        hole = "".join(map(str, s.hole_cards))
        p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
        bb = big_blind(s.action_history, 100)
        eff = min(s.your_stack, s.opponent_stacks[0]) / bb
        valid = s.valid_actions
        allin_vs = s.opponent_stacks[0] == 0
        if s.phase == "preflop":
            if allin_vs or s.to_call > 4 * bb:
                return ("call", 0) if p <= self.call else ("fold", 0)
            if eff <= self.deep:
                if p <= self.shove and "raise" in valid:
                    return "raise", s.max_raise
            elif p <= self.open and "raise" in valid:
                size = 2.5 * bb if s.to_call <= bb // 2 else 0 if self.min3 else self.x3 * (s.to_call + bb)  # x3 or min
                return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(size)))
            if s.to_call and p > self.open:
                return "fold", 0
            return ("check", 0) if "check" in valid else ("call", 0)
        cards = [eval7.Card(c) for c in [hole[:2], hole[2:]] + ["".join(map(str, [b])) for b in s.board]]
        made = eval7.handtype(eval7.evaluate(cards)) != "High Card"
        if made and "raise" in valid:
            return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, s.pot))
        if made and "call" in valid:
            return "call", 0
        return ("check", 0) if "check" in valid else ("fold", 0)


def decider(name, seed):
    from bot.chipzen.adapter import Decider, v8_engine
    from bot.engine import Engine
    if name == "v8":
        e = v8_engine()
    elif name in ("v8wr", "v8wr20"):               # wide_3bet + reraise_defend; v8wr20: min-3-bet up to 2.0x
        e = v8_engine(wide_3bet=True, reraise_defend=True)
        if name == "v8wr20":
            e.wide3_min_ratio = 2.0
    elif name.startswith("v8w"):                   # + the wide_3bet read; v8w50 = 4-bet at 50% equity
        e = v8_engine(wide_3bet=True)
        if name[3:]:                               # v8w45m35: 45% vs normal 3-bets, 35% vs min-3-bets
            a, _, b = name[3:].partition("m")
            e.wide3_eq = e.wide3_min_eq = int(a) / 100
            if b:
                e.wide3_min_eq = int(b) / 100
    elif name == "v8_off":
        e = Engine(pot_odds=True, exact_required=False, postflop_depths=(25, 40, 60, 100), postflop_nearest=True)
    elif name == "shover":
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed)
    elif name == "minraiser":                      # the shover, but every 3-bet is a min-raise (naive bots do this)
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, min3=True)
    elif name == "small3":                         # the shover, 3-betting to 1.8x the open (PluriBot: 2.25 -> 4)
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, x3=1.8)
    elif name == "nitsmall":                       # the nit, 3-betting to 1.8x: a tight range behind a small size
        return Rule(shove=0.15, call=0.12, open_=0.30, deep_bb=12, seed=seed, x3=1.8)
    elif name == "nit":
        return Rule(shove=0.15, call=0.12, open_=0.30, deep_bb=12, seed=seed)
    e.rng = np.random.default_rng(seed)
    return Decider(engine=e)


def run(job):
    """A duplicate pair of SERIES-match series. Rows: (hero, opp, seed, swap, match index, winner, hands)."""
    from bot.chipzen.sim import freezeout
    hero, opp, seed = job
    out = []
    for swap in (False, True):
        for m in range(SERIES):
            s = seed * 10 + m
            d = {0: decider(hero, s * 2 + swap), 1: decider(opp, s * 2 + swap + 7919)}
            win, hands = freezeout(d, s, swap=swap)
            out.append((hero, opp, seed, swap, m, win, hands))
    return out


def main(pairs=40, workers=6, base=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(h, o, base + k) for h, o in MATCHUPS for k in range(pairs)]
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            rows += r
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} pairs, {time.time() - t0:.0f}s", flush=True)
    (OUT / "freezeout.json").write_text(json.dumps(rows))
    print(f"\n{'hero':8} {'opp':9} {'matches':>7} {'win%':>6} {'+-SE':>5} {'hands':>6}")
    for h, o in MATCHUPS:
        rs = [r for r in rows if r[0] == h and r[1] == o]
        w = np.array([1.0 if r[5] == 0 else 0.5 if r[5] is None else 0.0 for r in rs])
        print(f"{h:8} {o:9} {len(w):7d} {100 * w.mean():6.1f} {100 * w.std(ddof=1) / np.sqrt(len(w)):5.1f} "
              f"{np.mean([r[6] for r in rs]):6.1f}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]))
