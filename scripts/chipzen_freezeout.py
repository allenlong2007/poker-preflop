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
OPPS = ("small3", "small3b", "shover", "minraiser", "nit", "v8", "station", "maniac", "limper", "jammer", "chaos")
MATCHUPS = tuple((h, o) for h in ("v8wr",) for o in OPPS)
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

    def __init__(self, shove, call, open_, deep_bb, seed, min3=False, x3=3.0, bluff=0.0, limp=0.0, station=0.0,
                 aggro=0.0, jam=0.0, jam_any=0.0, big=0.0, tell=0.0, rev=False, fit=False):
        from bot.engine import INDEX, hand_class, parse_cards as parse
        self.pct, self.INDEX, self.hand_class, self.parse = _strength(), INDEX, hand_class, parse
        self.shove, self.call, self.open, self.deep, self.min3, self.x3 = shove, call, open_, deep_bb, min3, x3
        self.bluff = bluff
        self.limp = limp                # share of button opens that limp instead of raising
        self.station = station          # postflop: never raises a made hand, calls air this often
        self.aggro = aggro              # postflop: raises / bets any two cards this often
        self.jam = jam                  # preflop at any depth: re-raises all-in with the top `jam` share
        self.jam_any = jam_any          # preflop at any depth: re-raises (or opens) all-in with any two this often
        self.big = big                  # preflop: re-raises to 60% of its stack (not all-in) with the top `big` share
        self.tell = tell                # postflop: sizing tell. Bets 1/3 pot with air/weak pairs (bluffing this often
        #                                 when checked to), pot with top pair or better; calls with any pair
        self.rev = rev                  # the reverse tell: small with value, pot with air
        self.fit = fit                  # postflop: only top pair / overpair or better counts as made (folds the rest)
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
            if "raise" in valid and not allin_vs and (s.to_call > bb // 2 and p <= self.jam
                                                       or self.jam_any and self.rng.random() < self.jam_any):
                return "raise", s.max_raise
            if "raise" in valid and not allin_vs and s.to_call > bb // 2 and p <= self.big:
                size = int(0.6 * (s.your_stack + (s.max_raise - s.your_stack)))  # 60% of what it started the street with
                return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise - 1, size))
            if allin_vs or s.to_call > 4 * bb:
                return ("call", 0) if p <= self.call else ("fold", 0)
            if eff <= self.deep:
                if p <= self.shove and "raise" in valid:
                    return "raise", s.max_raise
            elif p <= self.open and "raise" in valid and not (s.to_call == bb // 2 and self.rng.random() < self.limp):
                size = 2.5 * bb if s.to_call <= bb // 2 else 0 if self.min3 else self.x3 * (s.to_call + bb)  # x3 or min
                return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(size)))
            if s.to_call and p > self.open:
                return "fold", 0
            return ("check", 0) if "check" in valid else ("call", 0)
        cards = [eval7.Card(c) for c in [hole[:2], hole[2:]] + ["".join(map(str, [b])) for b in s.board]]
        made = eval7.handtype(eval7.evaluate(cards)) != "High Card"
        if self.fit:
            from bot.engine import weak_showdown
            made = not weak_showdown(cards[:2], cards[2:])
        if self.tell:
            from bot.engine import weak_showdown
            weak = len(s.board) < 3 or weak_showdown(cards[:2], cards[2:])
            if "raise" in valid and not s.to_call:
                big, small = (s.pot // 3, s.pot) if self.rev else (s.pot, s.pot // 3)
                if not weak:
                    return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, big))
                if self.rng.random() < self.tell:
                    return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, small))
            if s.to_call:
                return ("call", 0) if made and "call" in valid else ("fold", 0)
            return "check", 0
        if self.station:
            if s.to_call:
                return ("call", 0) if made or self.rng.random() < self.station else ("fold", 0)
            return "check", 0
        if self.aggro and self.rng.random() < self.aggro and "raise" in valid:
            return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, s.pot))
        if (made or not s.to_call and self.rng.random() < self.bluff) and "raise" in valid:
            return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, s.pot))
        if made and "call" in valid:
            return "call", 0
        return ("check", 0) if "check" in valid else ("fold", 0)


def decider(name, seed):
    from bot.chipzen.adapter import Decider, v8_engine
    from bot.engine import Engine
    if name == "v8":
        e = v8_engine()
    elif name.startswith("v8wr"):                  # the live config (wide_3bet + reraise_defend); v8wr+m5: short-stack
        flags = name.split("+")[0][4:]           # v8wra: + aggro_defend, v8wraj: + deep_jam; v8wra+a50: from 50%
        e = v8_engine(wide_3bet=True, reraise_defend=True, aggro_defend="a" in flags, deep_jam="j" in flags,
                      light_3bet="l" in flags, river_air="v" in flags,   # v8wrajl: + light_3bet, v: river_air
                      size_tell="s" in flags, bet_more="b" in flags)        # s: size_tell, b: bet_more
        for part in name.split("+")[1:]:
            if part[0] == "m":                       # calls need N points of equity over the price
                e.short_call_margin = int(part[1:]) / 100
            elif part in ("w", "F", "S", "T"):
                pass
            elif part[0] == "b":                     # +b0: deep_jam on true all-ins only
                e.deepjam_borrow_frac = int(part[1:]) / 100
            elif part[0] == "a":
                e.aggro_on = int(part[1:]) / 100
            elif part[0] == "r":                     # +r60: call a wide 3-bet when 60% of equity beats the price
                e.wide3_real = int(part[1:]) / 100
            elif part[0] == "o":                     # +o60: light_3bet acts from a 60% fold-to-3-bet share
                e.light3_on = int(part[1:]) / 100
            elif part[0] == "v":                     # +v2: river_air mode 2
                e.river_air_mode = int(part[1:])
            elif part[0] == "f":                     # +f50: bet_more acts from a 50% fold share
                e.betmore_on = int(part[1:]) / 100
            elif part[0] == "x":                     # +x25: bet_more scales bluffs at most x2.5
                e.betmore_max = int(part[1:]) / 10
            elif part[0] == "g":                     # +g8: bet_more also scales strong draws (hand group 8)
                e.betmore_bkt = int(part[1:])
            elif part[0] == "e":                     # +e40: 4-bet a wide (normal-size) 3-bet at 40% equity
                e.wide3_eq = int(part[1:]) / 100
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
    elif name == "small3b":                        # small3 that also bets pot with 40% of its misses when checked to
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, x3=1.8, bluff=0.4)
    elif name == "nitsmall":                       # the nit, 3-betting to 1.8x: a tight range behind a small size
        return Rule(shove=0.15, call=0.12, open_=0.30, deep_bb=12, seed=seed, x3=1.8)
    elif name == "station":                        # opens wide, limps half, calls down any pair and 50% of air
        return Rule(shove=0.30, call=0.50, open_=0.90, deep_bb=12, seed=seed, limp=0.5, station=0.5)
    elif name == "maniac":                         # opens 95%, 3-bets 3x, pot-bets/raises 60% of the time postflop
        return Rule(shove=0.60, call=0.55, open_=0.95, deep_bb=15, seed=seed, aggro=0.6)
    elif name == "limper":                         # small3b that limps 70% of its buttons
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, x3=1.8, bluff=0.4, limp=0.7)
    elif name == "teller":                         # alizabeth-like: opens 90%, 1/3-pot bluffs, pot-size value
        return Rule(shove=0.55, call=0.30, open_=0.90, deep_bb=30, seed=seed, x3=3.0, tell=0.5)
    elif name == "teller_rev":                     # the reverse tell: 1/3-pot value, pot-size bluffs
        return Rule(shove=0.55, call=0.30, open_=0.90, deep_bb=30, seed=seed, x3=3.0, tell=0.5, rev=True)
    elif name == "folder":                         # stebin-like: opens 85% small, folds ~88% of hands to a 3-bet
        return Rule(shove=0.55, call=0.12, open_=0.85, deep_bb=30, seed=seed, x3=1.8, bluff=0.4)
    elif name == "lilb":                           # lil-b2c-like: folds 45% / limps 25% / opens 30%, folds most hands
        #                                            to a 3-bet, fit-or-fold postflop, stabs 12% of misses when checked to
        return Rule(shove=0.30, call=0.15, open_=0.55, deep_bb=12, seed=seed, limp=0.45, bluff=0.12, fit=True)
    elif name == "jammer":                         # small3b that answers any raise all-in with its top 12%
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, x3=1.8, bluff=0.4, jam=0.12)
    elif name == "chaos":                          # small3b that jams any two cards 25% of the time, calls jams 50%
        return Rule(shove=0.55, call=0.50, open_=0.80, deep_bb=30, seed=seed, x3=1.8, bluff=0.4, jam_any=0.25)
    elif name == "bigraiser":                      # small3b that re-raises to 60% of its stack with its top 15%
        return Rule(shove=0.55, call=0.40, open_=0.80, deep_bb=30, seed=seed, x3=1.8, bluff=0.4, big=0.15)
    elif name == "nit":
        return Rule(shove=0.15, call=0.12, open_=0.30, deep_bb=12, seed=seed)
    e.rng = np.random.default_rng(seed)
    return Decider(engine=e)


F3_SEEDS = {"lilb": (13, 17)}                       # folds to our 3-bets, 3-bets faced
FOLD_SEEDS = {"lilb": (39, 60)}                    # folds to our postflop bets, times faced
SIZE_SEEDS = {"teller": ((10, 10), (0, 10)), "teller_rev": ((0, 10), (10, 10))}   # (weak, n) small, big


def run(job):
    """A duplicate pair of SERIES-match series. Rows: (hero, opp, seed, swap, match index, winner, hands)."""
    from bot.chipzen.sim import freezeout
    hero, opp, seed = job
    out = []
    for swap in (False, True):
        for m in range(SERIES):
            s = seed * 10 + m
            d = {0: decider(hero, s * 2 + swap), 1: decider(opp, s * 2 + swap + 7919)}
            if "+T" in hero and opp in F3_SEEDS:      # +T: the scouted fold-to-3-bet read (lil-b2c's 13 of 17)
                d[0].f3["unknown"] = list(F3_SEEDS[opp])
            if "+F" in hero and opp in FOLD_SEEDS:     # +F: a scouted bet_more fold read (lil-b2c's real 65%)
                d[0].folds["unknown"] = list(FOLD_SEEDS[opp])
            if "+S" in hero and opp in SIZE_SEEDS:     # +S: a scouted size_tell read (the opponent's true pattern)
                d[0].sizes["unknown"] = [list(c) for c in SIZE_SEEDS[opp]]
            warm = next((int(p[1:]) if p[1:].isdigit() else 1 for p in hero.split("+")[1:] if p[:1] == "w"), 0)
            for k in range(warm):                  # +w / +wN: hero first plays N warm-up matches vs the same bot type,
                freezeout({0: d[0], 1: decider(opp, s * 2 + swap + 31337 + k)}, s + 10**7 + k, swap=swap)  # reads carry
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
