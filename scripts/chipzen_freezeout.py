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


class JarvisBot:
    """Jarvis as scouted 2026-11-18 (325 hands): SB opens 80% to 3x deep / 64% mid, jams 55% under 15bb, never limps;
    BB folds ~65% to opens (3-bets 8%, calls the rest); folds 82% to 3-bets. Postflop it bets 2/3 pot into a check
    or as the first actor ~80% on the flop with any two, ~55% on the turn, the river mostly for value; vs our bets
    it folds ~2/3 of misses on the flop, is sticky on the turn, calls the river with top pair+."""

    def __init__(self, seed):
        from bot.engine import INDEX, hand_class, parse_cards as parse
        self.pct, self.INDEX, self.hand_class, self.parse = _strength(), INDEX, hand_class, parse
        self.rng = np.random.default_rng(seed)

    def _kind(self, hole, board):
        import eval7
        from bot.engine import weak_showdown
        h = [eval7.Card(hole[:2]), eval7.Card(hole[2:])]
        b = [eval7.Card(str(c)) for c in board]
        if eval7.evaluate(h + b) >> 24 >= 2 and eval7.evaluate(h + b) >> 24 > eval7.evaluate(b) >> 24:
            return "str"
        if not weak_showdown(h, b):
            return "top"
        if eval7.evaluate(h + b) >> 24 > eval7.evaluate(b) >> 24:
            return "pair"
        suits = [str(c)[1] for c in h + b]
        if len(b) < 5 and any(suits.count(x) == 4 and x in (hole[1], hole[3]) for x in set(suits)):
            return "draw"
        return "air"

    def __call__(self, s):
        a = self._act(s)
        if a[0] not in s.valid_actions:                              # e.g. a raise when only fold / call is left
            a = ("call", 0) if a[0] == "raise" and "call" in s.valid_actions else \
                ("check", 0) if "check" in s.valid_actions else ("fold", 0)
        return a

    def _act(self, s):
        from bot.chipzen.adapter import big_blind
        hole = "".join(map(str, s.hole_cards))
        p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
        bb = big_blind(s.action_history, 100)
        eff = min(s.your_stack, s.opponent_stacks[0]) / bb
        valid, r = s.valid_actions, self.rng.random()
        allin_vs = s.opponent_stacks[0] == 0
        to = lambda x: ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(x))))
        fold = ("check", 0) if "check" in valid else ("fold", 0)
        call = ("call", 0) if "call" in valid else ("check", 0)
        if s.phase == "preflop":
            if allin_vs:
                return call if p <= (0.20 if eff < 15 else 0.12) else fold
            if s.to_call == bb // 2:                                  # SB first in
                if eff < 15:
                    return ("raise", s.max_raise) if p <= 0.55 else fold
                return to(3 * bb) if p <= (0.80 if eff >= 35 else 0.64) else fold
            if s.to_call == 0:                                        # BB vs a limp
                return to(3.5 * bb) if p <= 0.31 and "raise" in valid else ("check", 0)
            if s.to_call <= 4 * bb:                                   # BB vs an open
                if p <= 0.08 and "raise" in valid:
                    return to(3.5 * (s.to_call + bb))
                return call if p <= (0.40 if eff >= 35 else 0.27) else fold
            return call if p <= 0.12 else fold                        # vs a 3-bet or bigger
        k = self._kind(hole, s.board)
        street = len(s.board) - 2                                    # 1 flop .. 3 river
        if not s.to_call:
            first = not any(a.get("phase") == s.phase and a.get("seat") != s.your_seat
                            for a in s.action_history if isinstance(a, dict))
            bet = {1: 0.93 if not first else 0.75, 2: 0.45, 3: 0.10}[street]
            if k in ("str", "top") and street < 3 or k == "str" or k == "top" and r < 0.7 \
                    or k in ("air", "draw", "pair") and r < bet:
                if "raise" in valid:
                    return to(0.66 * s.pot)
            return ("check", 0)
        if allin_vs:
            return call if k in ("str", "top") else fold
        if k == "str":
            return to(3 * s.to_call + s.pot // 2) if r < 0.4 and "raise" in valid else call
        if street == 1:
            if (k == "draw" and r < 0.5 or k == "top" and r < 0.3) and "raise" in valid:
                return to(3 * s.to_call + s.pot // 2)
            return call if k in ("top", "draw") or k == "pair" and r < 0.5 else fold
        if street == 2:
            return call if k in ("top", "pair", "draw") or r < 0.5 else fold
        return call if k == "top" or k == "pair" and r < 0.2 else fold


class TideBot(JarvisBot):
    """tidewater_rnr as scouted 2026-11-19 (576 hands): SB raises 64% (2-3x) / limps 30% / folds 6%; BB defends 81%
    vs opens (3-bets 18% to ~3x), raises 37% of limps; vs our 3-bet folds only 25% (4-bets 17%). Postflop: as the
    raiser c-bets 60% (flop bets are often air), checks the flop as the caller; turn and river bets are nearly always
    strong. Vs our bets it folds ~37%, calls down pairs, raises strong hands (~15%)."""

    def _act(self, s):
        from bot.chipzen.adapter import big_blind
        hole = "".join(map(str, s.hole_cards))
        p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
        bb = big_blind(s.action_history, 100)
        eff = min(s.your_stack, s.opponent_stacks[0]) / bb
        valid, r = s.valid_actions, self.rng.random()
        allin_vs = s.opponent_stacks[0] == 0
        to = lambda x: ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(x))))
        fold = ("check", 0) if "check" in valid else ("fold", 0)
        call = ("call", 0) if "call" in valid else ("check", 0)
        if s.phase == "preflop":
            if allin_vs:
                return call if p <= 0.40 else fold
            if s.to_call == bb // 2:                                  # SB first in
                if eff < 15:
                    return ("raise", s.max_raise) if p <= 0.35 else call if p <= 0.85 else fold
                if p <= 0.64:
                    return to((2.0 if r < 0.6 else 3.0) * bb)
                return call if p <= 0.94 else fold
            if s.to_call == 0:                                        # BB vs a limp
                return to(3 * bb) if p <= 0.37 and "raise" in valid else ("check", 0)
            if s.to_call <= 4 * bb:                                   # BB vs an open
                if p <= 0.18 and "raise" in valid:
                    return to(3 * (s.to_call + bb))
                return call if p <= 0.81 else fold
            if p <= 0.12 and "raise" in valid:                       # vs a 3-bet: 4-bet all-in, call wide
                return ("raise", s.max_raise)
            return call if p <= 0.50 else fold
        k = self._kind(hole, s.board)
        street = len(s.board) - 2
        if not s.to_call:
            mine_pfr = any(a.get("seat") == s.your_seat and a.get("action") in ("raise", "all_in")
                           and a.get("phase") == "preflop" for a in s.action_history if isinstance(a, dict))
            if street == 1:
                bet = (k in ("str", "top") or r < 0.5) if mine_pfr else (k == "str" and r < 0.5 or r < 0.02)
            elif street == 2:
                bet = k == "str" or k == "top" and r < 0.4 or r < 0.04
            else:
                bet = k == "str" or k == "top" and r < 0.2
            if bet and "raise" in valid:
                return to((0.33 if r < 0.3 else 0.66) * s.pot)
            return ("check", 0)
        if allin_vs:
            return call if k in ("str", "top") or k == "pair" and r < 0.5 else fold
        if k == "str":
            return to(3 * s.to_call + s.pot // 2) if r < 0.5 and "raise" in valid else call
        if street == 1:
            if k in ("top", "draw") and r < 0.35 and "raise" in valid:
                return to(3 * s.to_call + s.pot // 2)
            return call if k in ("top", "pair", "draw") or r < 0.4 else fold
        if street == 2:
            return call if k in ("top", "pair", "draw") or r < 0.15 else fold
        return call if k == "top" or k == "pair" and r < 0.85 or r < 0.1 else fold


class CuteBot(JarvisBot):
    """cutewalrus as scouted 2026-11-19 (452 hands): SB raises 92% deep / 77% mid to 2.5x, jams ~55% under 15bb, never
    limps; BB vs opens folds 45%, calls 35%, 3-bets 20%; folds 76% to our 3-bets. Postflop as the raiser bets 47% flop /
    33% turn / 35% river, mostly made hands; rarely leads as the caller. Vs our bets it calls ~70% on flop and turn
    (air included) but folds 64% on the river."""

    def _act(self, s):
        from bot.chipzen.adapter import big_blind
        hole = "".join(map(str, s.hole_cards))
        p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
        bb = big_blind(s.action_history, 100)
        eff = min(s.your_stack, s.opponent_stacks[0]) / bb
        valid, r = s.valid_actions, self.rng.random()
        allin_vs = s.opponent_stacks[0] == 0
        to = lambda x: ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(x))))
        fold = ("check", 0) if "check" in valid else ("fold", 0)
        call = ("call", 0) if "call" in valid else ("check", 0)
        if s.phase == "preflop":
            if allin_vs:
                return call if p <= 0.25 else fold
            if s.to_call == bb // 2:
                if eff < 15:
                    return ("raise", s.max_raise) if p <= 0.55 else to(2.5 * bb) if p <= 0.75 else fold
                return to(2.5 * bb) if p <= (0.92 if eff >= 35 else 0.77) else fold
            if s.to_call == 0:
                return to(3 * bb) if p <= 0.32 and "raise" in valid else ("check", 0)
            if s.to_call <= 4 * bb:
                if p <= 0.20 and "raise" in valid:
                    return to(3 * (s.to_call + bb))
                return call if p <= 0.55 else fold
            if p <= 0.06 and "raise" in valid:
                return ("raise", s.max_raise)
            return call if p <= 0.18 else fold
        k = self._kind(hole, s.board)
        street = len(s.board) - 2
        if not s.to_call:
            mine_pfr = any(a.get("seat") == s.your_seat and a.get("action") in ("raise", "all_in")
                           and a.get("phase") == "preflop" for a in s.action_history if isinstance(a, dict))
            made = k in ("str", "top") or k == "pair" and r < 0.5
            if mine_pfr:
                bet = made or k == "pair" and street == 1 or k in ("air", "draw") and r < {1: 0.15, 2: 0.03, 3: 0.03}[street]
                bet = bet and (street == 1 or k != "pair" or r < 0.4)
            else:
                bet = k == "str" and r < 0.6 or r < 0.02
            if bet and "raise" in valid:
                return to((1.3 if r < 0.12 else 0.66) * s.pot)
            return ("check", 0)
        if allin_vs:
            return call if k in ("str", "top") else fold
        if k == "str":
            return to(3 * s.to_call + s.pot // 2) if r < 0.15 and "raise" in valid else call
        if street == 1:
            return call if k in ("top", "pair", "draw") or r < 0.55 else fold
        if street == 2:
            return call if k in ("top", "pair", "draw") or r < 0.45 else fold
        return call if k == "top" or k == "pair" and r < 0.3 else fold


class CuteBot2(CuteBot):
    """CuteBot with the size reads of the 2026-11-19 evening rescout (565 HU + 65 multiway postflop hands). Fold share
    by our bet size: flop 18% at <=40% pot, 24-30% at 1/2-3/4, 59% at pot; turn 9% small, ~45% at 1/2-3/4, ~58% at pot+;
    river ~50% up to pot but 11/12 vs overbets and jams."""

    def _act(self, s):
        if s.phase == "preflop" or not s.to_call or s.opponent_stacks[0] == 0:
            return super()._act(s)
        hole = "".join(map(str, s.hole_cards))
        k, street, r = self._kind(hole, s.board), len(s.board) - 2, self.rng.random()
        frac = s.to_call / max(1, s.pot - s.to_call)
        call = ("call", 0)
        fold = ("fold", 0)
        if k == "str":
            if street < 3 and r < 0.15 and "raise" in s.valid_actions:
                return ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(3 * s.to_call + s.pot // 2))))
            return call
        if street == 3:
            if frac > 1.2:
                return call if k == "top" and r < 0.1 else fold
            return call if k == "top" or k == "pair" and r < 0.3 else fold
        if frac > 0.9:
            air = 0.15 if street == 1 else 0.12
            return call if k == "top" or k in ("pair", "draw") and r < 0.6 or r < air else fold
        air = (0.8 if frac <= 0.4 else 0.62) if street == 1 else (0.85 if frac <= 0.4 else 0.25)
        return call if k in ("top", "pair", "draw") or r < air else fold


class AutoBot(JarvisBot):
    """AutodidactInnovations as scouted 2026-11-19 (879 hands): SB limps 72% and raises the top 28% to 4x, never folds
    or jams; BB folds 71% to opens (calls 20%, 3-bets 9% to ~2.4x); calls ~2/3 of iso-raises after a limp. Postflop
    passive: bets mostly made hands; vs our bets raises the flop ~19% with pairs+, calls ~54% (air included), then
    folds 46% on the turn and 60% on the river."""

    def _act(self, s):
        from bot.chipzen.adapter import big_blind
        hole = "".join(map(str, s.hole_cards))
        p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
        bb = big_blind(s.action_history, 100)
        valid, r = s.valid_actions, self.rng.random()
        allin_vs = s.opponent_stacks[0] == 0
        to = lambda x: ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(x))))
        fold = ("check", 0) if "check" in valid else ("fold", 0)
        call = ("call", 0) if "call" in valid else ("check", 0)
        pre = [a for a in s.action_history if isinstance(a, dict) and a.get("phase") == "preflop"
               and not str(a.get("action", "")).startswith("post")]
        limped = any(a.get("seat") == s.your_seat and a.get("action") == "call" for a in pre)
        if s.phase == "preflop":
            if allin_vs:
                return call if p <= 0.20 else fold
            if s.to_call == bb // 2:
                return to(4 * bb) if p <= 0.28 else call
            if s.to_call == 0:
                return to(3.5 * bb) if p <= 0.37 and "raise" in valid else ("check", 0)
            if limped:                                                 # our iso-raise over its limp
                return call if p <= 0.76 else fold
            if s.to_call <= 4 * bb:
                if p <= 0.09 and "raise" in valid:
                    return to(2.4 * (s.to_call + bb))
                return call if p <= 0.29 else fold
            if p <= 0.10 and "raise" in valid:
                return ("raise", s.max_raise)
            return call if p <= 0.22 else fold
        k = self._kind(hole, s.board)
        street = len(s.board) - 2
        if not s.to_call:
            mine_pfr = any(a.get("seat") == s.your_seat and a.get("action") in ("raise", "all_in") for a in pre)
            if street == 1:
                bet = k in ("str", "top") or k == "pair" and r < (0.3 if mine_pfr else 0.05) or r < 0.02
            else:
                bet = k == "str" or k == "top" and r < 0.5 or r < 0.02
            if bet and "raise" in valid:
                return to((0.9 if r < 0.15 else 0.66) * s.pot)
            return ("check", 0)
        if allin_vs:
            return call if k in ("str", "top") else fold
        if street == 1:
            if k in ("str", "top", "pair") and r < 0.65 and "raise" in valid:
                return to(3 * s.to_call + s.pot // 2)
            return call if k in ("str", "top", "pair", "draw") or r < 0.5 else fold
        if street == 2:
            if k == "str" and r < 0.3 and "raise" in valid:
                return to(3 * s.to_call + s.pot // 2)
            return call if k in ("str", "top", "draw") or k == "pair" and r < 0.7 or r < 0.15 else fold
        return call if k in ("str", "top") or k == "pair" and r < 0.5 else fold


class AutoBot2(AutoBot):
    """AutoBot with the size reads of the 2026-11-20 rescout (1819 HU hands, seasons 12-15, style unchanged).
    Folds a limp to our iso-raise 14% under 3bb, 24% at 3-4bb, 36% at 4-5bb, ~50% at 5-6.5bb; to a jam 72% at 10bb+
    and 15% under 10bb; never
    re-raises a limp. Facing postflop bets, fold share by size (<=40% / 1/2 / 3/4 / pot / over): flop 9/21/32/43/60%,
    turn 42/44/47/65/~55%, river 37/38/56/61/~50%. River bets are near-pure value."""

    FOLD = {1: (0.09, 0.21, 0.32, 0.43, 0.60), 2: (0.42, 0.44, 0.47, 0.65, 0.55), 3: (0.37, 0.38, 0.56, 0.61, 0.50)}
    limp_jam_fold = None
    jam_call = None                              # share of hands that call a first-in jam (None = AutoBot's 22%)
    loose_jam = False                            # auto2L: calls 48% of limps vs a jam at any depth (sensitivity)
    WEAK = {1: 1.10, 2: 0.70, 3: 0.75}           # calibration divisors (sim fold share matched to the scouted one)

    def _act(self, s):
        from bot.chipzen.adapter import big_blind
        bb = big_blind(s.action_history, 100)
        if s.phase == "preflop":
            pre = [a for a in s.action_history if isinstance(a, dict) and a.get("phase") == "preflop"
                   and not str(a.get("action", "")).startswith("post")]
            limped = any(a.get("seat") == s.your_seat and a.get("action") == "call" for a in pre)
            if limped and s.to_call and s.opponent_stacks[0] > 0 or limped and s.opponent_stacks[0] == 0:
                hole = "".join(map(str, s.hole_cards))
                p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
                to = (s.to_call + bb) / bb                 # the iso-raise size in bb
                if s.opponent_stacks[0] == 0:             # a jam: real calls 8/29 at 10bb+, 13/15 under 10bb
                    f = self.limp_jam_fold if self.limp_jam_fold is not None else 0.52 if self.loose_jam else 0.72 if to >= 10 else 0.15
                else:
                    f = 0.14 if to < 3 else 0.24 if to < 4 else 0.36 if to < 5 else 0.50 if to < 6.5 else 0.60
                return ("call", 0) if p <= 0.28 + 0.72 * (1 - f) else ("fold", 0)
            if self.jam_call is not None and s.opponent_stacks[0] == 0 and not limped:   # facing a first-in jam
                hole = "".join(map(str, s.hole_cards))
                p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
                return ("call", 0) if p <= self.jam_call else ("fold", 0)
            return super()._act(s)
        if not s.to_call or s.opponent_stacks[0] == 0:
            return super()._act(s)
        hole = "".join(map(str, s.hole_cards))
        k, street, r = self._kind(hole, s.board), len(s.board) - 2, self.rng.random()
        frac = s.to_call / max(1, s.pot - s.to_call)
        i = 0 if frac <= 0.4 else 1 if frac <= 0.6 else 2 if frac <= 0.85 else 3 if frac <= 1.2 else 4
        if k in ("str", "top"):
            if street < 3 and k == "str" and r < 0.2 and "raise" in s.valid_actions:
                return ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(3 * s.to_call + s.pot // 2))))
            return ("call", 0)
        if street == 1 and k == "pair" and i in (1, 2) and r < 0.12 and "raise" in s.valid_actions:
            return ("raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(3 * s.to_call + s.pot // 2))))
        q = min(1.0, self.FOLD[street][i] / self.WEAK[street])
        q = min(1.0, q * (1.3 if k == "air" else 0.75 if k in ("pair", "draw") else 1.0))
        return ("fold", 0) if self.rng.random() < q else ("call", 0)


class AlizBot(Rule):
    """alizabethbot3s model from the 2026-11-20 exhibition: opens ~95% to 2.5bb, 3-bets ~28% of our opens,
    calls jams wide, floats small flop bets, then barrels the turn when checked to (value + air)."""

    def __init__(self, seed, float_=0.75, barrel=0.55):
        super().__init__(shove=0.55, call=0.36, open_=0.95, deep_bb=12, seed=seed, x3=3.2, jam=0.0)
        self.float_, self.barrel, self.three = float_, barrel, 0.28

    def __call__(self, s):
        from bot.engine import weak_showdown
        from bot.chipzen.adapter import big_blind
        import eval7
        hole = "".join(map(str, s.hole_cards))
        valid = s.valid_actions
        if s.phase == "preflop":
            p = self.pct[self.INDEX[self.hand_class(self.parse(hole))]]
            bb = big_blind(s.action_history, 100)
            if "raise" in valid and s.opponent_stacks[0] and bb // 2 < s.to_call <= 4 * bb and p <= self.three:
                size = int(self.x3 * (s.to_call + bb))
                return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, size))
            return super().__call__(s)
        cards = [eval7.Card(c) for c in [hole[:2], hole[2:]] + ["".join(map(str, [b])) for b in s.board]]
        made = eval7.handtype(eval7.evaluate(cards)) != "High Card"
        strong = not weak_showdown(cards[:2], cards[2:])
        street = {3: "flop", 4: "turn", 5: "river"}.get(len(s.board), "flop")
        r = self.rng.random()
        def bet(frac):
            return "raise", max(min(s.min_raise, s.max_raise), min(s.max_raise, int(s.pot * frac)))
        if not s.to_call:
            if "raise" not in valid:
                return "check", 0
            if street == "flop" and (made or r < 0.5):
                return bet(0.33)
            if street == "turn" and (made or r < self.barrel):
                return bet(0.66)
            if street == "river" and (strong or r < 0.3):
                return bet(0.75)
            return "check", 0
        frac = s.to_call / max(1, s.pot - s.to_call)
        if street == "flop":
            ok = made or r < (self.float_ if frac <= 0.6 else 0.35)
        elif frac <= 0.6:
            ok = strong or made and r < 0.6 or r < 0.2
        else:
            ok = strong and r < 0.7
        return ("call", 0) if ok and "call" in valid else ("fold", 0)


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
            elif part in ("w", "F", "S", "T") or part[:1] == "X":
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
            elif part[0] == "V":                     # +V50: value_up moves 50% of value checks/small bets to the top size
                e.value_up = int(part[1:]) / 100
            elif part[0] == "Y":                     # +Y6: value_up counts groups up to 6 (middle pair) as value
                e.value_bkt = int(part[1:])
            elif part[0] == "N":                     # +Nv75-100-100_b100-50-150_s100-100-130: minmax sizes (pot %)
                mm = {}
                for seg in part[1:].split("_"):
                    vals = tuple(int(x) / 100 for x in seg[1:].split("-"))
                    mm[{"v": "value", "b": "bluff", "s": "scale"}[seg[0]]] = vals
                e.minmax = mm
            elif part[0] == "M":                     # +M100: value_up targets the largest bet <= 1.00 pot
                e.value_max = int(part[1:]) / 100
            elif part[0] == "Q":                     # +Q60: bluff_big moves 60% of river air to the overbet
                e.bluff_big = int(part[1:]) / 100
            elif part[0] == "I":                     # +I65: iso-raise a limp to 6.5bb (I0 = the solved size)
                e.iso_to = int(part[1:]) / 10
            elif part[0] == "W":                     # +W60: hands >= 60% equity vs random jam first-in / over a limp
                e.jam_wide = int(part[1:]) / 100
            elif part[0] == "P":                     # +P30: over a limp, hands >= 30% equity vs random jam
                e.jam_limp = int(part[1:]) / 100
            elif part[0] == "O":                     # +O60: hands >= 60% equity vs random jam over an open
                e.jam_vs_open = int(part[1:]) / 100
            elif part[0] == "C":                     # +C8: short-stack calls need 8 points of equity over the price
                e.short_call_margin = int(part[1:]) / 100
            elif part[0] == "J":                     # +J50: over a limp, hands under 50% equity vs random never jam
                e.limp_jam_eq = int(part[1:]) / 100
            elif part[0] == "L":                     # +L50: move 50% of the BB's checks over a limp to the iso-raise
                e.iso_more = int(part[1:]) / 100
            elif part[0] == "K":                     # +K50: steal moves 50% of button fold/limp to the open
                e.steal = int(part[1:]) / 100
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
    elif name == "auto":                           # AutodidactInnovations model (AutoBot)
        return AutoBot(seed)
    elif name == "auto2":                          # AutodidactInnovations size-aware model (AutoBot2)
        return AutoBot2(seed)
    elif name in ("auto2W", "auto2T"):             # jam-call sensitivity: wide (40% / limps 52% call) or tight (12% / 15%)
        b = AutoBot2(seed)
        b.jam_call, b.limp_jam_fold = (0.40, 0.48) if name == "auto2W" else (0.12, 0.85)
        return b
    elif name == "auto2L":                         # AutoBot2 that calls jams over its limp 48% at any depth
        b = AutoBot2(seed)
        b.loose_jam = True
        return b
    elif name == "aliz":                          # alizabethbot3s model (AlizBot)
        return AlizBot(seed)
    elif name == "alizT":                         # AlizBot that barrels the turn only 30% (honest-ish)
        return AlizBot(seed, barrel=0.30)
    elif name == "cute":                           # cutewalrus model (CuteBot)
        return CuteBot(seed)
    elif name == "cute2":                           # cutewalrus model (CuteBot2)
        return CuteBot2(seed)
    elif name == "tide":                           # tidewater_rnr model (TideBot)
        return TideBot(seed)
    elif name == "jarvis2":                        # the faithful Jarvis model (JarvisBot)
        return JarvisBot(seed)
    elif name == "jarvis":                         # Jarvis-like: opens 77% of buttons at 3x, never limps, folds 82%
        #                                            to a 3-bet, bets 2/3 pot when checked to about half the time
        return Rule(shove=0.30, call=0.18, open_=0.77, deep_bb=12, seed=seed, bluff=0.5)
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


F3_SEEDS = {"lilb": (13, 17), "jarvis": (14, 17), "jarvis2": (14, 17), "tide": (3, 12), "cute": (22, 32), "cute2": (22, 32), "auto": (3, 13), "auto2": (3, 13), "auto2L": (3, 13), "auto2W": (3, 13), "auto2T": (3, 13), "aliz": (12, 13), "alizT": (12, 13)}                       # folds to our 3-bets, 3-bets faced
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
            xt = {"b": "x_bluff", "f": "x_def_flop", "t": "x_def_turn", "r": "x_def_river"}
            for part in hero.split("+")[1:]:              # +Xb120 / +Xf150 / +Xt150 / +Xr100: exploit-scale tune
                if part[:1] == "X" and part[1:2] in xt:
                    d[0].tune.setdefault("unknown", {})[xt[part[1]]] = int(part[2:]) / 100
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
