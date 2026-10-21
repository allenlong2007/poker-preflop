"""Experiment 35: is the bot too tight out of position postflop? Self-play, BB fold rates vs MDF and BB leads.

Self-play (engine vs engine) records every BB decision facing a BTN bet: street, bet size as a share of the
pot, and fold / call / raise. Minimum defense frequency: against a bet b into pot P the BB must continue at
least P / (P + b) of the time, or any two cards profit as a bluff. Also counts how often the BB leads
(bets first) on each street, and the check-raise rate.

Usage:
    python scripts/oop_check.py [deals]       (default 3000 deals = 6000 hands)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot.engine import Engine, State
from bot.selfplay import DECK, play_hand


class Recorder:
    def __init__(self, engine, log):
        self.engine, self.log = engine, log

    def act(self, seat, hole, board, history, facing):
        d = self.engine.decide(seat, hole, board, history)
        street = len(history) - 1
        if street > 0 and seat == "bb":
            st = State(history)
            self.log.append((street, facing, st.pot(), d.action, len(history[-1]) == 0))
        return d.action, d.to


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = np.random.default_rng(7)
    log = []
    a, b = Recorder(Engine(seed=1), log), Recorder(Engine(seed=2), log)
    for _ in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        play_hand({"btn": a, "bb": b}, {"btn": c[:2], "bb": c[2:4]}, c[4:])
    vs = defaultdict(lambda: defaultdict(int))
    lead = defaultdict(lambda: defaultdict(int))
    for street, facing, pot, act, first in log:
        name = ["", "flop", "turn", "river"][street]
        if facing > 1e-6:
            frac = facing / max(pot - facing, 1e-9)
            band = "<=40%" if frac <= 0.4 else "40-80%" if frac <= 0.8 else "80-120%" if frac <= 1.2 else ">120%"
            vs[(name, band)][act] += 1
            vs[(name, band)]["_mdf_fold"] += facing / pot          # max fold vs this bet: b / (P + b)
        elif first:
            lead[name]["bet" if act in ("bet", "raise") else "check"] += 1
    print("BB facing a bet (fold / call / raise), and the most it can fold (b / (pot + b)):")
    for k in sorted(vs):
        d = vs[k]
        n = sum(v for kk, v in d.items() if not kk.startswith("_"))
        print(f"  {k[0]:5} bet {k[1]:8} n={n:5}  fold {d['fold'] / n:5.1%}  call {d['call'] / n:5.1%}  "
              f"raise {(d['raise'] + d['bet']) / n:5.1%}   max fold {d['_mdf_fold'] / n:5.1%}")
    print("BB acting first (lead):")
    for s in ("flop", "turn", "river"):
        n = sum(lead[s].values())
        print(f"  {s:5} n={n:5}  bet {lead[s]['bet'] / max(n, 1):5.1%}")


if __name__ == "__main__":
    main()
