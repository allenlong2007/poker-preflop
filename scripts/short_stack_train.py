"""Experiment 49: short-stack play by duplicate self-play at short effective stacks.

Variants of the hero engine (bot/engine.py switches):
    old       nearest solved depth, 100bb opponent ranges, fixed 55% low-SPR jam rule (before experiment 49)
    new       depth-blended policies, ranges at the effective stack, range-vs-range EV at low SPR
    new_spr2  new, low-SPR mode up to 2 pots behind (default 1.2) or 12bb
    new_spr3  new, up to 3 pots behind
    ranges    old + depth-consistent ranges only (ablation)
Opponents: the old engine (head-to-head) and rival bots. Prints bb/100 +- SE per (variant, stack, opponent).

Usage: python scripts/short_stack_train.py [deals] [stacks e.g. 10,20,35] [variants] [opponents]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "match" / "short_stack_train.json"
VARIANTS = ("old", "new", "new_spr2", "new_spr3", "ranges")
OPPONENTS = ("old", "equity", "chart", "pushfold", "nit", "aggro")


def engine(variant, seed):
    from bot.engine import Engine
    e = Engine(exact_required=False, seed=seed, pot_odds=True)
    if variant in ("old", "ranges"):
        e.depth_blend, e.low_spr = False, "legacy"
        e.depth_ranges = variant == "ranges"
    if variant == "new_spr2":
        e.short_spr = 2.0
    if variant == "new_spr3":
        e.short_spr = 3.0
    return e


def opponent(name):
    from bot.rivals import make
    from bot.selfplay import Aggro, EngineAgent
    if name == "old":
        return EngineAgent(engine("old", 2))
    if name == "aggro":
        return Aggro(engine("old", 2), rate=0.4)
    return make(name)


def run(job):
    variant, stack, opp, deals = job
    from bot.selfplay import DECK, EngineAgent, play_hand
    hero, villain = EngineAgent(engine(variant, 1)), opponent(opp)
    hero.stack = villain.stack = stack              # engines inside agents must see the real effective stack
    rng = np.random.default_rng(7 + int(stack))
    out = np.zeros(deals)
    for k in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        out[k] = play_hand({"btn": hero, "bb": villain}, {"btn": c[:2], "bb": c[2:4]}, c[4:], stack=stack) \
            - play_hand({"btn": villain, "bb": hero}, {"btn": c[:2], "bb": c[2:4]}, c[4:], stack=stack)
    return job[:3], float(out.mean() / 2 * 100), float(out.std(ddof=1) / np.sqrt(deals) / 2 * 100)


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    stacks = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 20, 35]
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else VARIANTS
    opps = sys.argv[4].split(",") if len(sys.argv) > 4 else OPPONENTS
    jobs = [(v, s, o, deals) for v in variants for s in stacks for o in opps]
    with Pool(10) as pool:
        res = {key: (m, se) for key, m, se in pool.map(run, jobs)}
    print(f"bb/100 ({deals} duplicate deals each)")
    for s in stacks:
        print(f"-- {s:g}bb effective")
        print("   variant   " + " ".join(f"{o:>14}" for o in opps) + "     average")
        for v in variants:
            cells = [res[(v, s, o)] for o in opps]
            print(f"   {v:9} " + " ".join(f"{m:+7.0f} ±{se:4.0f}" for m, se in cells) + f"   {np.mean([m for m, _ in cells]):+7.1f}")
    OUT.write_text(json.dumps({"deals": deals, "results": {f"{v}|{s:g}|{o}": r for (v, s, o), r in res.items()}}, indent=1))


if __name__ == "__main__":
    main()
