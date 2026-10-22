"""Experiment 46: the bot against rival bots (bot/rivals.py), unadapted and after reading 30 / 200 of their deals.

For each rival: the bot's opponent model (profile, preflop adaptation, postflop exploit ratios, bluff model,
evidence strictness) is built from `history` deals of the rival's play, then a duplicate match is played.
history 0 = no read (the default strategy).

Usage: python scripts/rival_train.py [deals per match] [rival,rival,...]     (default 600, all rivals)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import strictness_train as stt
from bot.rivals import RIVALS, make as make_rival

HUMANS = ("aggro 0.4", "value-heavy", "maniac")
_human = stt.make


def make(name):
    return _human(name) if name in HUMANS else make_rival(name)


stt.make = make
OUT = ROOT / "output" / "match" / "rival_train.json"
HISTORY = tuple(int(x) for x in __import__("os").environ.get("HISTORY", "0,30,200").split(","))


def _tune():
    """TUNE="N_CONF=40,SIZE_SHRINK=20" overrides bot/adapt.py constants (for sweeps)."""
    import os
    import importlib
    for kv in filter(None, os.environ.get("TUNE", "").split(",")):
        k, v = kv.split("=")
        mod, k = k.split(".") if "." in k else ("adapt", k)
        setattr(importlib.import_module(f"bot.{mod}"), k, float(v))


def setup(job):
    _tune()
    name, hist = job
    if hist == 0:
        return job, None
    return stt.setup(job)


def play(job):
    (name, hist), ctx, deals = job
    _tune()
    from bot.engine import Engine
    from bot.selfplay import EngineAgent, match
    if ctx is None:
        r = match(EngineAgent(Engine(seed=1, pot_odds=True)), make(name), deals, seed=3)
        return (name, hist, "none"), float(r.mean() / 2 * 100), float(r.std(ddof=1) / np.sqrt(len(r)) / 2 * 100)
    else:
        return stt.play(((name, hist), "evidence", ctx, deals))
    return (name, hist, "none"), float(r.mean() / 2 * 100), float(r.std(ddof=1) / np.sqrt(len(r)) / 2 * 100)


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    rivals = sys.argv[2].split(",") if len(sys.argv) > 2 else RIVALS + HUMANS
    jobs = [(n, h) for n in rivals for h in HISTORY]
    with Pool(10) as pool:
        ctx = dict(pool.map(setup, jobs))
        res = pool.map(play, [(k, ctx[k], deals) for k in jobs])
    out = {}
    print(f"bb/100 vs rival bots ({deals} deals each, duplicate): no read | 30-deal read | 200-deal read")
    for n in rivals:
        cells = []
        for h in HISTORY:
            (_, _, _), m, se = next(x for x in res if x[0][0] == n and x[0][1] == h)
            out[f"{n}|{h}"] = (m, se)
            cells.append(f"{m:+7.0f} ± {se:3.0f}")
        print(f"   {n:10} " + " | ".join(cells), flush=True)
    for h in HISTORY:
        print(f"   average, history {h:3}: {np.mean([out[f'{n}|{h}'][0] for n in rivals]):+.1f}")
    import os
    if os.environ.get("TUNE"):
        return
    OUT.write_text(json.dumps({"deals": deals, "results": out}, indent=1))


if __name__ == "__main__":
    main()
