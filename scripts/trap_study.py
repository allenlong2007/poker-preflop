"""Experiment 45: how often a solver slow-plays (traps) strong hands, by hand class, board texture and street.

Reads the 32 flops TexasSolver solved for experiment 22 (medium tree, flop bets 33/66/100%, sizes accurate to 0.1bb,
turn strategies saved; BTN 2.25bb open vs the BB's calling range, 97.75bb behind) and measures, for each strong hand
class (bot.engine.strong_class), weighted by how often each combo reaches the spot:
    flop  btn_check   the BTN checks back after the BB checks           (in-position trap)
          bb_flat     the BB just calls a c-bet instead of raising      (out-of-position flat)
    turn  bb_check    the BB checks instead of leading                   (check to check-raise / call)
          btn_check   the BTN checks back after the BB checks
The result (output/trap_study/trap_rates.csv) is what bot/engine.py's TRAP table is fitted to.

Usage: python scripts/trap_study.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOLVER_CHIP_SCALE", "10")
os.environ.setdefault("SOLVER_FLOP_DONK", "33")
os.environ.setdefault("SOLVER_FLOP_SIZES", "33,66,100")
os.environ.setdefault("SOLVER_DUMP_ROUNDS", "2")
sys.argv = sys.argv[:2]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval7
import numpy as np
import pandas as pd
import solver_compare as sc
from bot.engine import strong_class, trap_texture

OUT = ROOT / "output" / "trap_study"
OUT.mkdir(parents=True, exist_ok=True)


def node_probs(node):
    acts = node["strategy"]["actions"]
    return acts, {c: dict(zip(acts, p)) for c, p in node["strategy"]["strategy"].items()}


def analyse():
    d = np.load(sc.POLICIES)
    pre = {"btn": dict(zip(sc.HANDS, d["btn_range"])), "bb": dict(zip(sc.HANDS, d["bb_range"]))}
    rows = []

    def spot(flop, street, board, node, who, name, reach, trap_of):
        cards = [eval7.Card(x) for x in board]
        tex = trap_texture(cards)
        for c, p in node_probs(node)[1].items():
            k = strong_class([eval7.Card(c[:2]), eval7.Card(c[2:])], cards)
            w = pre[who][sc.class_of(c)] * reach.get(c, 1.0)
            if k and w > 0:
                rows.append({"flop": flop, "street": street, "texture": tex, "class": k, "spot": name,
                             "trap": trap_of(p), "w": w})

    flat = lambda p: p.get("CALL", 0.0) / max(p.get("CALL", 0.0) + sum(v for a, v in p.items() if a.startswith("RAISE")), 1e-9)
    for path in sorted(sc.RAW.glob("*.json")):
        tree = json.load(open(path))
        flop = " ".join(path.stem[i:i + 2] for i in (0, 2, 4))
        chk = tree["childrens"]["CHECK"]                                  # the BB checks the flop (no lead here)
        spot(flop, "flop", flop.split(), chk, "btn", "btn_check", {}, lambda p: p.get("CHECK", 0.0))
        btn_flop = node_probs(chk)[1]
        for a, child in chk["childrens"].items():
            if a.startswith("BET"):
                spot(flop, "flop", flop.split(), child, "bb", "bb_flat", {}, flat)
        # Turns after the flop checks through (the BTN reached it by checking back).
        cc = chk["childrens"]["CHECK"]
        reach_btn = {c: p.get("CHECK", 0.0) for c, p in btn_flop.items()}
        for card, turn in cc.get("dealcards", {}).items():
            board = flop.split() + [card]
            if "strategy" not in turn:
                continue
            spot(flop, "turn", board, turn, "bb", "bb_check", {}, lambda p: p.get("CHECK", 0.0))
            if "strategy" in turn["childrens"].get("CHECK", {}):
                spot(flop, "turn", board, turn["childrens"]["CHECK"], "btn", "btn_check", reach_btn,
                     lambda p: p.get("CHECK", 0.0))
        del tree
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "trap_rows.parquet") if False else None
    df["tw"] = df.trap * df.w
    g = lambda keys: df.groupby(keys)[["tw", "w"]].sum().assign(trap=lambda x: x.tw / x.w).trap.unstack(["street", "spot"])
    by_tex = g(["texture", "class", "street", "spot"]).round(3)
    by_tex.to_csv(OUT / "trap_rates.csv")
    g(["flop", "street", "spot"]).round(3).to_csv(OUT / "trap_by_flop.csv")
    pd.set_option("display.width", 200)
    print(by_tex.to_string())
    print(g(["class", "street", "spot"]).round(3).to_string())
    print(g(["texture", "street", "spot"]).round(3).to_string())


if __name__ == "__main__":
    analyse()
