"""Experiment 34: does the bot bet pot-sized too often on the turn and river? TexasSolver vs the bot.

The bot's turn and river play comes from the whole-hand model (12 hand groups, board-texture classes;
experiment 30's sizes). This solves the turn and river exactly with TexasSolver, card by card, after a common
line, with the same bet sizes the bot has, and compares how often each size is used:

    flop  K♥8♦3♣ in the match's 3bb single-raised pot: BB checks, BTN bets 33%, BB calls
    turn  a few turn cards; ranges are the flop solve's (experiment 27), conditioned on the line
    sizes turn 33 / 50 / 75 / 100 / 150%, river 33 / 50 / 75 / 100 / 150 / 200% (leads too), raises 50% of pot

Usage:
    python scripts/turn_river_check.py solve [turn cards]     (default 2s As 7h; a few minutes each)
    python scripts/turn_river_check.py report
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preflop.cards import HANDS
from solver_compare import SOLVER_DIR, class_of, range_string

FLOP = "Kh 8d 3c"
OUT = ROOT / "output" / "turn_river_check"
RAW = OUT / "raw"                              # TexasSolver dumps (gitignored: large)
RAW.mkdir(parents=True, exist_ok=True)
SCALE = 10                                     # chips per bb, so sizes aren't rounded
POT, STACK = 10.0, 95.0                        # after 3bb open called + 2bb bet called
TURN_SIZES = (33, 50, 75, 100, 150)
RIVER_SIZES = (33, 50, 75, 100, 150, 200)
TURNS = ["2s", "As", "7h"]


def line_ranges():
    """Class weights for each player after: BB checks, BTN bets 2bb (33%), BB calls."""
    pol = json.load(open(ROOT / "output" / "match" / "flops" / "Kh8d3c.json"))["nodes"]
    base = np.load(ROOT / "output" / "match" / "flops" / "srp_ranges.npz")
    idx = {h: i for i, h in enumerate(HANDS)}

    def by_class(rows, key, facing=None):
        acc, n = np.zeros(169), np.zeros(169)
        for r in rows:
            if facing is not None and r.get("facing") != facing:
                continue
            i = idx[class_of(r["combo"])]
            acc[i] += r.get(key, 0.0)
            n[i] += 1
        return np.where(n > 0, acc / np.maximum(n, 1), 0.0)

    btn = base["btn_range"] * by_class(pol["btn_vs_check"], "bet 2bb")
    bb = base["bb_range"] * by_class(pol["bb_first"], "check") * by_class(pol["bb_vs_bet"], "call", "bet 2bb")
    return btn, bb


def config(turn, out_json):
    btn, bb = line_ranges()
    board = ",".join(FLOP.split() + [turn])
    lines = [f"set_pot {POT * SCALE:g}", f"set_effective_stack {STACK * SCALE:g}", f"set_board {board}",
             f"set_range_oop {range_string(bb)}", f"set_range_ip {range_string(btn)}"]
    for street, sizes in (("turn", TURN_SIZES), ("river", RIVER_SIZES)):
        sz = ",".join(map(str, sizes))
        for who in ("oop", "ip"):
            lines.append(f"set_bet_sizes {who},{street},bet,{sz}")
            lines.append(f"set_bet_sizes {who},{street},raise,50")
        lines.append(f"set_bet_sizes oop,{street},donk,{sz}")
    lines += ["set_allin_threshold 1.0", "set_raise_limit 1", "build_tree", "set_thread_num 10",
              "set_accuracy 0.3", "set_max_iteration 200", "set_print_interval 10", "set_use_isomorphism 1",
              "start_solve", "set_dump_rounds 2", f"dump_result {out_json}"]
    return "\n".join(lines) + "\n"


def solve(turns):
    for turn in turns:
        out = RAW / f"{turn}.json"
        if out.exists():
            print(f"{turn}: already solved")
            continue
        cfg = RAW / f"{turn}.txt"
        cfg.write_text(config(turn, out))
        t0 = time.time()
        with open(RAW / f"{turn}.log", "w") as logf:
            res = subprocess.run([str(SOLVER_DIR / "console_solver"), "-i", str(cfg), "-r", "resources"],
                                 cwd=SOLVER_DIR, stdout=logf, stderr=subprocess.STDOUT)
        log = (RAW / f"{turn}.log").read_text(errors="replace").replace("\r", "\n")
        expl = [l for l in log.splitlines() if l.startswith("Total exploitability")]
        print(f"{turn}: {time.time() - t0:.0f}s, exit {res.returncode}, {expl[-1] if expl else 'no result'}", flush=True)


def size_name(action, pot):
    """'BET 50.000000' -> pot fraction label, given the pot (in chips) at that node."""
    amt = float(action.split()[1])
    return f"{amt / pot:.2f}"


def mix(node, weights, pot):
    """Range-weighted action mix at a decision node; weights: class -> weight for the acting player."""
    acts = node["strategy"]["actions"]
    tot, wsum = np.zeros(len(acts)), 0.0
    for combo, probs in node["strategy"]["strategy"].items():
        w = weights.get(class_of(combo), 0.0)
        tot += w * np.array(probs)
        wsum += w
    out = {}
    for a, v in zip(acts, tot / max(wsum, 1e-12)):
        key = a.split()[0].lower() if a in ("CHECK", "CALL", "FOLD") else f"{a.split()[0].lower()} {size_name(a, pot)}"
        out[key] = out.get(key, 0) + v
    return out


def report():
    btn, bb = line_ranges()
    wb, wo = dict(zip(HANDS, btn)), dict(zip(HANDS, bb))
    rows = []
    pot = POT * SCALE
    for f in sorted(RAW.glob("*.json")):
        d = json.load(open(f))
        turn = f.stem
        # Turn: BB first, then the BTN after a check.
        rows.append({"turn": turn, "node": "turn: BB first", **mix(d, wo, pot)})
        if "CHECK" in d["childrens"]:
            rows.append({"turn": turn, "node": "turn: BTN after check", **mix(d["childrens"]["CHECK"], wb, pot)})
            # River after check-check: average the BTN's river betting (after BB checks) over river cards.
            xx = d["childrens"]["CHECK"]["childrens"].get("CHECK")
            if xx and "dealcards" in xx:
                acc, n = {}, 0
                for card, rn in xx["dealcards"].items():
                    node = rn.get("childrens", {}).get("CHECK")
                    if not node or "strategy" not in node:       # some branches carry no stored strategy
                        continue
                    n += 1
                    for k, v in mix(node, wb, pot).items():
                        acc[k] = acc.get(k, 0) + v
                acc = {k: v / max(n, 1) for k, v in acc.items()}
                rows.append({"turn": turn, "node": "river: BTN after x-x, x", **acc})
    df = pd.DataFrame(rows).fillna(0)
    cols = ["turn", "node"] + sorted([c for c in df.columns if c not in ("turn", "node")],
                                     key=lambda c: (c.split()[0] != "check", float(c.split()[1]) if " " in c else 0))
    df = df[cols]
    df.round(3).to_csv(OUT / "solver_sizes.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "solve":
        solve(sys.argv[2:] or TURNS)
    else:
        report()


if __name__ == "__main__":
    main()
