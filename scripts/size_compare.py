"""Experiment 38: the bot's turn / river bet sizes vs TexasSolver on experiment 34's spots.

Same spot as scripts/bb_turn_compare.py (K83, 1/3-pot c-bet called; turns 2s 7h As): range-weighted size mix
of the BB leading the turn, the BTN betting the turn after a check, and the BTN betting the river after the
turn checks through. Solver numbers: output/turn_river_check/solver_sizes.csv.

Usage:
    python scripts/size_compare.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bb_turn_compare import FLOP, OPEN, RIVERS, combos
from bot.engine import Act, Engine
from turn_river_check import RAW


def rng(turn, who):
    line = next(l for l in open(RAW / f"{turn}.txt") if l.startswith(f"set_range_{who}")).split(" ", 1)[1].strip()
    return {k: float(v) for k, v in (x.split(":") for x in line.split(","))}


def sizes(eng, seat, w, board, hist):
    dead = {board[i:i + 2] for i in range(0, len(board), 2)}
    acc, tot = {}, 0.0
    for cls, wt in w.items():
        for c in combos(cls, dead):
            d = eng.decide(seat, c, board, hist, sample=False)
            for k, v in d.probs.items():
                k = "check" if k == "check" else f"{float(k.split()[1]):.2f}"
                acc[k] = acc.get(k, 0) + wt * v
            tot += wt
    return {k: round(v / tot, 3) for k, v in sorted(acc.items())}


def main():
    eng = Engine(exact_required=False, seed=0)
    if len(sys.argv) > 2:
        eng.river_shift = (float(sys.argv[1]), float(sys.argv[2]))
    c = round(OPEN * 2 / 3, 4)
    pre = [Act("btn", "raise", OPEN), Act("bb", "call")]
    flop = [Act("bb", "check"), Act("btn", "bet", c), Act("bb", "call")]
    for turn in ("2s", "7h", "As"):
        bb, btn = rng(turn, "oop"), rng(turn, "ip")
        b = FLOP + turn
        print(f"turn {turn}: BB first     ", sizes(eng, "bb", bb, b, [pre, flop, []]))
        print(f"        BTN after check", sizes(eng, "btn", btn, b, [pre, flop, [Act("bb", "check")]]))
        riv = [sizes(eng, "btn", btn, b + r, [pre, flop, [Act("bb", "check"), Act("btn", "check")], [Act("bb", "check")]])
               for r in RIVERS]
        keys = sorted({k for x in riv for k in x})
        print(f"        river BTN after x-x, x", {k: round(sum(x.get(k, 0) for x in riv) / len(riv), 3) for k in keys})


if __name__ == "__main__":
    main()
