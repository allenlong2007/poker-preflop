"""Experiment 35b: the bot's BB turn/river play vs TexasSolver on the same spot (experiment 34's solves).

Spot: K♥8♦3♣, BTN opens, BB calls, BB checks, BTN bets 1/3 pot, BB calls; turn 2♠ / 7♥ / A♠. For every combo
in the solve's BB range (weights from the solve's own config), the engine's decision is averaged, at the BB's
first turn action, facing each turn bet size, and leading the river after the turn checks through.

Usage:
    python scripts/bb_turn_compare.py [turn keep, river keep, river lead]   (default: the engine's settings)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bot.engine import Act, Engine
from turn_river_check import RAW

FLOP = "Kh8d3c"
OPEN = 2.25
POT = 2 * OPEN + 2 * (OPEN * 2 / 3)          # pot after a 1/3-pot c-bet is called
RIVERS = ["4h", "Td", "Qc", "5s", "9d", "Jh"]


def bb_range(turn):
    line = next(l for l in open(RAW / f"{turn}.txt") if l.startswith("set_range_oop")).split(" ", 1)[1].strip()
    return {k: float(v) for k, v in (x.split(":") for x in line.split(","))}


def combos(cls, dead):
    r = "23456789TJQKA"
    s = "cdhs"
    a, b = cls[0], cls[1]
    if len(cls) == 2:
        out = [a + x + b + y for i, x in enumerate(s) for y in s[i + 1:]]
    elif cls[2] == "s":
        out = [a + x + b + x for x in s]
    else:
        out = [a + x + b + y for x in s for y in s if x != y]
    return [c for c in out if c[:2] not in dead and c[2:] not in dead]


def averaged(eng, rng, board, hist):
    dead = {board[i:i + 2] for i in range(0, len(board), 2)}
    acc, wsum = {}, 0.0
    for cls, w in rng.items():
        for c in combos(cls, dead):
            d = eng.decide("bb", c, board, hist, sample=False)
            for k, v in d.probs.items():
                k = k.split()[0] if k.split()[0] in ("fold", "call", "check") else ("bet" if "bet" in k or "lead" in k else "raise")
                acc[k] = acc.get(k, 0) + w * v
            wsum += w
    return {k: round(v / wsum, 3) for k, v in sorted(acc.items())}


def main():
    keep_t, keep_r, lead = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (None, None, None)
    po = os.environ.get("POT_ODDS") == "1"
    eng = Engine(seed=0, pot_odds=po) if keep_t is None else Engine(seed=0, defend_keep={1: keep_t, 2: keep_r}, river_lead=lead, pot_odds=po)
    c = round(OPEN * 2 / 3, 4)
    pre = [Act("btn", "raise", OPEN), Act("bb", "call")]
    flop = [Act("bb", "check"), Act("btn", "bet", c), Act("bb", "call")]
    for turn in ("2s", "7h", "As"):
        rng = bb_range(turn)
        board = FLOP + turn
        print(f"turn {turn}: BB first:", averaged(eng, rng, board, [pre, flop, []]))
        for frac in (0.33, 0.5, 0.75, 1.0, 1.5):
            b = round(frac * POT, 2)
            print(f"   BB vs turn bet {frac:.2f} pot:", averaged(eng, rng, board, [pre, flop, [Act("bb", "check"), Act("btn", "bet", b)]]),
                  f"| max fold {b / (POT + b):.3f}")
        leads = [averaged(eng, rng, board + rv, [pre, flop, [Act("bb", "check"), Act("btn", "check")], []]) for rv in RIVERS]
        print("   BB first on river after turn x-x:", {k: round(sum(l.get(k, 0) for l in leads) / len(leads), 3) for k in ("check", "bet")})
        xx = [Act("bb", "check"), Act("btn", "check")]
        for frac in (0.33, 0.5, 0.75, 1.0, 1.5, 2.0):
            b = round(frac * POT, 2)
            rs = [averaged(eng, rng, board + rv, [pre, flop, xx, [Act("bb", "check"), Act("btn", "bet", b)]]) for rv in RIVERS]
            print(f"   BB vs river bet {frac:.2f} pot (turn x-x):",
                  {k: round(sum(r.get(k, 0) for r in rs) / len(rs), 3) for k in ("fold", "call", "raise")},
                  f"| max fold {b / (POT + b):.3f}")


if __name__ == "__main__":
    main()
