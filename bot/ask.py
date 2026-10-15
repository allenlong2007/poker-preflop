"""Ask the match bot what to do in one spot.

    python -m bot.ask <seat> <hole> [board] [actions]

seat btn or bb; hole like AhKd; board like Kh8d3c (or "" preflop). Actions are space-separated tokens,
streets separated by "/": f fold, x check, c call, b<to> bet, r<to> raise (amounts in bb, the player's
total on that street; limp = c). Players alternate: the BTN acts first preflop, the BB first after.

    python -m bot.ask bb QsJs "" "r2.5"
    python -m bot.ask btn AcQd Kh8d3c "r3 c / x"
    python -m bot.ask bb Ts9s Qh8s4s "r3 c / x b4"
"""
import sys

from .engine import Act, Engine

KINDS = {"f": "fold", "x": "check", "c": "call", "b": "bet", "r": "raise"}


def parse_actions(text):
    hist = []
    for s, street in enumerate((text or "").split("/")):
        player = "btn" if s == 0 else "bb"
        acts = []
        for tok in street.split():
            kind = KINDS[tok[0]]
            acts.append(Act(player, kind, float(tok[1:]) if kind in ("bet", "raise") else 0.0))
            player = "bb" if player == "btn" else "btn"
        hist.append(acts)
    return hist or [[]]


def main():
    seat, hole = sys.argv[1], sys.argv[2]
    board = sys.argv[3] if len(sys.argv) > 3 else ""
    hist = parse_actions(sys.argv[4] if len(sys.argv) > 4 else "")
    d = Engine().decide(seat, hole, board, hist)
    print(f"{d.node} [{d.source}] {'; '.join(d.notes)}")
    for label, p in sorted(d.probs.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<12} {p:6.1%}")
    print(f"-> {d.action}{f' to {d.to:g}bb' if d.action in ('bet', 'raise', 'call') else ''}")


if __name__ == "__main__":
    main()
