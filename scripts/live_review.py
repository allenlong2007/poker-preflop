"""Experiment 47: review every live hand the bot played (output/pokernow/hands): where the money went.

Prints the bot's own frequencies (as 'opp' from the profile's point of view, by flipping seats), net by how the
hand ended, and the biggest losses with the line. Usage: python scripts/live_review.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import eval7

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.opponent import load, profile


def flip(r):
    """The record from the opponent's side, so profile() describes the bot."""
    return {**r, "opp_pos": r["hero_pos"], "shown": {"opp": r["hero_hole"]} if r.get("hero_hole") else {},
            "net": {"hero": -r["net"]["hero"]}}


def main():
    hands = load()
    me = profile([flip(r) for r in hands])["nodes"]
    keys = ["btn_first_in", "bb_vs_open", "bb_vs_limp", "btn_vs_3bet", "flop_cbet", "flop_vs_bet", "turn_vs_bet",
            "river_vs_bet", "flop_lead", "turn_lead", "river_lead", "flop_bet_when_checked_to",
            "turn_bet_when_checked_to", "river_bet_when_checked_to", "size_small", "size_medium", "size_big"]
    print("The bot's live frequencies:")
    for k in keys:
        if k in me:
            print(f"  {k:28} {me[k]}")
    end = defaultdict(lambda: [0, 0.0])
    for r in hands:
        h = r["history"]
        last = max(i for i, s in enumerate(h) if s) if any(h) else 0
        acts = [a for s in h for a in s]
        fold = acts and acts[-1][1] == "fold"
        who = ("bot folds" if acts[-1][0] == r["hero_pos"] else "opp folds") if fold else "showdown/all-in"
        k = f"{['preflop', 'flop', 'turn', 'river'][last]:7} {who}"
        end[k][0] += 1
        end[k][1] += r["net"]["hero"]
    print("\nNet by where the hand ended (bb):")
    for k, (n, s) in sorted(end.items(), key=lambda x: x[1][1]):
        print(f"  {k:28} n={n:3}  net {s:+7.1f}  avg {s / n:+6.2f}")
    print("\nBiggest losses:")
    for r in sorted(hands, key=lambda r: r["net"]["hero"])[:25]:
        line = " / ".join(" ".join(f"{'B' if p == r['hero_pos'] else 'o'}:{k}{'' if to is None else f' {to:g}'}"
                                   for p, k, to in s) for s in r["history"] if s)
        print(f"  {r['net']['hero']:+7.1f}  {r['opp_name'][:8]:8} {r['hero_pos']:3} {r.get('hero_hole', ''):5} "
              f"vs {r['shown'].get('opp', '????'):5} board {r['board']:11} {r['stacks']['hero']:5.0f}bb  {line}")


if __name__ == "__main__":
    main()
