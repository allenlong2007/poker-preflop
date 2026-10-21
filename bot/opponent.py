"""Opponent tracking: every hand the bot plays is stored, and the opponent's tendencies are counted from them.

    store   output/pokernow/hands/<opponent id>.jsonl, one line per finished hand (deduplicated by hand id):
            positions, every action (in big blinds), the board, the bot's cards, any cards shown, stacks
    stats   per-node frequencies for the opponent (how often they fold / call / raise at each preflop node,
            c-bet, fold to bets, lead, raise postflop), with sample sizes, and the hands they showed down

Usage:
    python -m bot.opponent import <log.json> [hero name]     (store every finished hand of a downloaded log)
    python -m bot.opponent stats [opponent id]               (print the profile)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bot.pokernow.parse import RE_START, cards, parse_hand

ROOT = Path(__file__).resolve().parents[1]
HANDS = ROOT / "output" / "pokernow" / "hands"
RE_ID = re.compile(r"-- starting hand #\d+ \(id: (?P<id>\w+)\)")
RE_SHOW = re.compile(r'"(?P<name>.+?) @ (?P<pid>[\w-]+)" shows a (?P<cards>.+?)\.?$')
HERO = "HUBot"


def split_hands(messages):
    """Messages in time order -> one list per hand (only finished hands)."""
    out, cur = [], None
    for m in messages:
        if RE_START.search(m):
            cur = [m]
        elif cur is not None:
            cur.append(m)
            if m.startswith("-- ending hand"):
                out.append(cur)
                cur = None
    return out


def record(msgs, hero_name=HERO):
    """One finished hand -> a stored record (None if the bot wasn't in it or it isn't heads-up)."""
    h = parse_hand(msgs, hero_name=hero_name)
    if h.hero_seat is None or len(h.seats) != 2:
        return None
    hero = h.seats[h.hero_seat]
    opp = next(s for s in h.seats.values() if s.seat != h.hero_seat)
    shown = {}
    for m in msgs:
        if (sm := RE_SHOW.search(m)):
            shown["hero" if sm.group("name") == hero.name else "opp"] = "".join(cards(sm.group("cards")))
    hid = RE_ID.search(msgs[0])
    return {"id": hid.group("id") if hid else str(h.number), "number": h.number,
            "opp_name": opp.name, "opp_id": opp.pid, "hero_pos": h.position(hero), "opp_pos": h.position(opp),
            "bb": h.bb, "stacks": {"hero": round(hero.start / h.bb, 2), "opp": round(opp.start / h.bb, 2)},
            "history": [[[a.player, a.kind, a.to] for a in street] for street in h.history],
            "board": "".join(h.board), "hero_hole": "".join(h.hole), "shown": shown,
            "net": {"hero": round((hero_net(msgs, hero.name) - hero.put_in) / h.bb, 2)}}


def hero_net(msgs, name):
    """Chips the player collected in the hand (pot won plus uncalled bets returned)."""
    got = 0.0
    for m in msgs:
        if m.startswith(f'"{name} @') and " collected " in m:
            got += float(re.search(r"collected ([\d.]+)", m).group(1))
        if m.startswith("Uncalled bet of") and f'"{name} @' in m:
            got += float(re.search(r"Uncalled bet of ([\d.]+)", m).group(1))
    return got


def store(records):
    """Append new records to each opponent's file; returns how many were new."""
    HANDS.mkdir(parents=True, exist_ok=True)
    new = 0
    by_opp = defaultdict(list)
    for r in records:
        if r:
            by_opp[r["opp_id"]].append(r)
    for opp, rs in by_opp.items():
        path = HANDS / f"{opp}.jsonl"
        seen = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        with open(path, "a") as f:
            for r in rs:
                if r["id"] not in seen:
                    f.write(json.dumps(r) + "\n")
                    seen.add(r["id"])
                    new += 1
    return new


def load(opp_id=None):
    """All stored hands against one opponent (or everyone, if opp_id is None)."""
    files = [HANDS / f"{opp_id}.jsonl"] if opp_id else sorted(HANDS.glob("*.jsonl"))
    return [json.loads(l) for f in files if f.exists() for l in open(f)]


class Count:
    """Action counts at one node: freq(kind) with the sample size, for blending toward a prior."""
    def __init__(self):
        self.c = defaultdict(int)
        self.n = 0

    def add(self, kind):
        self.c[kind] += 1
        self.n += 1

    def freq(self, kind):
        return self.c[kind] / self.n if self.n else None

    def as_dict(self):
        return {"n": self.n, **{k: round(v / self.n, 3) for k, v in sorted(self.c.items())}}


def _kind(a, facing):
    """Name an opponent preflop action: fold / call (limp) / raise / all-in."""
    p, k, to = a
    if k in ("fold", "check", "call"):
        return k
    return "all-in" if to is not None and to >= 0.6 * facing else "raise"


def profile(hands):
    """The opponent's frequencies at each node, from their point of view (they are 'opp')."""
    st = defaultdict(Count)
    shows = []
    for r in hands:
        opp, pre = r["opp_pos"], r["history"][0]
        eff = min(r["stacks"].values())
        acts = [tuple(a) for a in pre]
        # Preflop nodes, named from the opponent's seat.
        for i, a in enumerate(acts):
            if a[0] != opp:
                continue
            prev = [(p, k) for p, k, _ in acts[:i]]
            if opp == "btn" and not prev:
                node = "btn_first_in"
            elif opp == "bb" and prev == [("btn", "call")]:
                node = "bb_vs_limp"
            elif opp == "bb" and prev == [("btn", "raise")]:
                node = "bb_vs_open"
            elif opp == "btn" and prev == [("btn", "call"), ("bb", "raise")]:
                node = "btn_vs_iso"
            elif opp == "btn" and prev == [("btn", "raise"), ("bb", "raise")]:
                node = "btn_vs_3bet"
            elif opp == "bb" and prev == [("btn", "raise"), ("bb", "raise"), ("btn", "raise")]:
                node = "bb_vs_4bet"
            else:
                node = "preflop_other"
            k = _kind(a, eff)
            st[node].add("limp" if node == "btn_first_in" and k == "call" else k)
            if node == "btn_first_in" and k == "raise":
                st["open_size"].add(f"{a[2]:g}")
        # Postflop: c-bets, folds to bets, leads, raises.
        raiser = next((p for p, k, _ in reversed(acts) if k == "raise"), None)
        for s, street in enumerate(r["history"][1:], start=1):
            name = ["flop", "turn", "river"][s - 1]
            for i, (p, k, to) in enumerate(street):
                if p != opp:
                    continue
                before = street[:i]
                facing_bet = any(kk in ("bet", "raise") for _, kk, _ in before)
                if facing_bet:
                    st[f"{name}_vs_bet"].add({"fold": "fold", "call": "call"}.get(k, "raise"))
                elif s == 1 and raiser == opp:
                    st["flop_cbet"].add("bet" if k == "bet" else "check")
                elif opp == "bb" and not before:
                    st[f"{name}_lead"].add("bet" if k == "bet" else "check")
                else:
                    st[f"{name}_bet_when_checked_to"].add("bet" if k == "bet" else "check")
        if "opp" in r["shown"]:
            shows.append({"hand": r["number"], "cards": r["shown"]["opp"], "pos": opp,
                          "line": " / ".join(" ".join(f"{p}:{k}{'' if to is None else f' {to:g}'}"
                                                       for p, k, to in street) for street in r["history"]),
                          "board": r["board"]})
    return {"hands": len(hands), "nodes": {k: v.as_dict() for k, v in sorted(st.items())}, "shown": shows}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "import":
        log = json.load(open(sys.argv[2]))
        msgs = [x["msg"] for x in log]
        if len(log) > 1 and log[0]["created_at"] > log[-1]["created_at"]:
            msgs = msgs[::-1]                                        # the JSON log is newest first
        hero = sys.argv[3] if len(sys.argv) > 3 else HERO
        print(f"stored {store(record(m, hero) for m in split_hands(msgs))} new hands")
    else:
        print(json.dumps(profile(load(sys.argv[2] if len(sys.argv) > 2 else None)), indent=1))


if __name__ == "__main__":
    main()
