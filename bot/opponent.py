"""Opponent tracking: every hand the bot plays is stored, and the opponent's tendencies are counted from them.

    store   output/pokernow/hands/<opponent id>.jsonl, one line per finished hand (deduplicated by hand id):
            positions, every action (in big blinds), the board, the bot's cards, any cards shown, stacks
    stats   per-node frequencies for the opponent (how often they fold / call / raise at each preflop node,
            c-bet, fold to bets, lead, raise postflop), with sample sizes, and the hands they showed down

Usage:
    python -m bot.opponent import <log.json> [hero name]     (store every finished hand of a downloaded log)
    python -m bot.opponent stats [opponent id]               (print the profile)
    python -m bot.opponent compare <id> <other id>           (are two ids likely one person? evidence only)
    python -m bot.opponent link <main id> <other id> ...    (one person under several PokerNow ids; every new
                                                             id starts from zero: link only on overwhelming
                                                             evidence, confirmed by the user)
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
        path = _path(opp)
        seen = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        with open(path, "a") as f:
            for r in rs:
                if r["id"] not in seen:
                    f.write(json.dumps(r) + "\n")
                    seen.add(r["id"])
                    new += 1
    return new


SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _path(opp_id):
    """The hand file for a player id. Ids come from the game log, so only plain ids may become file names."""
    if not SAFE_ID.fullmatch(str(opp_id)):
        raise ValueError(f"unsafe player id {opp_id!r}")
    return HANDS / f"{opp_id}.jsonl"


def aliases():
    """{player id: canonical id}: one person seen under several PokerNow ids (another browser, cleared cookies)."""
    path = HANDS / "aliases.json"
    return json.load(open(path)) if path.exists() else {}


def canonical(opp_id):
    return aliases().get(opp_id, opp_id)


def link(*ids):
    """Record that these player ids are one person (the first is the canonical id)."""
    HANDS.mkdir(parents=True, exist_ok=True)
    a = aliases()
    for i in ids:
        _path(i)                                            # validates the id
        a[i] = ids[0]
    (HANDS / "aliases.json").write_text(json.dumps(a, indent=1))


def load(opp_id=None):
    """All stored hands against one opponent, under any of their linked ids (or everyone, if opp_id is None)."""
    if opp_id:
        c = canonical(opp_id)
        ids = {c} | {k for k, v in aliases().items() if v == c}
        files = [_path(i) for i in sorted(ids)]
    else:
        files = sorted(HANDS.glob("*.jsonl"))
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
        _size_bands(r, opp, st)
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
            _traps(r, opp, st)
            shows.append({"hand": r["number"], "cards": r["shown"]["opp"], "pos": opp,
                          "line": " / ".join(" ".join(f"{p}:{k}{'' if to is None else f' {to:g}'}"
                                                       for p, k, to in street) for street in r["history"]),
                          "board": r["board"]})
    return {"hands": len(hands), "nodes": {k: v.as_dict() for k, v in sorted(st.items())}, "shown": shows,
            "results": results(hands)}


def _traps(r, opp, st):
    """Experiment 45: what the opponent did on each street where their shown hand was strong (two pair or better,
    bot.engine.strong_class): slowplay (only checked / called), checkraise, or fast (bet or raised first). Only shown
    hands count, so fast play that won without a showdown is missing: read it as an upper bound on trapping."""
    import eval7
    from bot.engine import strong_class, trap_texture
    hole = [eval7.Card(r["shown"]["opp"][i:i + 2]) for i in (0, 2)]
    full = [eval7.Card(r["board"][i:i + 2]) for i in range(0, len(r["board"]), 2)]
    for s, street in enumerate(r["history"][1:], start=1):
        board = full[:2 + s]
        mine = [k for p, k, _ in street if p == opp]
        if len(board) < 3 or not mine or not strong_class(hole, board):
            continue
        if mine[0] in ("bet", "raise"):
            kind = "fast"
        elif any(k == "raise" for k in mine[1:]):
            kind = "checkraise"
        else:
            kind = "slowplay"
        st["trap"].add(kind)
        st[f"trap_{trap_texture(board)}"].add(kind)


def read_style(prof):
    """Experiment 46: a plain-language read of the opponent: style, key frequencies, and whether they play like a
    simple bot (few distinct bet sizes, pure push/fold, min-raises). Labels need MIN_READ hands; it reports what it
    has either way."""
    n = prof.get("nodes", {})
    f = lambda node, k: n.get(node, {}).get(k)
    cnt = lambda node: n.get(node, {}).get("n", 0)
    hands = prof.get("hands", 0)

    def pooled(nodes, kinds):
        tot = sum(cnt(x) for x in nodes)
        return (sum(cnt(x) * sum(n[x].get(k, 0.0) for k in kinds) for x in nodes if x in n) / tot) if tot else None

    vpip_btn = None if not cnt("btn_first_in") else 1 - f("btn_first_in", "fold") if f("btn_first_in", "fold") is not None else 1.0
    defend = None if not cnt("bb_vs_open") else 1 - (f("bb_vs_open", "fold") or 0.0)
    shove = pooled(["btn_first_in", "bb_vs_open", "bb_vs_limp", "btn_vs_3bet"], ["all-in"])
    limp = f("btn_first_in", "limp") if cnt("btn_first_in") else None
    post = ["flop", "turn", "river"]
    bet = pooled([f"{s}_{k}" for s in post for k in ("cbet", "lead", "bet_when_checked_to")], ["bet"])
    fold = pooled([f"{s}_vs_bet" for s in post], ["fold"])
    raise_ = pooled([f"{s}_vs_bet" for s in post], ["raise"])
    sizes = n.get("bet_size", {})
    nb = sizes.get("n", 0)
    top_sizes = sorted(((k, v) for k, v in sizes.items() if k != "n"), key=lambda x: -x[1])[:3]
    fixed = nb >= 10 and (top_sizes[0][1] >= 0.75 or sum(v for _, v in top_sizes[:2]) >= 0.9)
    big = sum(v for k, v in sizes.items() if k != "n" and float(k) >= 1.5) if nb else 0.0
    tiny = sum(v for k, v in sizes.items() if k != "n" and float(k) <= 0.2) if nb else 0.0
    post_n = sum(cnt(f"{s}_vs_bet") for s in post)
    labels = []
    if hands >= MIN_READ:
        if shove is not None and shove > 0.25 and (bet or 0) < 0.15:
            labels.append("push/fold")
        if (vpip_btn or 0) < 0.4 and (defend or 1) < 0.5:
            labels.append("nit")
        elif (vpip_btn or 0) < 0.55 and (defend or 1) < 0.45:
            labels.append("tight")
        if (limp or 0) > 0.5:
            labels.append("limper")
        if (bet or 0) > 0.7 and (raise_ or 0) > 0.3 and (fold or 0) < 0.2:
            labels.append("maniac")
        elif (bet or 0) > 0.55 or (raise_ or 0) > 0.2:
            labels.append("aggressive")
        if fold is not None and post_n >= 10 and fold < 0.15:
            labels.append("calling station")
        elif fold is not None and post_n >= 10 and fold > 0.6:
            labels.append("folds too much")
        if nb >= 8 and big > 0.4:
            labels.append("overbettor")
        if nb >= 8 and tiny > 0.4:
            labels.append("min-bettor")
        if not labels:
            labels.append("balanced")
    bot_like = []
    if fixed:
        bot_like.append(f"fixed bet sizes ({', '.join(k for k, v in top_sizes[:2] if v >= 0.1)} pot)")
    if "push/fold" in labels:
        bot_like.append("pure push/fold")
    if nb >= 8 and tiny > 0.4:
        bot_like.append("min-bets")
    if limp == 0 and cnt("btn_first_in") >= 20 and (vpip_btn or 0) > 0 and len(n.get("open_size", {})) == 2:
        bot_like.append("one open size, never limps")
    r = lambda x: None if x is None else round(x, 2)
    return {"hands": hands, "style": labels or ["not enough hands"], "bot_like": bot_like,
            "stats": {"btn_plays": r(vpip_btn), "bb_defends": r(defend), "shoves": r(shove), "limps": r(limp),
                      "postflop_bets": r(bet), "folds_to_bets": r(fold), "raises_bets": r(raise_),
                      "top_bet_sizes": [(k, round(v, 2)) for k, v in top_sizes]}}


MIN_READ = 20


def results(hands):
    """How the bot is doing against these hands: hands won / lost / tied (by the bot's net), win rate, net bb and
    bb/100, and showdowns won."""
    nets = [r["net"]["hero"] for r in hands if "net" in r]
    sd = [r for r in hands if "opp" in r["shown"] and "net" in r]
    n = len(nets)
    return {"hands": n, "won": sum(x > 0 for x in nets), "lost": sum(x < 0 for x in nets),
            "tied": sum(x == 0 for x in nets), "win_rate": round(sum(x > 0 for x in nets) / n, 3) if n else None,
            "net_bb": round(sum(nets), 1), "bb_per_100": round(100 * sum(nets) / n, 1) if n else None,
            "showdowns": len(sd), "showdowns_won": sum(r["net"]["hero"] > 0 for r in sd)}


def scoreboard():
    """Bot results per opponent ID (and overall), most hands first."""
    by = defaultdict(list)
    for r in load():
        by[r["opp_id"]].append(r)
    rows = sorted(((canonical(k), v[-1]["opp_name"], results(v)) for k, v in by.items()), key=lambda x: -x[2]["hands"])
    return rows + [("ALL", "", results([r for _, v in by.items() for r in v]))]


def compare(a, b, nodes=("btn_first_in", "bb_vs_open", "bb_vs_limp", "btn_vs_3bet", "flop_vs_bet", "flop_cbet",
                           "open_size")):
    """How alike two ids play: per node, the largest gap in an action's frequency and its z-score (two-proportion
    test). Many nodes with |z| < 1 and decent samples is consistent with one person; any |z| > 3 says two people.
    It is evidence, never proof: ids are linked only by hand (`link`)."""
    pa, pb = profile(load(a))["nodes"], profile(load(b))["nodes"]
    out, worst = {}, 0.0
    for n in nodes:
        x, y = pa.get(n), pb.get(n)
        if not x or not y:
            continue
        acts = {k for k in x if k != "n"} | {k for k in y if k != "n"}
        zs = {}
        for k in acts:
            p1, p2 = x.get(k, 0.0), y.get(k, 0.0)
            p = (p1 * x["n"] + p2 * y["n"]) / (x["n"] + y["n"])
            se = (p * (1 - p) * (1 / x["n"] + 1 / y["n"])) ** 0.5
            zs[k] = round((p1 - p2) / se, 2) if se > 0 else 0.0
        k = max(zs, key=lambda k: abs(zs[k]))
        out[n] = {"n": (x["n"], y["n"]), "biggest gap": k, "a": x.get(k, 0), "b": y.get(k, 0), "z": zs[k]}
        worst = max(worst, abs(zs[k]))
    verdict = ("different players" if worst > 3 else
               "not distinguishable (not proof of one player)" if worst < 1.5 else "inconclusive")
    return {"nodes": out, "max |z|": round(worst, 2), "verdict": verdict}


SIZE_BANDS = ((0.0, 0.45, "small"), (0.45, 0.85, "medium"), (0.85, 99.0, "big"))


def size_band(frac):
    """Total: any number (negative, huge, NaN) lands in a band; this runs in the live decision path."""
    frac = 0.0 if frac != frac else min(max(float(frac), 0.0), 98.999)
    return next((name for lo, hi, name in SIZE_BANDS if lo <= frac < hi), "big")


def _size_bands(r, opp, st):
    """Postflop bets / raises by size (share of the pot): how often, and at showdown how often a bluff (less than
    a pair beyond the board's own)."""
    import eval7
    pot = sum(_put(r["history"][0]).values()) or 1.5
    shown = r.get("shown", {}).get("opp")
    board = r.get("board", "")
    cards = [eval7.Card(board[i:i + 2]) for i in range(0, len(board), 2)] if board else []
    hole = [eval7.Card(shown[:2]), eval7.Card(shown[2:4])] if shown else None
    for s, street in enumerate(r["history"][1:], start=1):
        put = {"btn": 0.0, "bb": 0.0}
        for p, k, to in street:
            if k in ("bet", "raise") and p == opp and to:
                frac = (to - put[opp]) / max(pot + sum(put.values()), 1e-9) if k == "raise" else to / pot
                band = size_band(frac)
                st[f"size_{band}"].add(["", "flop", "turn", "river"][s])
                st["bet_size"].add(f"{round(frac * 20) / 20:.2f}")        # 5%-of-pot buckets (experiment 46)
                if hole and len(cards) == 5:
                    b = cards[:2 + s]
                    made = eval7.evaluate(hole + b) >> 24
                    bluff = made <= (eval7.evaluate(b) >> 24) and not _pair_above(hole, b)
                    st[f"shown_{band}"].add("bluff" if bluff else "value")
            if k in ("bet", "raise") and to:
                put[p] = to
            elif k == "call":
                put[p] = max(put.values())
        pot += sum(put.values())


def _pair_above(hole, board):
    return any(c.rank == b.rank for c in hole for b in board)


def _put(pre):
    put = {"btn": 0.5, "bb": 1.0}
    for p, k, to in pre:
        if k == "raise" and to:
            put[p] = to
        elif k == "call":
            put[p] = max(put.values())
    return put


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "compare":
        print(json.dumps(compare(sys.argv[2], sys.argv[3]), indent=1))
        return
    if cmd == "link":
        link(*sys.argv[2:])
        print(f"linked {sys.argv[3:]} -> {sys.argv[2]}")
        return
    if cmd == "read":
        print(json.dumps(read_style(profile(load(sys.argv[2] if len(sys.argv) > 2 else None))), indent=1))
        return
    if cmd == "results":
        print(f"{'opponent':24} {'hands':>5} {'won':>4} {'lost':>4} {'win%':>5} {'net bb':>7} {'bb/100':>7} {'SD won':>7}")
        for pid, name, x in scoreboard():
            wr = "-" if x["win_rate"] is None else f"{100 * x['win_rate']:.0f}"
            print(f"{(name + ' ' + pid)[:24]:24} {x['hands']:5} {x['won']:4} {x['lost']:4} {wr:>5} {x['net_bb']:7} "
                  f"{x['bb_per_100'] or 0:7} {x['showdowns_won']:3}/{x['showdowns']:<3}")
        return
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
