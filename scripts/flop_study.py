"""Experiment 27: card-level postflop strategy on one specific flop (the match's single-raised pot).

The whole-hand model plays every flop by texture class x 12 hand groups. Here TexasSolver solves one exact
flop, every combo separately, with the match setup (experiment 26):

    ranges   the BTN's 3bb opening range vs the BB's calling range (100bb; pot 6bb, 97bb behind)
    flop     BB: check or lead 33%; BTN: check or bet 33 / 66 / 100%; raises 50% of the pot after calling
    turn     either player bets 66% (no raises)        river   either player bets 75% (no raises)
    10 chips per bb, so bet sizes are not rounded (experiment 25)

Outputs per combo: the flop strategy for both players and the turn strategy after flop check-check and
after a called c-bet. It is compared with the whole-hand model's strategy on the same flop (its texture
class and hand groups) to see what card-level detail changes. The per-combo policy goes to
output/match/flops/<flop>.json for the bot.

Usage:
    python scripts/flop_study.py solve "Kh 8d 3c"     (~5 min on AC power, one solve at a time)
    python scripts/flop_study.py report "Kh 8d 3c"
    python scripts/flop_study.py solve-all [first] [last]   (the representative flops, REPS; cached)
    python scripts/flop_study.py report-all
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import eval7
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preflop import fullgame as fg
from preflop.cards import HANDS
from preflop.flopgame import texture as flop_texture
from preflop.plots import heatmap, range_grid

MATCH = ROOT / "output" / "match"
OUT = MATCH / "flops"
OUT.mkdir(parents=True, exist_ok=True)
RANGES = OUT / "srp_ranges.npz"
import match_strategy as ms
OPEN = ms.MAIN_OPEN                                # the match's open size (2.25bb since experiment 33)
ENV = {"SOLVER_TREE": "medium", "SOLVER_OPEN": f"{OPEN:g}", "SOLVER_RANGES": str(RANGES), "SOLVER_FLOP_SIZES": "33,66,100",
       "SOLVER_FLOP_DONK": "33", "SOLVER_DUMP_ROUNDS": "2", "SOLVER_CHIP_SCALE": "10", "SOLVER_PARALLEL": "1",
       "SOLVER_THREADS": "10"}
SCALE, POT = 10, 2 * OPEN
# One representative flop per texture class (experiment 28): the flop closest to the class's median ranks,
# from all 22,100 flops; K-high dry rainbow keeps experiment 27's Kh 8d 3c. 32 of the 48 nominal classes
# exist (a paired flop can't be monotone or connected here).
REPS = {0: "Ad Th 5s", 1: "Ad 5h 4s", 2: "Ah Ts 5s", 3: "Ah 5s 4s", 4: "As Ts 5s", 5: "As 5s 4s", 6: "Ah Ad 8s",
        8: "As Ah 7s", 12: "Kh 8d 3c", 13: "Kd Jh 9s", 14: "Kh 9s 4s", 15: "Kh Js 9s", 16: "Ks 9s 4s",
        17: "Ks Js 9s", 18: "Kh Kd 7s", 20: "Ks Kh 7s", 24: "Qd 8h 4s", 25: "Jd Th 8s", 26: "Qh 8s 4s",
        27: "Jh Ts 8s", 28: "Qs 8s 4s", 29: "Js Ts 8s", 30: "Jh Jd 6s", 32: "Js Jh 6s", 36: "9d 6h 3s",
        37: "8d 6h 4s", 38: "9h 6s 3s", 39: "8h 6s 4s", 40: "9s 6s 3s", 41: "8s 6s 4s", 42: "8d 4s 4h",
        44: "8s 4s 4h"}


def write_ranges():
    """The match's single-raised-pot ranges (current preflop solve) in the format solver_compare.py reads."""
    info = ms.preflop()
    _, btn, bb = ms.pot_ranges(ms.final_solves(info), info["s_main"])["srp"]
    np.savez(RANGES, btn_range=btn, bb_range=bb)


def sc():
    os.environ.update(ENV)
    import importlib
    import solver_compare
    return importlib.reload(solver_compare)


def solve(flop):
    write_ranges()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "solver_compare.py"), "solve"],
                   env={**os.environ, **ENV, "SOLVER_FLOPS": flop}, check=True)


def label(a):
    if a in ("CHECK", "CALL", "FOLD"):
        return a.lower()
    kind, amt = a.split()[0], float(a.split()[1])
    return f"{kind.lower()} {amt / SCALE:g}bb"


def node_rows(node, who, extra):
    acts = [label(a) for a in node["strategy"]["actions"]]
    return [{"combo": c, **extra, **dict(zip(acts, p))} for c, p in node["strategy"]["strategy"].items()]


def parse(flop):
    s = sc()
    d = json.load(open(s.RAW / f"{s.slug(flop)}.json"))
    rows = {"bb_first": node_rows(d, "bb", {})}
    for a, ch in d["childrens"].items():
        if a == "CHECK":
            rows["btn_vs_check"] = node_rows(ch, "btn", {})
            for b, ch2 in ch.get("childrens", {}).items():
                if b.startswith("BET") and "strategy" in ch2:
                    rows.setdefault("bb_vs_bet", []).extend(node_rows(ch2, "bb", {"facing": label(b)}))
        elif a.startswith("BET") and "strategy" in ch:
            rows.setdefault("btn_vs_lead", []).extend(node_rows(ch, "btn", {"facing": label(a)}))
    return d, {k: pd.DataFrame(v).fillna(0.0) for k, v in rows.items()}


def turn_rows(d):
    """Turn strategies after flop check-check and after a called 33% c-bet, averaged over turn cards."""
    out = []
    chk = d["childrens"]["CHECK"]
    paths = {"check-check": chk["childrens"].get("CHECK")}
    small = [b for b in chk.get("childrens", {}) if b.startswith("BET")]
    if small:
        bet = sorted(small, key=lambda b: float(b.split()[1]))[0]
        paths["BTN bets 33%, BB calls"] = chk["childrens"][bet]["childrens"].get("CALL")
    for name, node in paths.items():
        if node is None or "dealcards" not in node:
            continue
        for card, sub in node["dealcards"].items():
            if "strategy" not in sub:
                continue
            for c, p in sub["strategy"]["strategy"].items():          # BB acts first on the turn
                out.append({"line": name, "turn": card, "who": "BB first", "combo": c,
                            **dict(zip([label(a) for a in sub["strategy"]["actions"]], p))})
            ch = sub.get("childrens", {}).get("CHECK")
            if ch is not None and "strategy" in ch:
                for c, p in ch["strategy"]["strategy"].items():
                    out.append({"line": name, "turn": card, "who": "BTN after check", "combo": c,
                                **dict(zip([label(a) for a in ch["strategy"]["actions"]], p))})
    return pd.DataFrame(out).fillna(0.0)


def bucket_of(flop):
    cards = [eval7.Card(c) for c in flop.split()]
    return lambda c: fg.fine_bucket((eval7.Card(c[:2]), eval7.Card(c[2:])), cards)


def report(flop, quiet=False):
    s = sc()
    rng = np.load(RANGES)
    wb, wo = dict(zip(HANDS, rng["btn_range"])), dict(zip(HANDS, rng["bb_range"]))
    d, nodes = parse(flop)
    tag = s.slug(flop)
    bk = bucket_of(flop)
    model = np.load(MATCH / "policy" / "srp.npz")
    t = flop_texture([eval7.Card(c) for c in flop.split()])
    names = fg.BUCKETS
    summary = []

    def weighted(df, who):
        w = (wb if who == "btn" else wo)
        df = df.copy()
        df["cls"] = df.combo.map(s.class_of)
        df["w"] = df.cls.map(w)
        df["bucket"] = df.combo.map(bk)
        return df[df.w > 0]

    # 1. The BTN's c-bet: solver per combo vs the model per hand group.
    bt = weighted(nodes["btn_vs_check"], "btn")
    acts = [c for c in bt.columns if c in ("check",) or c.startswith("bet")]
    mod = model["sigma_0_btn_vs_check"][t, 0]                      # check, 1/3, 2/3, pot
    rows = []
    for b in sorted(bt.bucket.unique()):
        g = bt[bt.bucket == b]
        sol = {a: float((g[a] * g.w).sum() / g.w.sum()) for a in acts}
        spread = float(np.sqrt(((g["check"] - sol["check"]) ** 2 * g.w).sum() / g.w.sum()))
        rows.append({"hand": names[b], "weight": float(g.w.sum()),
                     **{f"solver {a}": v for a, v in sol.items()},
                     "solver combo spread (check)": spread,
                     **{f"model {a}": float(v) for a, v in zip(["check"] + acts[1:], mod[b])}})
    cb = pd.DataFrame(rows)
    cb["weight"] = cb.weight / cb.weight.sum()
    cb.round(3).to_csv(OUT / f"{tag}_btn_cbet.csv", index=False)
    tot = {a: float((bt[a] * bt.w).sum() / bt.w.sum()) for a in acts}
    mtot = {a: float((cb[f"model {a}"] * cb.weight).sum()) for a in ["check"] + acts[1:] if f"model {a}" in cb}
    summary.append({"decision": "BTN after the BB checks", "solver": tot, "model": mtot})

    # 2. The BB's flop: lead or check, and vs each c-bet size.
    bb0 = weighted(nodes["bb_first"], "bb")
    a0 = [c for c in bb0.columns if c == "check" or c.startswith("bet")]
    summary.append({"decision": "BB first", "solver": {a: float((bb0[a] * bb0.w).sum() / bb0.w.sum()) for a in a0},
                    "model": dict(zip(["check", "lead 1/3", "lead 3/4"],
                                      _avg(model["sigma_0_bb_first"][t, 0], bb0, model)))})
    vb = weighted(nodes["bb_vs_bet"], "bb")
    for size, g in vb.groupby("facing"):
        a1 = [c for c in g.columns if c in ("fold", "call") or c.startswith("raise")]
        summary.append({"decision": f"BB vs {size}", "solver": {a: float((g[a] * g.w).sum() / g.w.sum()) for a in a1}})

    # 3. Class-level grids for the BTN's c-bet (averaged over suits) and the BB's defence vs 33%.
    cls = bt.groupby("cls").apply(lambda g: pd.Series({a: (g[a] * 1).mean() for a in acts}), include_groups=False)
    bet_any = np.array([1 - cls.loc[h, "check"] if h in cls.index else np.nan for h in HANDS])
    range_grid(bet_any, f"BTN c-bets on {flop} (TexasSolver, 3bb open, 100bb)", OUT / f"{tag}_btn_cbet.png",
               note="Blank = not in the BTN's opening range. Averaged over each hand's suit combos.")
    small = vb[vb.facing == sorted(vb.facing.unique(), key=lambda x: float(x.split()[1][:-2]))[0]]
    cls2 = small.groupby("cls").apply(lambda g: pd.Series({"fold": g["fold"].mean()}), include_groups=False)
    defend = np.array([1 - cls2.loc[h, "fold"] if h in cls2.index else np.nan for h in HANDS])
    range_grid(defend, f"BB continues vs a 33% c-bet on {flop}", OUT / f"{tag}_bb_defend.png",
               note="Call or raise. Blank = not in the BB's calling range.")
    comp = cb.set_index("hand")[["solver check", "model check"]].rename(columns=lambda c: c.replace(" check", ""))
    heatmap(1 - comp, f"BTN c-bet frequency on {flop} by hand group: solver vs model", "", "BTN hand",
            OUT / f"{tag}_cbet_vs_model.png", fmt="pct", figsize=(6, 5.5))

    # 4. Turn: after check-check and after a called small c-bet.
    tr = turn_rows(d)
    turn_sum = []
    for (line, who), g in tr.groupby(["line", "who"]):
        g = weighted(g, "btn" if who.startswith("BTN") else "bb")
        acts_t = [c for c in g.columns if c in ("check",) or c.startswith("bet")]
        turn_sum.append({"line": line, "who": who, **{a: float((g[a] * g.w).sum() / g.w.sum()) for a in acts_t}})
        by_card = g.groupby("turn").apply(lambda x: float(1 - (x["check"] * x.w).sum() / x.w.sum()), include_groups=False)
        turn_sum[-1]["most bets on"] = " ".join(f"{c} {v:.0%}" for c, v in by_card.sort_values(ascending=False)[:4].items())
        turn_sum[-1]["fewest bets on"] = " ".join(f"{c} {v:.0%}" for c, v in by_card.sort_values()[:4].items())
    ts = pd.DataFrame(turn_sum)
    ts.round(3).to_csv(OUT / f"{tag}_turn.csv", index=False)

    # 5. The per-combo policy for the bot.
    policy = {k: v.drop(columns=[c for c in ("cls", "w", "bucket") if c in v]).round(4).to_dict("records")
              for k, v in nodes.items()}
    json.dump({"flop": flop, "pot_bb": POT, "behind_bb": 100 - OPEN, "nodes": policy}, open(OUT / f"{tag}.json", "w"))

    json.dump(summary, open(OUT / f"{tag}_summary.json", "w"), indent=1)
    if quiet:
        return summary, cb, ts
    pd.set_option("display.width", 250)
    for r in summary:
        print(r["decision"], "| solver", {k: round(v, 3) for k, v in r["solver"].items()},
              "| model", {k: round(v, 3) for k, v in r.get("model", {}).items()})
    print(cb.round(2).to_string(index=False))
    print(ts.round(3).to_string(index=False))
    return summary, cb, ts


def class_frequencies():
    """Share of all 22,100 flops in each texture class."""
    import itertools
    from collections import Counter
    deck = [r + su for r in "23456789TJQKA" for su in "shdc"]
    c = Counter(flop_texture([eval7.Card(x) for x in f]) for f in itertools.combinations(deck, 3))
    return {t: n / 22100 for t, n in c.items()}


def report_all():
    from preflop.flopgame import TEXTURES
    freq = class_frequencies()
    rows, index = [], {}
    for t, flop in REPS.items():
        s = sc()
        if not (s.RAW / f"{s.slug(flop)}.json").exists():
            continue
        summary, cb, _ = report(flop, quiet=True)
        name = ", ".join(TEXTURES[t])
        index[t] = {"class": name, "flop": flop, "file": f"{s.slug(flop)}.json", "share_of_flops": freq[t]}
        r = {"class": name, "flop": flop, "share": freq[t]}
        btn = next(x for x in summary if x["decision"] == "BTN after the BB checks")
        for k, v in btn["solver"].items():
            r[f"solver {k}"] = v
        for k, v in btn["model"].items():
            r[f"model {k}"] = v
        bb = next(x for x in summary if x["decision"] == "BB first")
        r["BB leads"] = 1 - bb["solver"].get("check", 1.0)
        for x in summary:
            if x["decision"].startswith("BB vs bet"):
                r[f"BB folds vs {x['decision'][10:]}"] = x["solver"].get("fold", np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "classes.csv", index=False)
    json.dump({"note": "texture class -> representative solved flop (experiment 28)", "open": OPEN,
               "classes": {str(k): v for k, v in index.items()}}, open(OUT / "index.json", "w"), indent=1)
    short = df.assign(label=df["flop"] + "  (" + (df["share"] * 100).round(1).astype(str) + "%)").set_index("label")
    sizes = [c for c in short.columns if c.startswith("solver ")]
    heatmap(short[sizes].rename(columns=lambda c: c.replace("solver ", "")),
            "BTN after the BB checks, by flop class (TexasSolver)", "", "Representative flop (share of all flops)",
            OUT / "classes_cbet.png", fmt="pct", figsize=(7.5, 12))
    mod = [c for c in short.columns if c.startswith("model ")]
    diff = pd.DataFrame({"solver bets": 1 - short["solver check"], "model bets": 1 - short["model check"]})
    heatmap(diff, "BTN c-bet frequency: solver vs 12-group model", "", "Representative flop", OUT / "classes_vs_model.png",
            fmt="pct", figsize=(5.5, 12))
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    w = df["share"] / df["share"].sum()
    print("weighted by how often each class comes up:",
          {c: round(float((df[c] * w).sum()), 3) for c in sizes + mod + ["BB leads"]})


def _avg(sig, df, model):
    """The model's BB-first strategy averaged over the BB's combos on this flop (by hand group)."""
    w = df.groupby("bucket").w.sum()
    return list((sig[w.index] * (w.values / w.sum())[:, None]).sum(0))


def main():
    cmd = sys.argv[1]
    if cmd == "solve-all":
        flops = list(REPS.values())
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(flops)
        write_ranges()
        subprocess.run([sys.executable, str(ROOT / "scripts" / "solver_compare.py"), "solve"],
                       env={**os.environ, **ENV, "SOLVER_FLOPS": ",".join(flops[lo:hi])}, check=True)
    elif cmd == "report-all":
        report_all()
    elif cmd == "solve":
        solve(sys.argv[2] if len(sys.argv) > 2 else "Kh 8d 3c")
    else:
        report(sys.argv[2] if len(sys.argv) > 2 else "Kh 8d 3c")


if __name__ == "__main__":
    main()
