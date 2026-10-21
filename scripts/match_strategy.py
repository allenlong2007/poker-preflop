"""Experiment 26: one complete 100bb heads-up strategy, both seats, every street.

For the head-to-head match. Everything is solved for 100bb effective, against an opponent who plays the
same game:

1. Preflop, one tree (preflop/shortstack.py with a normal 3-bet size and experiment 24's per-hand postflop
   adjustments):
       BTN:  fold | limp | raise to s | all-in
         limp  -> BB: check | raise to r | all-in       -> BTN: fold | call | all-in -> BB: fold | call
         raise -> BB: fold | call | 3-bet to t | all-in -> BTN: fold | call | all-in -> BB: fold | call
   Grid over s (2.0-4.0), t (2.5x-5x s) and r (3-5bb): the BB picks t and r against each open, the BTN
   picks s. The non-chosen sizes are kept too, so the BB has an answer to any open size and the BTN to
   any 3-bet size.
2. Postflop, the whole-hand model (`OPTS_MATCH`: c-bets 1/3, 2/3, pot; pot-sized raises) for every pot
   type that reaches a flop, with the ranges from step 1:
       srp      the BTN raises, the BB calls
       3bet     the BB 3-bets, the BTN calls (the BB is the aggressor, out of position)
       limp     the BTN limps, the BB checks
       iso      the BTN limps, the BB raises, the BTN calls
       srp_alt  a single-raised pot after a 2.25bb open (if the main open is 2.25: after 3.0bb), so the BB
                is ready for an opponent who opens a different size
   A 4-bet is all-in at 100bb, so 4-bet pots have no postflop play.
3. A strategy book (output/match/STRATEGY.md) and the policy files the bot will read
   (output/match/policy/*.npz).

Usage:
    python scripts/match_strategy.py preflop           (~5 min)
    python scripts/match_strategy.py postflop [pots]   (~15-25 min per pot type, cached)
    python scripts/match_strategy.py postflop-fine [pots]   (experiment 30, ~30 min per pot type)
    python scripts/match_strategy.py book
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from open_size_study import OPTS_FINE, OPTS_MATCH
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.flopgame import TEXTURES as FLOP_TEXTURES
from preflop.plots import ACTION_COLORS, action_grid
from preflop.shortstack import solve_short

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("match_quick" if QUICK else "match")
CACHE = OUT / "cache"                          # gitignored
POLICY = OUT / "policy"                        # trained postflop policies for the bot
POLICY_FINE = OUT / "policy_fine"              # experiment 30: finer turn / river sizes (OPTS_FINE)
for d in (CACHE, POLICY, POLICY_FINE):
    d.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 200_000)
STACK = 100
OPENS = [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
T_MULT = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ISO = [3.0, 3.5, 4.0, 5.0]
ITERS_GRID, ITERS_FINAL = 1500, 5000
_EQ = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = _EQ["E"], _EQ["W"]
COMBOS = np.array([n_combos(h) for h in HANDS])
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
ALPHA, R_OOP = (float(v) for v in pd.read_csv(FIT, header=None, index_col=0)[1][["alpha", "r_oop"]])
ADJ = np.load(ROOT / "output" / "realization_loop" / "cache" / "shape" / "final.npz")   # experiment 24
A_IP, A_OOP = ADJ["adj_btn"], ADJ["adj_bb"]
STREETS = ("flop", "turn", "river")
# Manual preflop overrides, applied to the book and to the bot's preflop policy (not to the solves).
# 22 as the BTN: the model folds it (fold -0.50 vs limp -0.58bb) because experiment 24 rates small pairs
# poorly postflop, which looks like an artifact of the 12 hand groups (set value is lumped in). Limp it.
# (Experiment 26 forced 22 to limp; since experiment 33's calibration the model raises it 95% itself.)
OVERRIDES = {}
LINES = {"no bet last street": [0, 1, 2], "BTN bet last street": [3, 4, 5], "BB bet last street": [6, 7, 8]}
HIGH = [("A-high", 0, "A-high"), ("K-high", 0, "K-high"), ("Q/J-high", 0, "Q/J-high"),
        ("T-high or lower", 0, "T-high or lower"), ("paired", 1, "paired"), ("monotone", 2, "monotone"),
        ("connected", 3, "connected")]


for _a, _c in {"limp": "#86b6ef", "raise": "#2a78d6", "all-in": "#eb6834", "check": "#e4e3df",
            "4-bet": "#b8401a", "re-raise": "#b8401a"}.items():
    ACTION_COLORS.setdefault(_a, _c)


def pct(p, w=None):
    w = COMBOS if w is None else COMBOS * w
    return float((p * w).sum() / max(w.sum(), 1e-12))


def tag(x):
    return f"{x:g}".replace(".", "_")


# ----------------------------------------------------------------------------- 1. preflop

# Experiment 33: pots that see a flop are valued the way the whole-hand model plays them (a level correction
# per pot type, scripts/preflop_calibration.py). Limped pots keep their old value: their correction rests on
# the whole-hand model's limped-pot value, which its over-folding BB inflates, and it alone flips the BTN
# to limping 73%. The open is fixed at 2.25bb (2.0 scores slightly higher in the model; 2.25 is the
# minimax-regret choice across the old and calibrated models and keeps a trained postflop pot).
CALIBRATION = OUT / "calibration.json"
LIMP_LEVEL_SHARE = 0.0
MAIN_OPEN = 2.25
# Experiment 36: a non-all-in 4-bet (and a re-raise over the BB's raise of a limp) to F4 x the raise; the BTN's
# all-in 4-bet only at JAM4_DEPTH bb or shallower; the main spot is also solved at DEPTHS for shorter stacks.
# Limped and iso-raised pots lose LIMP_ISO_SHIFT of the pot for the BTN: limping is worth only ~1bb/100 in
# the model and vanishes at -0.03, inside the model's error (its BB over-folds, experiment 35), so the BTN
# limps ~14% instead of ~32%.
F4 = 2.2
JAM4_DEPTH = 40
DEPTHS = (60, 40, 25, 15)
SMALL_3BETS = (1.5, 2.0)                       # extra 3-bet sizes (x the open) the BTN gets answers for
LIMP_ISO_SHIFT = -0.015
VERSION = "exp36"


def levels_for(s):
    if not CALIBRATION.exists():
        return None
    best = json.load(open(CALIBRATION))["best"]
    key = min(best, key=lambda k: abs(float(k) - s))
    lv = dict(best[key]["levels"])
    lv["limp"] = lv["limp"] * LIMP_LEVEL_SHARE + LIMP_ISO_SHIFT
    lv["iso"] += LIMP_ISO_SHIFT
    return lv


def _solve(job):
    s, t, r, iters, *rest = job
    stack = rest[0] if rest else STACK
    return solve_short(E, W, s, r, stack, iters, r_oop=R_OOP, alpha=ALPHA, t3=t, adj_ip=A_IP, adj_oop=A_OOP,
                       levels=levels_for(s), f4=F4, jam4_depth=JAM4_DEPTH)


def _ev(job):
    return _solve(job).btn_ev()


def preflop():
    path = CACHE / f"preflop_{VERSION}.json"
    if path.exists():
        return json.load(open(path))
    grid = [(s, round(m * s, 4), r) for s in OPENS for m in T_MULT for r in ISO]
    t0 = time.time()
    with Pool(12) as pool:
        evs = dict(zip(grid, pool.map(_ev, [(s, t, r, ITERS_GRID) for s, t, r in grid])))
    print(f"preflop grid: {len(grid)} solves in {time.time() - t0:.0f}s", flush=True)
    best = {}
    for s in OPENS:
        cand = {(t, r): v for (s2, t, r), v in evs.items() if s2 == s}
        t, r = min(cand, key=cand.get)          # the BB minimises the BTN's EV
        best[s] = {"t": t, "r": r, "btn_ev": cand[(t, r)]}
    s_main = MAIN_OPEN if MAIN_OPEN in best else max(best, key=lambda s: best[s]["btn_ev"])
    info = {"s_main": s_main, "best": {str(k): v for k, v in best.items()},
            "grid": [{"s": s, "t": t, "r": r, "btn_ev": v} for (s, t, r), v in evs.items()]}
    json.dump(info, open(path, "w"), indent=1)
    return info


def node_policies(res):
    return {name: {a: v.tolist() for a, v in res.actions(name).items()} for name in res.strat}


def final_solves(info):
    """Full strategies: the main spot, every open size (BB's answers) and every 3-bet size (BTN's answers)."""
    path = CACHE / f"preflop_final_{VERSION}.json"
    if path.exists():
        raw = json.load(open(path))
        return {k: {float(x): v for x, v in raw[k].items()} for k in raw}
    s_main = info["s_main"]
    b = info["best"]
    jobs = [(s, b[str(s)]["t"], b[str(s)]["r"], ITERS_FINAL) for s in OPENS]
    jobs += [(s_main, round(m * s_main, 4), b[str(s_main)]["r"], ITERS_FINAL) for m in SMALL_3BETS + tuple(T_MULT)]
    deep = [(s_main, b[str(s_main)]["t"], b[str(s_main)]["r"], ITERS_FINAL, depth) for depth in DEPTHS]
    with Pool(12) as pool:
        results = pool.map(_solve, jobs + deep)
    by_depth = {depth: {"nodes": node_policies(res), "btn_ev": res.btn_ev()}
                for (*_, depth), res in zip(deep, results[len(jobs):])}
    results = results[:len(jobs)]
    d = {"by_open": {}, "by_3bet": {}, "by_depth": by_depth}
    for (s, t, r, _), res in zip(jobs[:len(OPENS)], results[:len(OPENS)]):
        d["by_open"][s] = {"t": t, "r": r, "btn_ev": res.btn_ev(), "nodes": node_policies(res),
                           "reach": {n: [x.tolist() for x in res.reach(n)] for n in res.strat}}
    for (s, t, r, _), res in zip(jobs[len(OPENS):], results[len(OPENS):]):
        d["by_3bet"][t] = {"s": s, "r": r, "nodes": node_policies(res)}
    json.dump({k: {str(x): v for x, v in d[k].items()} for k in d}, open(path, "w"))
    return d


def pot_ranges(d, s_main):
    """BTN and BB ranges (per-hand weights) entering each postflop pot type."""
    alt = 2.25 if s_main != 2.25 else 3.0
    main, other = d["by_open"][s_main], d["by_open"][alt]
    n = lambda spot, name, a: np.array(spot["nodes"][name][a])
    t, r = main["t"], main["r"]
    return {
        "srp": (s_main, n(main, "btn_root", "raise"), n(main, "bb_vs_raise", "call")),
        "3bet": (t, n(main, "btn_root", "raise") * n(main, "btn_vs_3bet_small", "call"), n(main, "bb_vs_raise", "3-bet")),
        "limp": (1.0, n(main, "btn_root", "limp"), n(main, "bb_vs_limp", "check")),
        "iso": (r, n(main, "btn_root", "limp") * n(main, "btn_vs_iso", "call"), n(main, "bb_vs_limp", "raise")),
        "srp_alt": (alt, n(other, "btn_root", "raise"), n(other, "bb_vs_raise", "call")),
    }


# ----------------------------------------------------------------------------- 2. postflop

def train(pot, s, btn, bb, fine=False):
    """fine: experiment 30's finer turn / river sizes (OPTS_FINE, 1.6x the batches for the bigger tables)."""
    opts, folder = (OPTS_FINE, POLICY_FINE) if fine else (OPTS_MATCH, POLICY)
    path = folder / f"{pot}.npz"
    if path.exists():
        print(f"{pot}: cached")
        return
    print(f"== {pot}{' (fine sizes)' if fine else ''}: pot {2 * s:g}bb, {STACK - s:g}bb behind", flush=True)
    t0 = time.time()
    batches = int(BATCHES * 1.6) if fine else BATCHES
    sigma, _ = fg.learn(btn, bb, opts, s=s, stack=STACK, batches=batches, hands_per_batch=HANDS_PER_BATCH,
                        seed=1, verbose=False)
    # Deep-stacked limped pots learn ~2.5x slower per batch with the finer sizes (~75 min); a quarter of the
    # evaluation hands keeps each run under two hours (the bot only needs sigma; this feeds the report).
    print(f"   learned in {(time.time() - t0) / 60:.1f} min; evaluating", flush=True)
    n_eval = EVAL_HANDS // 4 if fine and pot in ("limp", "iso") else EVAL_HANDS
    value, acc = fg.evaluate(btn, bb, sigma, opts, s=s, stack=STACK, n=n_eval)
    out = {"value_btn": value, "pot_to": s, "btn_range": btn, "bb_range": bb,
           "opts_json": json.dumps({k: v for k, v in opts.items()})}
    for key in sigma:
        out[f"sigma_{key[0]}_{key[1]}"] = sigma[key]
        out[f"strat_{key[0]}_{key[1]}"] = acc.strat[key]
        out[f"reach_{key[0]}_{key[1]}"] = acc.reach[key]
    np.savez_compressed(path, **out)
    print(f"   BTN net {value - s:+.3f}bb per pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ----------------------------------------------------------------------------- 3. the book

def action_names(street, kind):
    leads, bets = fg.street_sizes(street, OPTS_MATCH)
    nm = lambda x: {1 / 3: "1/3", 2 / 3: "2/3", 1 / 2: "1/2", 3 / 4: "3/4", 1.0: "pot", 3 / 2: "1.5x"}.get(x, f"{x:g}")
    raise_ = ["raise"] * len(fg.raise_sizes(OPTS_MATCH, street))
    return {"bb_first": ["check"] + [f"lead {nm(x)}" for x in leads],
            "btn_vs_check": ["check"] + [f"bet {nm(x)}" for x in bets],
            "bb_vs_bet": ["fold", "call"] + raise_, "btn_vs_xr": ["fold", "call"],
            "btn_vs_lead": ["fold", "call"] + raise_, "bb_vs_raise": ["fold", "call"]}[kind]


def node_table(d, street, kind, lines=None, tex=None):
    """Action frequencies by hand bucket (rows) for one node, optionally for some lines / textures."""
    key = f"{street}_{kind}"
    if f"strat_{key}" not in d.files:
        return None
    strat, reach = d[f"strat_{key}"], d[f"reach_{key}"]
    m = np.ones(reach.shape)
    if lines is not None:
        keep = np.zeros(reach.shape[1], bool)
        keep[[x for x in lines if x < reach.shape[1]]] = True
        m[:, ~keep] = 0
    if tex is not None:
        keep = np.zeros(reach.shape[0], bool)
        keep[tex] = True
        m[~keep] = 0
    names = action_names(street, kind)
    rows, buckets = [], (fg.RIVER_BUCKETS if street == 2 else fg.BUCKETS)
    for b, bname in enumerate(buckets):
        rb = (reach * m)[:, :, b].sum()
        if rb <= 0:
            continue
        sb = (strat * m[..., None])[:, :, b].reshape(-1, strat.shape[-1]).sum(0) / rb
        rows.append({"hand": bname, "share": rb, **{names[k]: sb[k] for k in range(len(names))}})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["share"] = df["share"] / df["share"].sum()
    tot = {"hand": "**all hands**", "share": 1.0,
           **{c: float((df[c] * df["share"]).sum()) for c in names}}
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) if c == "hand" else f"{r[c]:.0%}" for c in cols) + " |")
    return "\n".join(out)


NODE_TITLES = {"bb_first": "BB acts first (check or lead)", "btn_vs_check": "BTN after the BB checks",
               "bb_vs_bet": "BB facing a BTN bet", "btn_vs_xr": "BTN facing a check-raise",
               "btn_vs_lead": "BTN facing a BB lead", "bb_vs_raise": "BB's lead got raised"}
POT_TITLES = {"srp": "Single-raised pot (BTN raises, BB calls)", "3bet": "3-bet pot (BB 3-bets, BTN calls)",
              "limp": "Limped pot (BTN limps, BB checks)", "iso": "Iso-raised pot (BTN limps, BB raises, BTN calls)",
              "srp_alt": "Single-raised pot after a different open size (2.25bb, or 3.0bb if the main open is 2.25)"}


def book():
    info = preflop()
    d = final_solves(info)
    s_main = info["s_main"]
    main = d["by_open"][s_main]
    L = ["# 100bb heads-up strategy book", "",
         "Generated by `scripts/match_strategy.py` (experiment 26). All frequencies are shares of the hands that "
         "reach the decision. Postflop tables are by hand strength (12 groups), averaged over boards.", ""]
    # Preflop.
    L += ["## Preflop", "", f"**BTN opens {s_main:g}bb.** The BB 3-bets to **{main['t']:g}bb** and raises limps to "
          f"**{main['r']:g}bb**. A 4-bet goes to {F4:g}x the 3-bet (all-in only at {JAM4_DEPTH}bb or shorter).", ""]
    nd = main["nodes"]
    for (node, hand), new in OVERRIDES.items():
        for a, v in new.items():
            nd[node][a][HANDS.index(hand)] = v
    L += ["Manual override: " + "; ".join(f"{h} at `{n}` -> " + ", ".join(f"{a} {v:.0%}" for a, v in new.items() if v)
                                          for (n, h), new in OVERRIDES.items()) + " (see `OVERRIDES`).", ""]
    reach = {k: (np.array(v[0]), np.array(v[1])) for k, v in main["reach"].items()}
    for name, who, title in (("btn_root", 0, "BTN first action"), ("bb_vs_raise", 1, "BB vs the open"),
                             ("btn_vs_3bet_small", 0, "BTN vs the 3-bet"), ("bb_vs_4bet_small", 1, "BB vs a 4-bet"),
                             ("btn_vs_5bet", 0, "BTN vs a 5-bet (all-in)"), ("bb_vs_4bet", 1, "BB vs a 4-bet (all-in)"),
                             ("bb_vs_limp", 1, "BB vs a limp"), ("btn_vs_iso", 0, "BTN vs the BB's raise over its limp"),
                             ("bb_vs_limp_3bet", 1, "BB vs a re-raise of its iso-raise"),
                             ("bb_vs_limp_reraise", 1, "BB vs a limp-reraise (all-in)"),
                             ("btn_vs_3bet", 0, "BTN vs an all-in 3-bet"), ("bb_vs_open_jam", 1, "BB vs an all-in open")):
        if name not in nd:
            continue
        w, w_opp = reach[name][who], reach[name][1 - who]
        if w.sum() < 1e-6:
            continue
        # A spot the opponent's equilibrium never reaches is kept as our answer in case they go there anyway.
        off_path = pct(w_opp) < 0.005
        freqs = {a: pct(np.array(v), w) for a, v in nd[name].items()}
        acts = {a: np.array(v) for a, v in nd[name].items()}
        heading = title + (" (off the equilibrium path: our answer if the opponent does this)" if off_path else "")
        action_grid(acts, f"{title} ({s_main:g}bb open, 100bb)", OUT / f"pre_{name}.png",
                    subtitle="   ".join(f"{a.capitalize()} {v:.0%}" for a, v in freqs.items()))
        L += [f"### {heading}", "", "   ".join(f"**{a}** {v:.0%}" for a, v in freqs.items()), "",
              f"![{title}](pre_{name}.png)", ""]
    # Answers to other open sizes and 3-bet sizes.
    L += ["### The BB against other open sizes", "", "| BTN opens to | BB folds | calls | 3-bets | to | BTN EV (mbb/hand) |",
          "|---|---|---|---|---|---|"]
    for s, v in sorted(d["by_open"].items()):
        n2 = v["nodes"]["bb_vs_raise"]
        L.append(f"| {s:g}bb | {pct(np.array(n2['fold'])):.0%} | {pct(np.array(n2['call'])):.0%} | "
                 f"{pct(np.array(n2['3-bet'])):.0%} | {v['t']:g}bb | {1000 * v['btn_ev']:+.0f} |")
    L += ["", "### The BTN against other 3-bet sizes", "", "| BB 3-bets to | BTN folds | calls | 4-bets |",
          "|---|---|---|---|"]
    raised = np.array(main["nodes"]["btn_root"]["raise"])
    for t, v in sorted(d["by_3bet"].items()):
        n3 = v["nodes"]["btn_vs_3bet_small"]
        L.append(f"| {t:g}bb | {pct(np.array(n3['fold']), raised):.0%} | {pct(np.array(n3['call']), raised):.0%} | "
                 f"{pct(np.array(n3.get('4-bet', n3.get('all-in'))), raised):.0%} |")
    L.append("")
    # Per-hand preflop table for the bot and for reference.
    pre_rows = []
    for i, h in enumerate(HANDS):
        r = {"hand": h}
        for name, acts in nd.items():
            for a, v in acts.items():
                r[f"{name}:{a}"] = round(v[i], 3)
        pre_rows.append(r)
    pd.DataFrame(pre_rows).to_csv(OUT / "preflop_by_hand.csv", index=False)
    # The bot's preflop policy: the main spot with overrides, plus the answers to other sizes.
    json.dump({"open": s_main, "threebet_to": main["t"], "iso_to": main["r"], "nodes": nd,
               "vs_open": {f"{k:g}": {"threebet_to": v["t"], "bb_vs_raise": v["nodes"]["bb_vs_raise"]}
                           for k, v in d["by_open"].items()},
               "vs_3bet": {f"{k:g}": v["nodes"]["btn_vs_3bet_small"] for k, v in d["by_3bet"].items()},
               "fourbet_x": F4, "jam4_depth": JAM4_DEPTH,
               "by_depth": {f"{k:g}": v["nodes"] for k, v in d.get("by_depth", {}).items()},
               "hands": HANDS, "overrides": [f"{h}@{n}" for n, h in OVERRIDES]},
              open(POLICY / "preflop.json", "w"))

    # Postflop.
    groups = {name: [t for t, tex in enumerate(FLOP_TEXTURES) if tex[k] == v] for name, k, v in HIGH}
    summary = []
    for pot, title in POT_TITLES.items():
        path = POLICY / f"{pot}.npz"
        if not path.exists():
            continue
        dd = np.load(path)
        s = float(dd["pot_to"])
        summary.append({"pot": title, "pot_bb": 2 * s, "btn_net": float(dd["value_btn"]) - s})
        L += [f"## {title}", "", f"Pot {2 * s:g}bb, {STACK - s:g}bb behind. BTN nets "
              f"{float(dd['value_btn']) - s:+.2f}bb per pot.", ""]
        for st, sname in enumerate(STREETS):
            L += [f"### {sname.capitalize()}", ""]
            for kind, ntitle in NODE_TITLES.items():
                splits = {"": None} if st == 0 else LINES
                for lname, lines in splits.items():
                    t = node_table(dd, st, kind, lines)
                    if t is None or t["share"].iloc[:-1].sum() == 0:
                        continue
                    L += [f"**{ntitle}{' — ' + lname if lname else ''}**", "", md(t), ""]
            if st == 0:                        # the c-bet by board
                rows = []
                for gname, ts in groups.items():
                    t = node_table(dd, 0, "btn_vs_check", tex=ts)
                    if t is not None:
                        rows.append({"hand": gname, **{c: t.iloc[-1][c] for c in t.columns if c not in ("hand", "share")}})
                if rows:
                    L += ["**BTN after the BB checks, by board (all hands)**", "", md(pd.DataFrame(rows)), ""]
    pd.DataFrame(summary).round(3).to_csv(OUT / "postflop_summary.csv", index=False)
    (OUT / "STRATEGY.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'STRATEGY.md'} ({len(L)} lines)")
    print(pd.DataFrame(summary).round(3).to_string(index=False))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "book"
    if cmd == "preflop":
        info = preflop()
        d = final_solves(info)
        print("main open", info["s_main"], {s: (v["t"], v["r"], round(1000 * v["btn_ev"], 1)) for s, v in d["by_open"].items()})
    elif cmd in ("postflop", "postflop-fine"):
        info = preflop()
        d = final_solves(info)
        ranges = pot_ranges(d, info["s_main"])
        for pot in (sys.argv[2:] or list(ranges)):
            s, btn, bb = ranges[pot]
            train(pot, s, btn, bb, fine=cmd == "postflop-fine")
    else:
        book()


if __name__ == "__main__":
    main()
