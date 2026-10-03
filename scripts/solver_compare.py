"""Experiment 14: compare the learned flop strategy with a real solver (TexasSolver).

TexasSolver (github.com/bupticybee/TexasSolver, AGPL-3.0) solves the full
flop-turn-river game on exact cards. It is not part of this repo: download the
v0.2.0 macOS release into tools/ (gitignored). The binary is x86_64 (runs under
Rosetta on Apple silicon).

Each flop is solved with the same ranges as experiment 13 (BTN 2.25bb opening
range vs BB calling range from the preflop solver), pot 4.5bb, stacks 97.75bb,
and similar bet sizes:

    flop  BB: check only            BTN: check, bet 33% / 75% pot    BB vs a bet: fold / call / raise
    turn  BB: check or lead 66%     BTN: check or bet 66%            one raise allowed
    river BB: check or lead 75%     BTN: check or bet 75%            one raise allowed
    raises are by RAISE_PCT of the pot after calling (~3x a small bet), then fold / call

TREE = "full" gives the turn and river the same structure as the model: either
player can bet, one size, one raise. It needs 10-16 GB of memory per flop and
~3 min per iteration under Rosetta, so it is impractical on a laptop.
TREE = "medium" (default): either player can bet the turn and river, no raises there.
TREE = "simple": only the BTN can bet the turn and river; the BB calls or folds.
The flop decisions are identical in all three and match the model exactly.

Usage:
    python scripts/solver_compare.py solve   [medium|simple|full]   # solve the flops (skips solved ones)
    python scripts/solver_compare.py compare [medium|simple|full]   # compare solved flops with the model

"""
import json
import subprocess
import sys
import time
from pathlib import Path

import eval7
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, combos
from preflop.flopgame import BUCKETS, TEXTURES, bucket, texture
from preflop.plots import grouped_bars, scatter_compare

SOLVER_DIR = ROOT / "tools" / "TexasSolver-v0.2.0-MacOs"
TREE = sys.argv[2] if len(sys.argv) > 2 else "medium"
OUT = ROOT / "output" / "solver_compare" / TREE
RAW = ROOT / "output" / "solver_compare" / "raw" / TREE     # configs + JSON results (gitignored: large)
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
POLICIES = ROOT / "output" / "cbet_frequency" / "flop_policies.npz"
MODEL = ROOT / "output" / "bet_sizing" / "flop.npz"

POT, STACK = 4.5, 97.75
RAISE_PCT = 50
THREADS = 3                             # per solve; several flops run in parallel (PARALLEL)
PARALLEL = int(__import__("os").environ.get("SOLVER_PARALLEL", 5))
ACCURACY = 0.3                          # target exploitability, % of the pot
MAX_ITER = 150
FLOPS = ["As 7d 2c", "Ah Kd 5h", "Kh 8d 3c", "Qd Jh 4d", "Td 9h 6c", "8s 6s 4d",
         "7c 5d 3h", "Jh 7h 2h", "Kd Kc 4s", "7h 7d 2s", "Ts 5c 2d", "9c 6c 5d"]
BET_SIZES = {"flop": (33, 75), "turn": (66,), "river": (75,)}


def range_string(freqs):
    return ",".join(f"{h}:{f:.3f}" for h, f in zip(HANDS, freqs) if f > 0.005)


def config(flop, out_json, ranges):
    board = ",".join(flop.split())
    lines = [f"set_pot {POT}", f"set_effective_stack {STACK}", f"set_board {board}",
             f"set_range_oop {range_string(ranges['bb'])}", f"set_range_ip {range_string(ranges['btn'])}"]
    for street, sizes in BET_SIZES.items():
        sz = ",".join(str(x) for x in sizes)
        lines.append(f"set_bet_sizes ip,{street},bet,{sz}")
        if street == "flop" or TREE == "full":
            lines += [f"set_bet_sizes oop,{street},raise,{RAISE_PCT}", f"set_bet_sizes ip,{street},raise,{RAISE_PCT}"]
        if street != "flop" and TREE in ("full", "medium"):   # the BB may lead the turn and river (never the flop)
            lines += [f"set_bet_sizes oop,{street},bet,{sz}", f"set_bet_sizes oop,{street},donk,{sz}"]
    lines += ["set_allin_threshold 1.0", "set_raise_limit 1", "build_tree", f"set_thread_num {THREADS}",
              f"set_accuracy {ACCURACY}", f"set_max_iteration {MAX_ITER}", "set_print_interval 10",
              "set_use_isomorphism 1", "start_solve", "set_dump_rounds 1", f"dump_result {out_json}"]
    return "\n".join(lines) + "\n"


def slug(flop):
    return flop.replace(" ", "")


def _solve_one(flop, ranges):
    out_json = RAW / f"{slug(flop)}.json"
    if out_json.exists():
        return f"{flop}: already solved"
    cfg = RAW / f"{slug(flop)}.txt"
    cfg.write_text(config(flop, out_json, ranges))
    t0 = time.time()
    log_path = RAW / f"{slug(flop)}.log"
    with open(log_path, "w") as logf:                     # streamed, so progress can be watched
        res = subprocess.run([str(SOLVER_DIR / "console_solver"), "-i", str(cfg), "-r", "resources"],
                             cwd=SOLVER_DIR, stdout=logf, stderr=subprocess.STDOUT)
    log = log_path.read_text(errors="replace").replace("\r", "\n")
    expl = [l for l in log.splitlines() if l.startswith("Total exploitability")]
    return f"{flop}: {time.time() - t0:.0f}s, exit {res.returncode}, {expl[-1] if expl else 'no exploitability line'}"


def solve(flops):
    from concurrent.futures import ThreadPoolExecutor
    d = np.load(POLICIES)
    ranges = {"btn": d["btn_range"], "bb": d["bb_range"]}
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for msg in ex.map(lambda f: _solve_one(f, ranges), flops):
            print(msg, flush=True)


RANKS = "23456789TJQKA"


def class_of(combo):
    """'AhKd' -> 'AKo'."""
    a, b = combo[:2], combo[2:]
    ra, rb = sorted([a[0], b[0]], key=RANKS.index, reverse=True)
    if ra == rb:
        return ra + rb
    return ra + rb + ("s" if a[1] == b[1] else "o")


def solver_flop(flop):
    """Per-combo flop strategies from a solved flop.

    Returns two DataFrames:
      btn: combo, class, check, small, big          (BTN after the BB checks)
      bb:  combo, class, size (0 small / 1 big), fold, call, raise   (BB facing each c-bet)
    """
    d = json.load(open(RAW / f"{slug(flop)}.json"))
    node = d["childrens"]["CHECK"] if d["actions"] == ["CHECK"] else d
    acts = node["strategy"]["actions"]
    bets = sorted([a for a in acts if a.startswith("BET")], key=lambda a: float(a.split()[1]))
    rows = []
    for combo, probs in node["strategy"]["strategy"].items():
        p = dict(zip(acts, probs))
        rows.append({"combo": combo, "class": class_of(combo), "check": p.get("CHECK", 0.0),
                     "small": p[bets[0]], "big": p[bets[1]]})
    btn = pd.DataFrame(rows)
    rows = []
    for k, bet in enumerate(bets):
        child = node["childrens"][bet]
        cacts = child["strategy"]["actions"]
        for combo, probs in child["strategy"]["strategy"].items():
            p = dict(zip(cacts, probs))
            raise_p = sum(v for a, v in p.items() if a.startswith("RAISE"))
            rows.append({"combo": combo, "class": class_of(combo), "size": k, "fold": p.get("FOLD", 0.0),
                         "call": p.get("CALL", 0.0), "raise": raise_p})
    return btn, pd.DataFrame(rows)


def model_flop(flop, btn, bb, pol):
    """The learned model's (experiment 13) strategy for the same combos on the same flop."""
    cards = [eval7.Card(c) for c in flop.split()]
    t = texture(cards)
    out_b, out_bb = btn.copy(), bb.copy()
    hole = lambda c: (eval7.Card(c[:2]), eval7.Card(c[2:]))
    bk_btn = np.array([bucket(hole(c), cards) for c in btn.combo])
    p1 = pol["p1"][t, bk_btn]                                  # check, small, big
    out_b[["check", "small", "big"]] = p1
    out_b["bucket"] = [BUCKETS[b] for b in bk_btn]
    bk_bb = np.array([bucket(hole(c), cards) for c in bb.combo])
    p2 = pol["p2"][t, bk_bb, bb["size"].to_numpy()]            # fold, call, raise
    out_bb[["fold", "call", "raise"]] = p2
    out_bb["bucket"] = [BUCKETS[b] for b in bk_bb]
    return out_b, out_bb


def compare():
    d = np.load(POLICIES)
    weights = {"btn": dict(zip(HANDS, d["btn_range"])), "bb": dict(zip(HANDS, d["bb_range"]))}
    m = np.load(MODEL)
    pol = {"p1": m["p1"], "p2": m["p2"]}
    flops = [f for f in FLOPS if (RAW / f"{slug(f)}.json").exists()]
    per_flop, per_bucket_btn, per_bucket_bb = [], [], []
    for flop in flops:
        s_btn, s_bb = solver_flop(flop)
        m_btn, m_bb = model_flop(flop, s_btn, s_bb, pol)
        s_btn["bucket"], s_bb["bucket"] = m_btn["bucket"], m_bb["bucket"]
        wb = s_btn["class"].map(weights["btn"]).to_numpy()
        wo = s_bb["class"].map(weights["bb"]).to_numpy()
        t = TEXTURES[texture([eval7.Card(c) for c in flop.split()])]
        row = {"flop": flop, "texture": " / ".join(t)}
        for src, b, o in (("solver", s_btn, s_bb), ("model", m_btn, m_bb)):
            for a in ("check", "small", "big"):
                row[f"{src}_btn_{a}"] = float((b[a] * wb).sum() / wb.sum())
            for k, name in enumerate(("small", "big")):
                sel = (o["size"] == k).to_numpy()
                for a in ("fold", "call", "raise"):
                    row[f"{src}_bb_{a}_vs_{name}"] = float((o[a][sel] * wo[sel]).sum() / wo[sel].sum())
        per_flop.append(row)
        for src, b, o in (("solver", s_btn, s_bb), ("model", m_btn, m_bb)):
            per_bucket_btn.append(b.assign(source=src, flop=flop, w=wb))
            per_bucket_bb.append(o.assign(source=src, flop=flop, w=wo))
    pf = pd.DataFrame(per_flop)
    pf.round(3).to_csv(OUT / "per_flop.csv", index=False)
    pb = pd.concat(per_bucket_btn)
    po = pd.concat(per_bucket_bb)
    agg = lambda df, cols: df.groupby(["source", "bucket"]).apply(
        lambda g: pd.Series({c: (g[c] * g.w).sum() / g.w.sum() for c in cols} | {"weight": g.w.sum()}))
    bb_btn = agg(pb, ["check", "small", "big"]).reset_index()
    bb_bb = po.groupby(["source", "size", "bucket"]).apply(
        lambda g: pd.Series({c: (g[c] * g.w).sum() / g.w.sum() for c in ("fold", "call", "raise")})).reset_index()
    bb_btn.round(3).to_csv(OUT / "btn_by_bucket.csv", index=False)
    bb_bb.round(3).to_csv(OUT / "bb_by_bucket.csv", index=False)
    return pf, bb_btn, bb_bb


def report():
    pf, bb_btn, bb_bb = compare()
    pf["solver_btn_bet"] = pf.solver_btn_small + pf.solver_btn_big
    pf["model_btn_bet"] = pf.model_btn_small + pf.model_btn_big
    pf["solver_big_share"] = pf.solver_btn_big / pf.solver_btn_bet
    pf["model_big_share"] = pf.model_btn_big / pf.model_btn_bet
    pf.round(3).to_csv(OUT / "per_flop.csv", index=False)

    # Agreement metrics across flops, for each decision.
    metrics = []
    pairs = [("BTN c-bets", "btn_bet"), ("BTN bets big (of its bets)", "big_share"), ("BTN bets small", "btn_small"),
             ("BTN bets big", "btn_big")]
    for size in ("small", "big"):
        for a in ("fold", "call", "raise"):
            pairs.append((f"BB {a}s vs {size} c-bet", f"bb_{a}_vs_{size}"))
    for name, col in pairs:
        sv, mv = pf[f"solver_{col}"], pf[f"model_{col}"]
        metrics.append({"decision": name, "solver_avg": sv.mean(), "model_avg": mv.mean(),
                        "mean_abs_diff": (sv - mv).abs().mean(), "correlation_across_flops": sv.corr(mv)})
    metrics = pd.DataFrame(metrics)
    metrics.round(3).to_csv(OUT / "agreement.csv", index=False)

    labels = list(pf.flop)
    grouped_bars(labels, {"solver": list(pf.solver_btn_bet), "model": list(pf.model_btn_bet)},
                 "BTN c-bet frequency by flop: TexasSolver vs the learned model", "C-bet frequency",
                 OUT / "cbet_by_flop.png")
    scatter_compare(pf.solver_btn_bet, pf.model_btn_bet, labels, "BTN c-bet frequency per flop",
                    "TexasSolver", "Learned model", OUT / "cbet_scatter.png")
    grouped_bars(labels, {"solver": list(pf.solver_bb_raise_vs_small), "model": list(pf.model_bb_raise_vs_small)},
                 "BB check-raise vs a 1/3-pot c-bet, by flop", "Check-raise frequency", OUT / "xr_by_flop.png")
    grouped_bars(labels, {"solver": list(pf.solver_bb_fold_vs_small), "model": list(pf.model_bb_fold_vs_small)},
                 "BB folds vs a 1/3-pot c-bet, by flop", "Fold frequency", OUT / "fold_by_flop.png")

    # By hand bucket (pooled over flops).
    b = bb_btn.pivot(index="bucket", columns="source", values=["check", "small", "big"]).reindex(BUCKETS).dropna()
    grouped_bars(list(b.index), {"solver": list(1 - b[("check", "solver")]), "model": list(1 - b[("check", "model")])},
                 "BTN c-bet frequency by hand (all flops)", "C-bet frequency", OUT / "cbet_by_bucket.png")
    grouped_bars(list(b.index), {"solver": list(b[("big", "solver")] / (1 - b[("check", "solver")])),
                                 "model": list(b[("big", "model")] / (1 - b[("check", "model")]))},
                 "Share of the BTN's c-bets that use the big (3/4 pot) size, by hand", "Big-size share",
                 OUT / "big_share_by_bucket.png")
    for k, size in enumerate(("1/3", "3/4")):
        o = bb_bb[bb_bb["size"] == k].pivot(index="bucket", columns="source",
                                            values=["fold", "call", "raise"]).reindex(BUCKETS).dropna()
        tag = "small" if k == 0 else "big"
        grouped_bars(list(o.index), {"solver": list(o[("raise", "solver")]), "model": list(o[("raise", "model")])},
                     f"BB check-raise vs a {size}-pot c-bet, by hand", "Check-raise frequency",
                     OUT / f"xr_by_bucket_{tag}.png")
        grouped_bars(list(o.index), {"solver": list(o[("fold", "solver")]), "model": list(o[("fold", "model")])},
                     f"BB fold vs a {size}-pot c-bet, by hand", "Fold frequency", OUT / f"fold_by_bucket_{tag}.png")

    pd.set_option("display.width", 250)
    cols = ["flop", "solver_btn_bet", "model_btn_bet", "solver_big_share", "model_big_share",
            "solver_bb_fold_vs_small", "model_bb_fold_vs_small", "solver_bb_raise_vs_small", "model_bb_raise_vs_small",
            "solver_bb_fold_vs_big", "model_bb_fold_vs_big", "solver_bb_raise_vs_big", "model_bb_raise_vs_big"]
    print(pf[cols].round(2).to_string(index=False))
    print(metrics.round(3).to_string(index=False))
    print(bb_btn.round(2).to_string(index=False))
    print(bb_bb.round(2).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "solve":
        solve(FLOPS)
    else:
        report()
