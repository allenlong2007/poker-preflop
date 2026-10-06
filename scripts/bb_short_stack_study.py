"""Experiment 19: the BB's ranges and sizes when stacks are short, by BTN open size.

Only the effective (smaller) stack matters heads-up, so a 25bb BB vs a 175bb BTN plays the same
game as a 25bb BTN vs a 175bb BB. This study covers 20-100bb effective vs BTN opens of 2.0-3.0bb.

1. Realization by stack. The preflop model's BB realization (r_oop = 0.915 from the pot-odds
   simulation) was fit at 100bb. The whole-hand models from experiment 18 (25 / 50 / 100bb, 2.25bb
   open) show how much better the BB does with shorter stacks. r_oop is scaled by that ratio and
   fit as a line in log(stack), capped at 1.0 (the BTN's realization).
2. Preflop. For every stack x open, every 3-bet size from 2.5x to 6x the open plus an all-in 3-bet;
   the best one for the BB is kept. Run with the fixed and the stack-adjusted realization.
3. Postflop. Whole-hand models (experiment 16 sizes) at 25 / 50bb for 2.0 / 2.25 / 3.0bb opens:
   how the BB plays the flop, turn and river.

Usage:
    python scripts/bb_short_stack_study.py preflop             # ~2 min
    python scripts/bb_short_stack_study.py postflop 25:2.0 ...  # train whole-hand models (cached)
    python scripts/bb_short_stack_study.py report
"""
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
from open_size_study import OPTS
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, heatmap, small_multiples, stacked_bars
from preflop.solver import Payoffs, solve

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("bb_short_stack_quick" if QUICK else "bb_short_stack")
CACHE = OUT / "cache"                          # gitignored
PRE = ROOT / "output" / "bb_short_stack" / "cache"       # preflop results (cheap, shared with quick mode)
for _d in (CACHE, PRE):
    _d.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 100_000)
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
EXP18 = ROOT / "output" / "donk_sizes_stacks" / "cache"
EXP16 = ROOT / "output" / "open_sizes" / "cache" / "model_2_25.npz"

STACKS = [20, 25, 30, 40, 50, 75, 100]
OPENS = [2.0, 2.25, 2.5, 2.75, 3.0]
POST = [(25, 2.0), (25, 2.25), (25, 3.0), (50, 2.0), (50, 2.25), (50, 3.0)]
CALIB_OPEN = 2.25
NOTABLE = ["AA", "AKo", "AQs", "AJo", "ATo", "A5s", "A2o", "KQo", "KTs", "K9o", "K5s", "QJs", "Q8o",
           "JTs", "J7s", "T9s", "98o", "76s", "65s", "54s", "TT", "77", "55", "22", "T4o", "83s"]
HAND_CLASSES = {"two pair+": [0, 1, 2], "overpair / top pair": [3, 4, 5], "middle / weak pair": [6, 7],
                "strong draw": [8], "weak draw": [9], "overcards / air": [10, 11]}
COMBOS = np.array([n_combos(h) for h in HANDS])
_EQ = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = _EQ["E"], _EQ["W"]


def tag(x):
    return f"{x:g}".replace(".", "_")


def fit_params():
    f = pd.read_csv(FIT, header=None, index_col=0)[1]
    return float(f["alpha"]), float(f["r_oop"])


def pct(p, w=None):
    w = COMBOS if w is None else COMBOS * w
    return float((p * w).sum() / max(w.sum(), 1e-12))


# ----------------------------------------------------------------------------- 1. realization

def implied_r_oop(stack):
    """BB realization that makes the preflop model's called-pot value match the whole-hand model."""
    a, _ = fit_params()
    p = np.load(EXP18 / f"preflop_2_25_{stack}.npz")
    model = EXP16 if stack == 100 else EXP18 / f"stack_{stack}.npz"
    target = float(np.load(model)["value_btn"]) - CALIB_OPEN          # BTN net per called pot
    w = W * p["btn_range"][:, None] * p["bb_range"][None, :]
    val = lambda r: float((Payoffs(E, CALIB_OPEN, 7.0, stack, 1.0, r, a).call_open * w).sum() / w.sum())
    lo, hi = 0.2, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if val(mid) > target else (lo, mid)
    return mid


def realization_by_stack():
    """r_oop per stack: the 100bb fit scaled by the whole-hand model's stack effect (cached)."""
    path = PRE / "realization.csv"
    if path.exists():
        return pd.read_csv(path)
    a, r100 = fit_params()
    pts = {s: implied_r_oop(s) for s in (25, 50, 100)}
    scaled = {s: r100 * v / pts[100] for s, v in pts.items()}
    slope, icpt = np.polyfit(np.log(list(scaled)), list(scaled.values()), 1)
    df = pd.DataFrame({"stack_bb": STACKS,
                       "implied_r_oop": [pts.get(s, np.nan) for s in STACKS],
                       "r_oop_scaled": [scaled.get(s, np.nan) for s in STACKS],
                       # capped at the BTN's r_ip = 1: out of position never realizes more than in position
                       "r_oop_used": [min(icpt + slope * np.log(s), 1.0) for s in STACKS]})
    df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------------- 2. preflop grid

def _solve_one(job):
    s, t, stack, a, r, iters = job
    return -solve(E, W, s, t, stack=stack, iters=iters, alpha=a, r_oop=r).btn_ev()


def threebet_sizes(s, stack):
    ts = np.arange(2.5, 6.001, 0.25) * s
    return np.append(ts[ts < stack - 1e-9], float(stack))     # last = all-in


def preflop_grid():
    path = PRE / "preflop_grid.npz"
    if path.exists():
        return
    a, r_fixed = fit_params()
    r_used = dict(zip(*realization_by_stack()[["stack_bb", "r_oop_used"]].values.T))
    configs = [(variant, stack, s, r) for variant, rr in (("fixed", None), ("by_stack", r_used))
               for stack in STACKS for s in OPENS for r in [r_fixed if rr is None else rr[stack]]]
    jobs = [(s, float(t), stack, a, r, 1500) for _, stack, s, r in configs for t in threebet_sizes(s, stack)]
    t0 = time.time()
    with Pool(12) as pool:
        evs = iter(pool.map(_solve_one, jobs))
    print(f"3-bet size grid: {len(jobs)} solves in {time.time() - t0:.0f}s", flush=True)
    out, rows, sizes = {}, [], []
    for variant, stack, s, r in configs:
        ts = threebet_sizes(s, stack)
        ev = np.array([next(evs) for _ in ts])
        for t, v in zip(ts, ev):
            sizes.append({"variant": variant, "stack_bb": stack, "open_bb": s, "threebet_to": t,
                          "all_in": t >= stack, "bb_ev": v})
        t = float(ts[int(np.argmax(ev))])
        res = solve(E, W, s, t, stack=stack, iters=5000, alpha=a, r_oop=r)
        fold, call, three = res.bb_vs_open.T
        raised = res.btn_open[:, 1]
        key = f"{variant}_{stack}_{tag(s)}"
        out.update({f"{key}_fold": fold, f"{key}_call": call, f"{key}_3bet": three, f"{key}_btn": raised})
        btn3 = res.btn_vs_3bet
        rows.append({
            "variant": variant, "stack_bb": stack, "open_bb": s, "r_oop": r,
            "btn_opens": pct(raised), "bb_fold": pct(fold), "bb_call": pct(call), "bb_3bet": pct(three),
            "threebet_to": t, "threebet_all_in": t >= stack,
            "bb_ev_best": float(ev.max()), "bb_ev_jam": float(ev[-1]),
            "bb_ev_best_small": float(ev[:-1].max()) if len(ev) > 1 else np.nan,
            "best_small_to": float(ts[:-1][int(np.argmax(ev[:-1]))]) if len(ev) > 1 else np.nan,
            # BTN response to the 3-bet; after an all-in 3-bet, "call" is the only way to continue
            "btn_fold_vs_3bet": pct(btn3[:, 0], raised),
            "btn_call_vs_3bet": pct(btn3[:, 1] + (btn3[:, 2] if t >= stack else 0), raised),
            "btn_jam_vs_3bet": 0.0 if t >= stack else pct(btn3[:, 2], raised),
            "bb_call_jam": np.nan if t >= stack else pct(res.bb_vs_jam[:, 1], three),
            "btn_ev": res.btn_ev()})
    np.savez(path, **out)
    pd.DataFrame(rows).to_csv(PRE / "preflop_summary.csv", index=False)
    pd.DataFrame(sizes).to_csv(PRE / "threebet_sizes.csv", index=False)


def ranges(variant, stack, s):
    d = np.load(PRE / "preflop_grid.npz")
    key = f"{variant}_{stack}_{tag(s)}"
    return {k: d[f"{key}_{k}"] for k in ("fold", "call", "3bet", "btn")}


# ----------------------------------------------------------------------------- 3. postflop models

def train_model(stack, s):
    path = CACHE / f"post_{stack}_{tag(s)}.npz"
    if path.exists():
        print(f"{stack}bb, {s}bb open: cached")
        return
    rg = ranges("by_stack", stack, s)
    print(f"== {stack}bb effective, {s}bb open", flush=True)
    t0 = time.time()
    sigma, _ = fg.learn(rg["btn"], rg["call"], OPTS, s=s, stack=stack, batches=BATCHES,
                        hands_per_batch=HANDS_PER_BATCH, seed=1)
    value, acc = fg.evaluate(rg["btn"], rg["call"], sigma, OPTS, s=s, stack=stack, n=EVAL_HANDS)
    out = {"value_btn": value, "line_keys": np.array(list(acc.lines.keys())),
           "line_vals": np.array(list(acc.lines.values()))}
    for (st, kind), v in sigma.items():
        out[f"strat_{st}_{kind}"] = acc.strat[(st, kind)]
        out[f"reach_{st}_{kind}"] = acc.reach[(st, kind)]
    np.savez_compressed(path, **out)
    print(f"   BB net {s - value:+.3f}bb per called pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


def node_freq(d, st, kind, mask=None):
    strat, reach = d[f"strat_{st}_{kind}"], d[f"reach_{st}_{kind}"]
    if mask is not None:
        strat, reach = strat * mask[..., None], reach * mask
    return strat.reshape(-1, strat.shape[-1]).sum(0) / max(float(reach.sum()), 1e-12)


def cell_mask(shape, lines=None, buckets=None, size=None):
    m = np.ones(shape)
    for axis, keep_idx in ((1, lines), (2, buckets), (3, size)):
        if keep_idx is None:
            continue
        keep = np.zeros(shape[axis], bool)
        keep[np.atleast_1d(keep_idx)] = True
        idx = [slice(None)] * len(shape)
        idx[axis] = ~keep
        m[tuple(idx)] = 0
    return m


BTN_LINES = [3, 4, 5]          # the BTN was the last aggressor (it bet and the BB called)


def postflop_rows(stack, s, d):
    shp = lambda st, k: d[f"reach_{st}_{k}"].shape
    row = {"stack_bb": stack, "open_bb": s, "spr_after_call": (stack - s) / (2 * s),
           "bb_net": s - float(d["value_btn"])}
    row["bb_donks"] = 1 - node_freq(d, 0, "bb_first")[0]
    f = node_freq(d, 0, "btn_vs_check")
    row["btn_cbet"] = 1 - f[0]
    for k, n in enumerate(("1/3", "3/4")):
        f = node_freq(d, 0, "bb_vs_bet", cell_mask(shp(0, "bb_vs_bet"), size=k))
        row.update({f"vs_{n}_fold": f[0], f"vs_{n}_call": f[1], f"vs_{n}_raise": f[2:].sum()})
    f = node_freq(d, 0, "bb_vs_bet")
    row["xr_big_share"] = f[3] / max(f[2] + f[3], 1e-12)                  # 4x raises among check-raises
    row["turn_lead_after_call"] = 1 - node_freq(d, 1, "bb_first", cell_mask(shp(1, "bb_first"), lines=BTN_LINES))[0]
    for st, name in ((1, "turn"), (2, "river")):
        f = node_freq(d, st, "bb_vs_bet", cell_mask(shp(st, "bb_vs_bet"), lines=BTN_LINES))
        row.update({f"{name}_vs_barrel_fold": f[0], f"{name}_vs_barrel_call": f[1],
                    f"{name}_vs_barrel_raise": f[2:].sum()})
    keys, lv = d["line_keys"], d["line_vals"].sum(1)
    allin = [i for i, k in enumerate(keys) if len(k.split("/")) < 3 and not k.endswith("f")]
    row["all_in_before_river"] = float(lv[allin].sum() / lv.sum())
    hands = []
    for cls, b in HAND_CLASSES.items():
        f = node_freq(d, 0, "bb_vs_bet", cell_mask(shp(0, "bb_vs_bet"), buckets=b, size=0))
        t = node_freq(d, 1, "bb_vs_bet", cell_mask(shp(1, "bb_vs_bet"), lines=BTN_LINES, buckets=b))
        dk = node_freq(d, 0, "bb_first", cell_mask(shp(0, "bb_first"), buckets=b))
        hands.append({"stack_bb": stack, "open_bb": s, "hand": cls, "donk": 1 - dk[0],
                      "vs_1/3_fold": f[0], "vs_1/3_call": f[1], "vs_1/3_raise": f[2:].sum(),
                      "turn_vs_barrel_fold": t[0], "turn_vs_barrel_raise": t[2:].sum()})
    return row, hands


# ----------------------------------------------------------------------------- report

def report():
    real = realization_by_stack()
    real.round(4).to_csv(OUT / "realization_by_stack.csv", index=False)
    pf = pd.read_csv(PRE / "preflop_summary.csv")
    pf.round(4).to_csv(OUT / "preflop_summary.csv", index=False)
    sizes = pd.read_csv(PRE / "threebet_sizes.csv")
    sizes.round(5).to_csv(OUT / "threebet_sizes.csv", index=False)
    main = pf[pf.variant == "by_stack"]

    # How the BB's preflop reaction changes with the open, one line per stack.
    panels = {}
    for col, title in (("bb_fold", "BB folds"), ("bb_call", "BB calls"), ("bb_3bet", "BB 3-bets")):
        panels[title] = {f"{st}bb": (list(g.open_bb), list(g[col] * 100)) for st, g in main.groupby("stack_bb")}
    small_multiples(panels, "BB vs the BTN's open size, by effective stack (% of hands)", "BTN open (bb)",
                    OUT / "bb_freq_by_open.png", legend_title="Effective stack")
    piv = main.pivot(index="stack_bb", columns="open_bb", values="threebet_to")
    heatmap(piv, "BB's best 3-bet size (bb); a size equal to the stack is all-in", "BTN open (bb)", "Effective stack (bb)",
            OUT / "bb_3bet_size.png")
    jam_gain = main.pivot(index="stack_bb", columns="open_bb", values="bb_ev_jam") \
        - main.pivot(index="stack_bb", columns="open_bb", values="bb_ev_best_small")
    heatmap(jam_gain * 1000, "All-in 3-bet minus the best smaller 3-bet (BB, milli-bb per hand)", "BTN open (bb)",
            "Effective stack (bb)", OUT / "bb_jam_vs_small.png", fmt="signed", diverging=True)
    for col, title, name in (("bb_3bet", "BB 3-bet frequency", "bb_3bet_freq"),
                             ("bb_fold", "BB fold frequency", "bb_fold_freq")):
        heatmap(main.pivot(index="stack_bb", columns="open_bb", values=col), title, "BTN open (bb)",
                "Effective stack (bb)", OUT / f"{name}.png", fmt="pct")
    for stack in (25, 50, 100):
        for s in OPENS:
            rg = ranges("by_stack", stack, s)
            r = main[(main.stack_bb == stack) & (main.open_bb == s)].iloc[0]
            jam = " all-in" if r.threebet_all_in else ""
            action_grid({"fold": rg["fold"], "call": rg["call"], "3-bet": rg["3bet"]},
                        f"BB vs a {s:g}bb open, {stack}bb effective (3-bet to {r.threebet_to:.1f}{jam})",
                        OUT / f"bb_range_{stack}bb_{tag(s)}.png",
                        subtitle=f"Fold {r.bb_fold:.0%}   Call {r.bb_call:.0%}   3-bet {r.bb_3bet:.0%}")
    # Notable hands: the BB's main action per stack and open.
    rows = []
    for stack in (20, 25, 30, 50, 100):
        for s in OPENS:
            rg = ranges("by_stack", stack, s)
            for h in NOTABLE:
                i = HANDS.index(h)
                f = {k: float(rg[k][i]) for k in ("fold", "call", "3bet")}
                rows.append({"stack_bb": stack, "open_bb": s, "hand": h, **f})
    nh = pd.DataFrame(rows)
    nh.round(3).to_csv(OUT / "notable_hands.csv", index=False)
    for s in (2.0, 3.0):
        g = nh[nh.open_bb == s]
        for k in ("3bet", "fold"):
            heatmap(g.pivot(index="hand", columns="stack_bb", values=k).reindex(NOTABLE),
                    f"BB {'3-bets' if k == '3bet' else 'folds'} vs a {s:g}bb open, by stack", "Effective stack (bb)",
                    "Hand", OUT / f"hands_{k}_{tag(s)}.png", fmt="pct", figsize=(6.5, 8))

    # Postflop.
    post, hands = [], []
    for stack, s in POST:
        path = CACHE / f"post_{stack}_{tag(s)}.npz"
        if path.exists():
            r, h = postflop_rows(stack, s, np.load(path))
            post.append(r)
            hands += h
    if os.path.exists(EXP16):
        r, h = postflop_rows(100, 2.25, np.load(EXP16))
        post.append(r)
        hands += h
    if post:
        pt = pd.DataFrame(post)
        pt.round(4).to_csv(OUT / "postflop_summary.csv", index=False)
        hd = pd.DataFrame(hands)
        hd.round(3).to_csv(OUT / "postflop_by_hand.csv", index=False)
        lab = [f"{r.stack_bb}bb, {r.open_bb:g} open" for r in pt.itertuples()]
        stacked_bars(lab, {"fold": list(pt["vs_1/3_fold"]), "call": list(pt["vs_1/3_call"]),
                           "raise": list(pt["vs_1/3_raise"])},
                     "BB vs a 1/3-pot c-bet, by stack and open", "Share of hands", OUT / "post_vs_cbet.png",
                     colors={"fold": "#e4e3df", "call": "#2a78d6", "raise": "#eb6834"})
        hd["config"] = [f"{a}bb / {b:g}" for a, b in zip(hd.stack_bb, hd.open_bb)]
        for col, title in (("vs_1/3_raise", "BB check-raises a 1/3-pot c-bet"),
                           ("vs_1/3_fold", "BB folds to a 1/3-pot c-bet"),
                           ("turn_vs_barrel_fold", "BB folds to a turn barrel")):
            piv = hd.pivot(index="hand", columns="config", values=col).reindex(list(HAND_CLASSES))
            heatmap(piv[list(dict.fromkeys(hd.config))], f"{title}, by hand (stack / open)", "Stack / BTN open",
                    "BB hand", OUT / f"post_hand_{col.replace('/', '').replace('vs_13', 'vs_third')}.png", fmt="pct",
                    figsize=(9, 4.4))
        print(pt.round(3).T.to_string())
        print(hd.round(2).to_string())

    pd.set_option("display.width", 220)
    cols = ["stack_bb", "open_bb", "r_oop", "btn_opens", "bb_fold", "bb_call", "bb_3bet", "threebet_to",
            "best_small_to", "bb_ev_jam", "bb_ev_best_small", "btn_fold_vs_3bet", "btn_jam_vs_3bet", "bb_call_jam"]
    print(real.round(3).to_string())
    for v in ("by_stack", "fixed"):
        print(f"--- preflop, {v} realization")
        print(pf[pf.variant == v][cols].round(3).to_string(index=False))
    print(f"Saved to {OUT}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "preflop":
        print(realization_by_stack().round(3).to_string())
        preflop_grid()
    elif cmd == "postflop":
        todo = [(int(a), float(b)) for a, b in (x.split(":") for x in sys.argv[2:])] or POST
        for stack, s in todo:
            train_model(stack, s)
    else:
        report()


if __name__ == "__main__":
    main()
