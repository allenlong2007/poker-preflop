"""Experiment 18: donk sizes x open sizes, and the BTN as the short stack.

Part A (donk sizes). At BTN opens of 2.0 / 2.25 / 2.5 / 3.0bb (pots of 4-6bb), the BB may donk the
flop 1/4, 1/2, 3/4 or pot (other sizes as in experiment 16). The whole-hand model learns which sizes
it uses. A "what if" evaluation then plays out every flop option for every board and hand against
the BTN's actual responses, giving the value of donking each size vs checking.

Part B (short stacks). Only the effective (smaller) stack matters heads-up: a 50bb BTN vs a 150bb BB
plays a 50bb game. At 25 / 50 / 100bb effective (2.25bb open; 100bb = experiment 16's model):
the best open size, preflop ranges incl. all-ins, and the whole-hand model's BTN and BB strategies.

Usage:
    python scripts/donk_size_stack_study.py donk 2.0 2.25    # train part A for these opens (cached)
    python scripts/donk_size_stack_study.py stacks          # train part B (cached)
    python scripts/donk_size_stack_study.py report
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from open_size_study import OPTS, RAISES, SIZES, line_text
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.flopgame import TEXTURES as FLOP_TEXTURES
from preflop.plots import action_grid, grouped_bars, heatmap, line, multi_line, range_grid, stacked_bars
from preflop.sizing import bb_ev_grid, best_threebet
from preflop.solver import solve

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("donk_sizes_stacks_quick" if QUICK else "donk_sizes_stacks")
CACHE = OUT / "cache"                      # gitignored
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 100_000)
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
EXP16 = ROOT / "output" / "open_sizes" / "cache" / "model_2_25.npz"

OPENS = [2.0, 2.25, 2.5, 3.0]
DONK_SIZES = (1 / 4, 1 / 2, 3 / 4, 1.0)
DONK_NAMES = ["1/4", "1/2", "3/4", "pot"]
DONK_OPTS = {"donk": True, "raises": RAISES, "eval_all": {(0, "bb_first")},
             "sizes": ({"lead": DONK_SIZES, "bet": SIZES[0]["bet"]}, SIZES[1], SIZES[2])}
STACKS = [25, 50, 100]
STACK_OPEN = 2.25
OPEN_GRID = np.round(np.arange(2.0, 3.51, 0.25), 2)
GROUPS = [("A-high", 0, "A-high"), ("K-high", 0, "K-high"), ("Q/J-high", 0, "Q/J-high"),
          ("T-high or lower", 0, "T-high or lower"), ("paired", 1, "paired"), ("monotone", 2, "monotone"),
          ("connected", 3, "connected")]
HAND_CLASSES = {"two pair+": [0, 1, 2], "overpair / top pair": [3, 4, 5], "middle / weak pair": [6, 7],
                "draw": [8, 9], "overcards / air": [10, 11]}
GRAY, BLUE, ORANGE, DARK_ORANGE = "#e4e3df", "#2a78d6", "#f6a07a", "#c4501f"


def tag(x):
    return f"{x:g}".replace(".", "_")


def fit_params():
    f = pd.read_csv(FIT, header=None, index_col=0)[1]
    return float(f["alpha"]), float(f["r_oop"])


def preflop(s, stack):
    """Preflop equilibrium at open s and this stack, with the BB's best 3-bet size (cached)."""
    path = CACHE / f"preflop_{tag(s)}_{stack}.npz"
    if not path.exists():
        d = np.load(ROOT / "data" / "equity_matrix.npz")
        E, W = d["E"], d["W"]
        a, r = fit_params()
        ts = np.arange(2.5, 6.001, 0.25) * s
        ts = ts[ts < stack]
        evs = [-solve(E, W, s, t, stack=stack, iters=2000, alpha=a, r_oop=r).btn_ev() for t in ts]
        t, _, _ = best_threebet(ts, np.array(evs))
        res = solve(E, W, s, t, stack=stack, iters=5000, alpha=a, r_oop=r)
        combos = np.array([n_combos(h) for h in HANDS])
        pct = lambda p: float((p * combos).sum() / combos.sum())
        fold, call, three = res.bb_vs_open.T
        raised = res.btn_open[:, 1]
        np.savez(path, btn_range=raised, bb_range=call, bb_fold_h=fold, bb_call_h=call, bb_3bet_h=three,
                 btn_vs_3bet=res.btn_vs_3bet, bb_vs_jam=res.bb_vs_jam, threebet_to=t, btn_ev=res.btn_ev(),
                 btn_raise=pct(raised), bb_fold=pct(fold), bb_call=pct(call), bb_3bet=pct(three),
                 btn_fold_vs_3bet=float((res.btn_vs_3bet[:, 0] * raised * combos).sum() / (raised * combos).sum()),
                 btn_jam_share=float((res.btn_vs_3bet[:, 2] * raised * combos).sum() / (raised * combos).sum()),
                 bb_call_jam=float((res.bb_vs_jam[:, 1] * three * combos).sum() / max((three * combos).sum(), 1e-9)))
    return np.load(path)


def train_model(name, opts, s, stack):
    path = CACHE / f"{name}.npz"
    if path.exists():
        print(f"{name}: cached")
        return
    p = preflop(s, stack)
    print(f"== {name}: open {s}bb, {stack}bb effective", flush=True)
    t0 = time.time()
    sigma, _ = fg.learn(p["btn_range"], p["bb_range"], opts, s=s, stack=stack, batches=BATCHES,
                        hands_per_batch=HANDS_PER_BATCH, seed=1)
    value, acc = fg.evaluate(p["btn_range"], p["bb_range"], sigma, opts, s=s, stack=stack, n=EVAL_HANDS)
    out = {"value_btn": value, "line_keys": np.array(list(acc.lines.keys())),
           "line_vals": np.array(list(acc.lines.values()))}
    for (st, kind), v in sigma.items():
        out[f"sigma_{st}_{kind}"] = v
        out[f"strat_{st}_{kind}"] = acc.strat[(st, kind)]
        out[f"reach_{st}_{kind}"] = acc.reach[(st, kind)]
        out[f"avals_{st}_{kind}"] = acc.avals[(st, kind)]
    np.savez_compressed(path, **out)
    print(f"   {name}: BB net {s - value:+.3f}bb per called pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ----------------------------------------------------------------------------- report helpers

def node_freq(d, st, kind, mask=None):
    strat, reach = d[f"strat_{st}_{kind}"], d[f"reach_{st}_{kind}"]
    if mask is not None:
        strat, reach = strat * mask[..., None], reach * mask
    return strat.reshape(-1, strat.shape[-1]).sum(0) / max(float(reach.sum()), 1e-12), float(reach.sum())


def cell_mask(shape, tex=None, lines=None, buckets=None, size=None):
    m = np.ones(shape)
    for axis, keep_idx in ((0, tex), (1, lines), (2, buckets), (3, size)):
        if keep_idx is None:
            continue
        keep = np.zeros(shape[axis], bool)
        keep[np.atleast_1d(keep_idx)] = True
        idx = [slice(None)] * len(shape)
        idx[axis] = ~keep
        m[tuple(idx)] = 0
    return m


def group_tex(k, v):
    return [t for t, tex in enumerate(FLOP_TEXTURES) if tex[k] == v]


# ----------------------------------------------------------------------------- report: part A

def report_donk():
    rows, gain_rows = [], []
    for s in OPENS:
        path = CACHE / f"donk_{tag(s)}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        st, rc, av = d["strat_0_bb_first"], d["reach_0_bb_first"], d["avals_0_bb_first"]
        f, _ = node_freq(d, 0, "bb_first")
        rows.append({"open_bb": s, "pot_bb": 2 * s, "bb_net": s - float(d["value_btn"]), "bb_donks": 1 - f[0],
                     **{f"donk {n}": f[1 + j] for j, n in enumerate(DONK_NAMES)}})
        # What-if: BB's gain (bb per called pot) from donking size j instead of checking, per group of cells.
        for gname, k, v in GROUPS + [("all flops", None, None)]:
            for cname, bs in list(HAND_CLASSES.items()) + [("all hands", None)]:
                m = cell_mask(rc.shape, tex=group_tex(k, v) if k is not None else None, buckets=bs)
                r = (rc * m).sum()
                if r < 50:
                    continue
                vals = (av * m[..., None]).reshape(-1, av.shape[-1]).sum(0) / r        # BTN units
                gain = -(vals[1:] - vals[0])                                           # BB gain vs checking
                gain_rows.append({"open_bb": s, "board": gname, "bb_hand": cname, "weight": r,
                                  **{f"gain {n}": g for n, g in zip(DONK_NAMES, gain)},
                                  "best": DONK_NAMES[int(np.argmax(gain))] if gain.max() > 0 else "check"})
    if not rows:
        return
    freq = pd.DataFrame(rows)
    freq.round(4).to_csv(OUT / "donk_frequency_by_open.csv", index=False)
    gains = pd.DataFrame(gain_rows)
    gains.round(4).to_csv(OUT / "donk_gain.csv", index=False)
    labels = [f"{s:g}bb (pot {2 * s:g})" for s in freq.open_bb]
    heatmap(pd.DataFrame({n: list(freq[f"donk {n}"]) for n in DONK_NAMES}, index=labels),
            "How often the BB donks each size (learned), by open size", "Donk size", "BTN open",
            OUT / "donk_freq_by_open.png", fmt="pct", figsize=(7, 3.6))
    allh = gains[(gains.bb_hand == "all hands")]
    for s in freq.open_bb:
        g = allh[allh.open_bb == s].set_index("board")
        heatmap(g[[f"gain {n}" for n in DONK_NAMES]].rename(columns=lambda c: c[5:]) * 100,
                f"BB gain from donking instead of checking (bb per 100 called pots), {s:g}bb open",
                "Donk size", "Board", OUT / f"donk_gain_board_{tag(s)}.png", fmt="signed", figsize=(7, 4.6),
                diverging=True)
        h = gains[(gains.open_bb == s) & (gains.board == "all flops")].set_index("bb_hand")
        heatmap(h[[f"gain {n}" for n in DONK_NAMES]].rename(columns=lambda c: c[5:]) * 100,
                f"BB gain from donking instead of checking, by hand (bb per 100 called pots), {s:g}bb open",
                "Donk size", "BB hand", OUT / f"donk_gain_hand_{tag(s)}.png", fmt="signed", figsize=(7, 4),
                diverging=True)
    # Where the learned strategy actually donks (any size), by board x hand, and how often donking is
    # "break-even": the best donk size within 1bb per 100 called pots of checking in that spot.
    used_rows = []
    for s in freq.open_bb:
        d = np.load(CACHE / f"donk_{tag(s)}.npz")
        st, rc, av = d["strat_0_bb_first"], d["reach_0_bb_first"], d["avals_0_bb_first"]
        safe = np.maximum(rc, 1e-12)[..., None]
        cell_gain = -(av[..., 1:] - av[..., :1]) / safe                      # BB gain per cell and size
        near = (cell_gain.max(-1) >= -0.01).astype(float)                     # within 1bb / 100 of checking
        grid = {}
        for gname, k, v in GROUPS:
            row = {}
            for cname, bs in HAND_CLASSES.items():
                m = cell_mask(rc.shape, tex=group_tex(k, v), buckets=bs)
                r = (rc * m).sum()
                if r < 50:
                    continue
                row[cname] = float((st[..., 1:].sum(-1) * m).sum() / r)
                used_rows.append({"open_bb": s, "board": gname, "bb_hand": cname, "donk_freq": row[cname],
                                  "break_even_share": float((near * rc * m).sum() / r),
                                  **{f"size {n}": float((st[..., 1 + j] * m).sum() / max((st[..., 1:].sum(-1) * m).sum(), 1e-12))
                                     for j, n in enumerate(DONK_NAMES)}})
            grid[gname] = row
        heatmap(pd.DataFrame(grid).T, f"How often the BB donks (any size), board x hand, {s:g}bb open",
                "BB hand on the flop", "Board", OUT / f"donk_used_{tag(s)}.png", fmt="pct", figsize=(8, 4.6))
    pd.DataFrame(used_rows).round(3).to_csv(OUT / "donk_where_used.csv", index=False)
    best = allh.pivot(index="board", columns="open_bb", values="best")
    best.to_csv(OUT / "best_donk_size.csv")
    # BTN response to each donk size.
    resp = []
    for s in freq.open_bb:
        d = np.load(CACHE / f"donk_{tag(s)}.npz")
        for j, n in enumerate(DONK_NAMES):
            m = cell_mask(d["reach_0_btn_vs_lead"].shape, size=j)
            f, w = node_freq(d, 0, "btn_vs_lead", m)
            resp.append({"open_bb": s, "donk": n, "btn_fold": f[0], "btn_call": f[1], "btn_raise": f[2] + f[3]})
    pd.DataFrame(resp).round(3).to_csv(OUT / "btn_vs_donk_size.csv", index=False)
    print(freq.round(3).to_string(index=False))
    print(allh[allh.board == "all flops"].round(4).to_string(index=False))
    print(best.to_string())
    print(pd.DataFrame(resp).round(2).to_string(index=False))


# ----------------------------------------------------------------------------- report: part B

def report_stacks():
    rows = []
    for stack in STACKS:
        p = preflop(STACK_OPEN, stack)
        path = EXP16 if stack == 100 else CACHE / f"stack_{stack}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        row = {"stack_bb": stack, "spr_after_call": (stack - STACK_OPEN) / (2 * STACK_OPEN),
               "btn_opens": float(p["btn_raise"]), "bb_fold": float(p["bb_fold"]), "bb_call": float(p["bb_call"]),
               "bb_3bet": float(p["bb_3bet"]), "bb_3bet_to": float(p["threebet_to"]),
               "btn_folds_to_3bet": float(p["btn_fold_vs_3bet"]), "btn_jams_vs_3bet": float(p["btn_jam_share"]),
               "bb_calls_jam": float(p["bb_call_jam"]), "bb_net_postflop": STACK_OPEN - float(d["value_btn"])}
        f, _ = node_freq(d, 0, "btn_vs_check")
        row.update({"btn_flop_check": f[0], "btn_cbet_1/3": f[1], "btn_cbet_3/4": f[2]})
        for k, n in enumerate(("1/3", "3/4")):
            f, _ = node_freq(d, 0, "bb_vs_bet", cell_mask(d["reach_0_bb_vs_bet"].shape, size=k))
            row.update({f"bb_vs_{n}_fold": f[0], f"bb_vs_{n}_call": f[1], f"bb_vs_{n}_raise": f[2] + f[3]})
        f, _ = node_freq(d, 0, "bb_first")
        row["bb_donks"] = 1 - f[0]
        f, _ = node_freq(d, 1, "btn_vs_check", cell_mask(d["reach_1_btn_vs_check"].shape, lines=[3, 4, 5]))
        row["btn_turn_barrel"] = 1 - f[0]
        f, _ = node_freq(d, 2, "btn_vs_check", cell_mask(d["reach_2_btn_vs_check"].shape, lines=[3, 4, 5]))
        row["btn_river_bet_after_barrel"] = 1 - f[0]
        keys, lv = d["line_keys"], d["line_vals"].sum(1)
        allin = [i for i, k in enumerate(keys) if len(k.split("/")) < 3 and not k.endswith("f")]
        row["all_in_before_river"] = float(lv[allin].sum() / lv.sum())   # stacks all in on the flop / turn
        rows.append(row)
        range_grid(p["btn_range"], f"BTN opening range, {stack}bb effective ({float(p['btn_raise']):.0%} of hands)",
                   OUT / f"btn_open_{stack}bb.png")
        action_grid({"fold": p["bb_fold_h"], "call": p["bb_call_h"], "3-bet": p["bb_3bet_h"]},
                    f"BB vs a 2.25bb open, {stack}bb effective (3-bet to {float(p['threebet_to']):.1f})",
                    OUT / f"bb_preflop_{stack}bb.png",
                    subtitle=f"Fold {float(p['bb_fold']):.0%}   Call {float(p['bb_call']):.0%}   3-bet {float(p['bb_3bet']):.0%}")
    st = pd.DataFrame(rows)
    st.round(4).to_csv(OUT / "stacks_summary.csv", index=False)

    # By hand: BTN c-bet and the BB's response, per stack.
    hand_rows = []
    for stack in st.stack_bb:
        d = np.load(EXP16 if stack == 100 else CACHE / f"stack_{int(stack)}.npz")
        for b, name in enumerate(fg.BUCKETS):
            f, w = node_freq(d, 0, "btn_vs_check", cell_mask(d["reach_0_btn_vs_check"].shape, buckets=[b]))
            g, _ = node_freq(d, 0, "bb_vs_bet", cell_mask(d["reach_0_bb_vs_bet"].shape, buckets=[b], size=0))
            hand_rows.append({"stack_bb": stack, "hand": name, "btn_cbet": 1 - f[0], "btn_cbet_big": f[2],
                              "bb_fold_vs_1/3": g[0], "bb_raise_vs_1/3": g[2] + g[3]})
    hd = pd.DataFrame(hand_rows)
    hd.round(3).to_csv(OUT / "stacks_by_hand.csv", index=False)
    for col, title in (("btn_cbet", "BTN c-bet frequency by hand"), ("bb_raise_vs_1/3", "BB check-raise vs a 1/3 c-bet"),
                       ("bb_fold_vs_1/3", "BB fold vs a 1/3 c-bet")):
        piv = hd.pivot(index="hand", columns="stack_bb", values=col).reindex(fg.BUCKETS)
        piv.columns = [f"{int(c)}bb" for c in piv.columns]
        heatmap(piv, f"{title}, by effective stack", "Effective stack", "Hand", OUT / f"stacks_{col.replace('/', '')}.png",
                fmt="pct", figsize=(6, 6))
    # Preflop: best open size per stack (BB picks its best 3-bet size).
    a, r = fit_params()
    curves = {}
    path = CACHE / "open_curves.npz"
    if not path.exists():
        models = {stack: {"alpha": a, "r_oop": r, "stack": stack} for stack in STACKS}
        table = bb_ev_grid(OPEN_GRID, lambda s: np.arange(2.5, 6.001, 0.25) * s, models)
        np.savez(path, **{f"{stack}": np.array([-best_threebet(*table[(stack, s)])[1] for s in OPEN_GRID])
                          for stack in STACKS})
    c = np.load(path)
    curves = {f"{stack}bb": (list(OPEN_GRID), list(c[f"{stack}"] * 1000)) for stack in STACKS}
    multi_line(curves, "BTN profit by open size, by effective stack (preflop model)", "BTN open size (bb)",
               "BTN EV (milli-bb per hand)", OUT / "best_open_by_stack.png", legend_title="Effective stack")
    best_open = {stack: float(OPEN_GRID[int(np.argmax(c[f"{stack}"]))]) for stack in STACKS}
    # Most probable lines per stack.
    top = []
    for stack in st.stack_bb:
        d = np.load(EXP16 if stack == 100 else CACHE / f"stack_{int(stack)}.npz")
        keys, vals = d["line_keys"], d["line_vals"].sum(1)
        order = np.argsort(-vals)[:8]
        for i in order:
            top.append({"stack_bb": stack, "prob": vals[i] / vals.sum(), "line": line_text(keys[i])})
    pd.DataFrame(top).round(4).to_csv(OUT / "stacks_top_lines.csv", index=False)
    print(st.round(3).T.to_string())
    print("best open by stack:", best_open)
    print(hd.round(2).to_string(index=False))
    print(pd.DataFrame(top).round(3).to_string(index=False))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "donk":
        for s in [float(x) for x in sys.argv[2:]] or OPENS:
            train_model(f"donk_{tag(s)}", DONK_OPTS, s, 100)
    elif cmd == "stacks":
        for stack in STACKS:
            if stack != 100:
                train_model(f"stack_{stack}", OPTS, STACK_OPEN, stack)
    else:
        pd.set_option("display.width", 250)
        report_donk()
        report_stacks()
        print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
