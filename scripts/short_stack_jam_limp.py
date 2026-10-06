"""Experiment 20: BTN open-jams and limps for stacks of 25bb and below.

Uses preflop/shortstack.py: the BTN may fold, limp, raise to s or go all-in. Vs a limp the BB checks,
raises to r or goes all-in; vs a raise it folds, calls or 3-bets all-in.

For each stack: the BB picks its best raise-vs-limp size r for every open size s, and the BTN picks
the open size that earns the most given that. Then the same stack is re-solved with limps and/or
open-jams removed to measure what each option is worth.

Usage:  python scripts/short_stack_jam_limp.py        (~3 min)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS, n_combos
from preflop.plots import ACTION_COLORS, action_grid, heatmap, small_multiples, stacked_bars
from preflop.shortstack import solve_short

STACKS = [6, 8, 10, 12, 15, 17, 20, 25]
OPENS = [2.0, 2.25, 2.5, 3.0]
ISO = [2.5, 3.0, 3.5, 4.0, 5.0]
ITERS_GRID, ITERS_FINAL = 1500, 4000
OFF_PATH = 0.005     # a BTN action used less than this is treated as never taken
R_OOP = 1.0          # experiment 19: the BB's stack-adjusted realization is 1.0 at 25bb and below
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
OUT = ROOT / "output" / "short_stack"
OUT.mkdir(parents=True, exist_ok=True)
_EQ = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = _EQ["E"], _EQ["W"]
COMBOS = np.array([n_combos(h) for h in HANDS])
ALPHA = float(pd.read_csv(FIT, header=None, index_col=0)[1]["alpha"])
VARIANTS = {"full": (True, True), "no limp": (False, True), "no open-jam": (True, False),
            "raise only": (False, False)}
for k, c in {"limp": "#86b6ef", "raise": "#2a78d6", "all-in": "#eb6834", "check": "#86b6ef"}.items():
    ACTION_COLORS.setdefault(k, c)


def pct(p, w=None):
    w = COMBOS if w is None else COMBOS * w
    return float((p * w).sum() / max(w.sum(), 1e-12))


def _ev(job):
    stack, s, r, limp, jam, iters = job
    return solve_short(E, W, s, r, stack, iters, r_oop=R_OOP, alpha=ALPHA, limp=limp, jam=jam).btn_ev()


def best_sizes(evs, stack):
    """BB's best raise-vs-limp size for each open, then the BTN's best open given that."""
    best = {}
    for s in OPENS:
        if s >= stack:
            continue
        rs = [r for r in ISO if r < stack]
        r = min(rs, key=lambda r: evs[(s, r)])           # BB minimises the BTN's EV
        best[s] = (r, evs[(s, r)])
    s = max(best, key=lambda s: best[s][1])
    return s, best[s][0], best


def main():
    grid = [(st, s, r) for st in STACKS for s in OPENS if s < st for r in ISO if r < st]
    with Pool(12) as pool:
        evs = dict(zip(grid, pool.map(_ev, [(st, s, r, True, True, ITERS_GRID) for st, s, r in grid])))
        chosen = {}
        for st in STACKS:
            s, r, per_open = best_sizes({(s, r): evs[(x, s, r)] for x, s, r in grid if x == st}, st)
            chosen[st] = (s, r, per_open)
        jobs = [(st, chosen[st][0], chosen[st][1], lp, jm, ITERS_FINAL)
                for st in STACKS for lp, jm in VARIANTS.values()]
        variant_ev = dict(zip([(st, v) for st in STACKS for v in VARIANTS], pool.map(_ev, jobs)))

    rows, hand_rows = [], []
    for st in STACKS:
        s, r, per_open = chosen[st]
        res = solve_short(E, W, s, r, st, ITERS_FINAL, r_oop=R_OOP, alpha=ALPHA)
        root = res.actions("btn_root")
        vs_limp, vs_raise = res.actions("bb_vs_limp"), res.actions("bb_vs_raise")
        vs_jam, vs_iso = res.actions("bb_vs_open_jam"), res.actions("btn_vs_iso") if r < st else None
        # Each node's frequencies are weighted by how often the deciding player's hands reach it.
        pb = lambda name: res.reach(name)[0]     # BTN decisions
        po = lambda name: res.reach(name)[1]     # BB decisions
        row = {"stack_bb": st, "best_open": s, "bb_raise_vs_limp_to": r,
               "btn_ev_mbb": 1000 * variant_ev[(st, "full")], "exploitability_mbb": 1000 * res.exploitability(),
               **{f"btn_{a}": pct(v) for a, v in root.items()},
               "bb_vs_limp_check": pct(vs_limp["check"], po("bb_vs_limp")),
               "bb_vs_limp_raise": pct(vs_limp.get("raise", 0 * COMBOS), po("bb_vs_limp")),
               "bb_vs_limp_jam": pct(vs_limp["all-in"], po("bb_vs_limp")),
               "bb_vs_raise_fold": pct(vs_raise["fold"], po("bb_vs_raise")),
               "bb_vs_raise_call": pct(vs_raise["call"], po("bb_vs_raise")),
               "bb_vs_raise_jam": pct(vs_raise["all-in"], po("bb_vs_raise")),
               "bb_calls_open_jam": pct(vs_jam["call"], po("bb_vs_open_jam")),
               "btn_calls_3bet_jam": pct(res.actions("btn_vs_3bet")["call"], pb("btn_vs_3bet")),
               "btn_calls_limp_jam": pct(res.actions("btn_vs_limp_jam")["call"], pb("btn_vs_limp_jam"))}
        if vs_iso is not None:
            reach = pb("btn_vs_iso")
            row.update({f"btn_vs_iso_{a}": pct(v, reach) for a, v in vs_iso.items()})
        # Nodes the BTN never sends the BB to have arbitrary strategies: blank them.
        if row["btn_limp"] < OFF_PATH:
            row.update({k: np.nan for k in row if k.startswith(("bb_vs_limp", "btn_vs_iso", "btn_calls_limp"))})
            row["bb_raise_vs_limp_to"] = np.nan
        if row["btn_raise"] < OFF_PATH:
            row.update({k: np.nan for k in row if k.startswith(("bb_vs_raise", "btn_calls_3bet"))})
            row["best_open"] = np.nan
        for v in VARIANTS:
            row[f"value_{v}"] = 1000 * variant_ev[(st, v)]
        for s2, (r2, ev2) in per_open.items():
            row[f"btn_ev_open_{s2:g}"] = 1000 * ev2
            row[f"bb_iso_vs_open_{s2:g}"] = r2
        rows.append(row)
        for i, h in enumerate(HANDS):
            hand_rows.append({"stack_bb": st, "hand": h, **{f"btn_{a}": float(v[i]) for a, v in root.items()},
                              "bb_call_jam": float(vs_jam["call"][i]),
                              **{f"bb_vs_limp_{a}": float(v[i]) for a, v in vs_limp.items()},
                              **{f"bb_vs_raise_{a}": float(v[i]) for a, v in vs_raise.items()}})
        if st in (8, 10, 12, 15, 20):
            sub = lambda d: "   ".join(f"{a.capitalize()} {pct(v):.0%}" for a, v in d.items())
            action_grid(root, f"BTN at {st}bb effective (raise to {s:g}bb)", OUT / f"btn_{st}bb.png",
                        subtitle=sub(root))
            action_grid({"fold": vs_jam["fold"], "call": vs_jam["call"]}, f"BB vs a BTN all-in, {st}bb",
                        OUT / f"bb_vs_jam_{st}bb.png",
                        subtitle=f"Calls {row['bb_calls_open_jam']:.0%} of hands that reach it")
            if row["btn_limp"] >= OFF_PATH:
                action_grid(vs_limp, f"BB vs a BTN limp, {st}bb (raise to {r:g}bb)", OUT / f"bb_vs_limp_{st}bb.png",
                            subtitle=f"Check {row['bb_vs_limp_check']:.0%}   Raise {row['bb_vs_limp_raise']:.0%}   "
                                     f"All-in {row['bb_vs_limp_jam']:.0%}")
            if row["btn_raise"] >= OFF_PATH:
                action_grid({"fold": vs_raise["fold"], "call": vs_raise["call"], "3-bet": vs_raise["all-in"]},
                            f"BB vs a {s:g}bb raise, {st}bb (3-bet = all-in)", OUT / f"bb_vs_raise_{st}bb.png",
                            subtitle=f"Fold {row['bb_vs_raise_fold']:.0%}   Call {row['bb_vs_raise_call']:.0%}   "
                                     f"All-in {row['bb_vs_raise_jam']:.0%}")

    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(hand_rows).round(3).to_csv(OUT / "by_hand.csv", index=False)

    stacked_bars([f"{st}bb" for st in df.stack_bb],
                 {a: list(df[f"btn_{a}"]) for a in ("fold", "limp", "raise", "all-in")},
                 "What the BTN does first, by effective stack", "Share of hands", OUT / "btn_actions_by_stack.png",
                 colors={"fold": "#e4e3df", "limp": "#86b6ef", "raise": "#2a78d6", "all-in": "#eb6834"})
    lim = df[df.btn_limp >= OFF_PATH]
    stacked_bars([f"{st}bb" for st in lim.stack_bb],
                 {"check": list(lim.bb_vs_limp_check), "raise": list(lim.bb_vs_limp_raise),
                  "all-in": list(lim.bb_vs_limp_jam)},
                 "BB vs a limp, by effective stack", "Share of hands facing a limp", OUT / "bb_vs_limp_by_stack.png",
                 colors={"check": "#86b6ef", "raise": "#2a78d6", "all-in": "#eb6834"})
    val = pd.DataFrame({v: df[f"value_full"] - df[f"value_{v}"] for v in ("no limp", "no open-jam", "raise only")})
    val.index = df.stack_bb
    val.columns = ["limping", "open-jamming", "both"]
    heatmap(val, "What each option is worth to the BTN (milli-bb per hand vs removing it)", "Option removed",
            "Effective stack (bb)", OUT / "option_value.png", fmt="num", figsize=(7, 5))
    small_multiples({"BB calls an all-in (% of hands facing it)": {"BB": (list(df.stack_bb), list(df.bb_calls_open_jam * 100))},
                     "BTN EV (milli-bb per hand)": {"BTN": (list(df.stack_bb), list(df.btn_ev_mbb))}},
                    "Short-stack summary", "Effective stack (bb)", OUT / "jam_call_and_ev.png")

    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
    print(val.round(1).to_string())
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
