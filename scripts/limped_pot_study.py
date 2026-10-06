"""Experiment 21: the whole-hand model on limped pots at 10-25bb.

Experiment 20's preflop tree valued limped pots with the realization formula only. Here the whole-hand
model (experiment 16 sizes) plays them out:

1. Ranges come from experiment 20's equilibrium: the BTN's limps vs the BB's checks (pot 2bb), and,
   for comparison, the BTN's raises vs the BB's calls (15-25bb, where the BTN raises).
2. Each whole-hand model gives the BTN's value per pot. Matching the preflop formula to it gives an
   implied BB realization (r_oop) for limped and for raised pots.
3. The preflop tree is re-solved with a limped-pot realization:
     relative -- raised pots keep r_oop = 1.0 (experiment 19), limped pots get 1.0 x the pooled ratio
                 (implied limped / implied raised, averaged over 15 / 20 / 25bb)
     absolute -- both limped and raised pots use the whole-hand model's implied r_oop (nearest stack)

Usage:
    python scripts/limped_pot_study.py train limp_10 limp_12 ...   (~15-20 min per model, cached)
    python scripts/limped_pot_study.py report
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
import short_stack_jam_limp as exp20
from bb_short_stack_study import cell_mask, node_freq
from open_size_study import OPTS
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.plots import grouped_bars, heatmap, stacked_bars
from preflop.shortstack import solve_short
from preflop.solver import Payoffs

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("limped_pots_quick" if QUICK else "limped_pots")
CACHE = OUT / "cache"                          # gitignored
CACHE.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 100_000)
E, W, ALPHA = exp20.E, exp20.W, exp20.ALPHA
COMBOS = np.array([n_combos(h) for h in HANDS])
LIMP_STACKS = [10, 12, 15, 20, 25]
RAISE_STACKS = [15, 20, 25]
MODELS = [f"limp_{s}" for s in LIMP_STACKS] + [f"raise_{s}" for s in RAISE_STACKS]
CLASSES = {"two pair+": [0, 1, 2], "overpair / top pair": [3, 4, 5], "middle / weak pair": [6, 7],
           "strong draw": [8], "weak draw": [9], "overcards / air": [10, 11]}
NO_AGGR, BTN_AGGR = [0, 1, 2], [3, 4, 5]


def pct(p, w=None):
    w = COMBOS if w is None else COMBOS * w
    return float((p * w).sum() / max(w.sum(), 1e-12))


def ranges():
    """Experiment 20's equilibrium at each stack: limp / check and raise / call ranges (cached)."""
    path = CACHE / "ranges.npz"
    if not path.exists():
        summ = pd.read_csv(ROOT / "output" / "short_stack" / "summary.csv").set_index("stack_bb")
        out = {}
        for st in sorted(set(LIMP_STACKS + RAISE_STACKS)):
            s = summ.best_open.get(st)
            s = 2.0 if pd.isna(s) else float(s)
            r = float(summ.bb_raise_vs_limp_to[st])
            res = solve_short(E, W, s, r, st, exp20.ITERS_FINAL, r_oop=exp20.R_OOP, alpha=ALPHA)
            out.update({f"{st}_open": s, f"{st}_btn_limp": res.actions("btn_root")["limp"],
                        f"{st}_bb_check": res.actions("bb_vs_limp")["check"],
                        f"{st}_btn_raise": res.actions("btn_root")["raise"],
                        f"{st}_bb_call": res.actions("bb_vs_raise")["call"]})
        np.savez(path, **out)
    return np.load(path)


def model_spec(name):
    kind, st = name.split("_")
    st = int(st)
    rg = ranges()
    if kind == "limp":
        return st, 1.0, rg[f"{st}_btn_limp"], rg[f"{st}_bb_check"]
    return st, float(rg[f"{st}_open"]), rg[f"{st}_btn_raise"], rg[f"{st}_bb_call"]


def train(name):
    path = CACHE / f"{name}.npz"
    if path.exists():
        print(f"{name}: cached")
        return
    st, p, btn, bb = model_spec(name)
    print(f"== {name}: pot {2 * p:g}bb, {st}bb effective", flush=True)
    t0 = time.time()
    sigma, _ = fg.learn(btn, bb, OPTS, s=p, stack=st, batches=BATCHES, hands_per_batch=HANDS_PER_BATCH, seed=1)
    value, acc = fg.evaluate(btn, bb, sigma, OPTS, s=p, stack=st, n=EVAL_HANDS)
    out = {"value_btn": value, "line_keys": np.array(list(acc.lines.keys())),
           "line_vals": np.array(list(acc.lines.values()))}
    for key in sigma:
        out[f"strat_{key[0]}_{key[1]}"] = acc.strat[key]
        out[f"reach_{key[0]}_{key[1]}"] = acc.reach[key]
    np.savez_compressed(path, **out)
    print(f"   BTN net {value - p:+.3f}bb per pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


def preflop_value(p, st, btn, bb, r_oop):
    w = W * btn[:, None] * bb[None, :]
    return float((Payoffs(E, p, p, st, 1.0, r_oop, ALPHA).call_open * w).sum() / w.sum())


def implied_r_oop(p, st, btn, bb, target):
    lo, hi = 0.05, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if preflop_value(p, st, btn, bb, mid) > target else (lo, mid)
    return mid


def postflop_stats(name, d):
    shp = lambda st, k: d[f"reach_{st}_{k}"].shape
    st, p, _, _ = model_spec(name)
    row = {"model": name, "pot_type": name.split("_")[0], "stack_bb": st, "pot_bb": 2 * p,
           "spr": (st - p) / (2 * p), "btn_net": float(d["value_btn"]) - p}
    row["btn_net_pct_pot"] = row["btn_net"] / (2 * p)
    row["bb_leads_flop"] = 1 - node_freq(d, 0, "bb_first")[0]
    f = node_freq(d, 0, "btn_vs_check")
    row.update({"btn_bets_flop": 1 - f[0], "btn_bet_small": f[1], "btn_bet_big": f[2]})
    f = node_freq(d, 0, "bb_vs_bet")
    row.update({"bb_vs_bet_fold": f[0], "bb_vs_bet_call": f[1], "bb_vs_bet_raise": f[2:].sum()})
    f = node_freq(d, 0, "btn_vs_lead")
    row.update({"btn_vs_lead_fold": f[0], "btn_vs_lead_call": f[1], "btn_vs_lead_raise": f[2:].sum()})
    # Turn after a checked-through flop: who bets?
    row["bb_leads_turn_after_xx"] = 1 - node_freq(d, 1, "bb_first", cell_mask(shp(1, "bb_first"), lines=NO_AGGR))[0]
    row["btn_bets_turn_after_xx"] = 1 - node_freq(d, 1, "btn_vs_check",
                                                  cell_mask(shp(1, "btn_vs_check"), lines=NO_AGGR))[0]
    keys, lv = d["line_keys"], d["line_vals"].sum(1)
    tot = lv.sum()
    row["checked_to_showdown"] = float(sum(v for k, v in zip(keys, lv) if k == "xx/xx/xx") / tot)
    row["all_in_before_river"] = float(sum(v for k, v in zip(keys, lv)
                                           if len(k.split("/")) < 3 and not k.endswith("f")) / tot)
    hands = []
    for cls, b in CLASSES.items():
        bet = node_freq(d, 0, "btn_vs_check", cell_mask(shp(0, "btn_vs_check"), buckets=b))
        lead = node_freq(d, 0, "bb_first", cell_mask(shp(0, "bb_first"), buckets=b))
        vs = node_freq(d, 0, "bb_vs_bet", cell_mask(shp(0, "bb_vs_bet"), buckets=b))
        hands.append({"model": name, "stack_bb": st, "pot_type": row["pot_type"], "hand": cls,
                      "btn_bets_flop": 1 - bet[0], "bb_leads_flop": 1 - lead[0],
                      "bb_folds_to_bet": vs[0], "bb_raises_bet": vs[2:].sum()})
    return row, hands


def report():
    rows, hands = [], []
    for name in MODELS:
        path = CACHE / f"{name}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        row, h = postflop_stats(name, d)
        st, p, btn, bb = model_spec(name)
        row["preflop_model_btn_net"] = preflop_value(p, st, btn, bb, exp20.R_OOP)
        row["implied_r_oop"] = implied_r_oop(p, st, btn, bb, row["btn_net"])
        rows.append(row)
        hands += h
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "postflop_summary.csv", index=False)
    hd = pd.DataFrame(hands)
    hd.round(3).to_csv(OUT / "postflop_by_hand.csv", index=False)

    # Limped-pot realization per stack. The per-stack limped / raised ratios are noisy (each pair of models
    # has different ranges), so the relative version uses their pooled average.
    lim = df[df.pot_type == "limp"].set_index("stack_bb")
    rai = df[df.pot_type == "raise"].set_index("stack_bb")
    ratio = (lim.implied_r_oop / rai.implied_r_oop).dropna()
    pooled = float(ratio.mean())
    near = lambda d, st: d[min(d, key=lambda k: abs(k - st))]
    r_rel = {st: exp20.R_OOP * pooled for st in exp20.STACKS}
    r_abs_limp = {st: near(lim.implied_r_oop.to_dict(), st) for st in exp20.STACKS}
    r_abs_raise = {st: near(rai.implied_r_oop.to_dict(), st) for st in exp20.STACKS}
    cal = pd.DataFrame({"stack_bb": LIMP_STACKS, "implied_limped": [lim.implied_r_oop[s] for s in LIMP_STACKS],
                        "implied_raised": [rai.implied_r_oop.get(s, np.nan) for s in LIMP_STACKS],
                        "limped_over_raised": [ratio.get(s, np.nan) for s in LIMP_STACKS]})
    cal.round(4).to_csv(OUT / "realization.csv", index=False)
    print(f"pooled limped / raised ratio {pooled:.3f}")

    # Re-solve experiment 20 with the whole-hand model's realization.
    base, _ = exp20.main(None, OUT / "resolve_baseline")
    rel, _ = exp20.main(r_rel, OUT / "resolve_relative")
    ab, _ = exp20.main(r_abs_limp, OUT / "resolve_absolute", r_raise=r_abs_raise)
    cmp_rows = []
    for label, t in (("experiment 20", base), ("relative", rel), ("absolute", ab)):
        for _, r in t.iterrows():
            cmp_rows.append({"version": label, "stack_bb": r.stack_bb, "fold": r.btn_fold, "limp": r.btn_limp,
                             "raise": r.btn_raise, "all-in": r["btn_all-in"],
                             "limp_value_mbb": r.value_full - r["value_no limp"], "btn_ev_mbb": r.btn_ev_mbb,
                             "bb_vs_limp_check": r.bb_vs_limp_check, "bb_vs_limp_raise": r.bb_vs_limp_raise,
                             "bb_vs_limp_jam": r.bb_vs_limp_jam})
    cmp = pd.DataFrame(cmp_rows)
    cmp.round(4).to_csv(OUT / "preflop_resolved.csv", index=False)

    # Charts.
    lab = [f"{s}bb" for s in lim.index]
    grouped_bars(lab, {"whole-hand model": list(lim.btn_net_pct_pot * 100),
                       "preflop formula (exp 20)": list(lim.preflop_model_btn_net / lim.pot_bb * 100)},
                 "BTN profit in limped pots, % of the 2bb pot", "BTN net (% of pot)", OUT / "limped_value.png",
                 pct=False)
    main = cmp[cmp.version == "relative"]
    stacked_bars([f"{st}bb" for st in main.stack_bb], {a: list(main[a]) for a in ("fold", "limp", "raise", "all-in")},
                 "BTN's first action with calibrated limped pots (relative)", "Share of hands",
                 OUT / "btn_actions_calibrated.png",
                 colors={"fold": "#e4e3df", "limp": "#86b6ef", "raise": "#2a78d6", "all-in": "#eb6834"})
    piv = cmp.pivot(index="stack_bb", columns="version", values="limp")[["experiment 20", "relative", "absolute"]]
    heatmap(piv, "How often the BTN limps, by limped-pot realization", "Version", "Effective stack (bb)",
            OUT / "limp_freq_versions.png", fmt="pct", figsize=(6.5, 5))
    piv = cmp.pivot(index="stack_bb", columns="version", values="limp_value_mbb")[["experiment 20", "relative", "absolute"]]
    heatmap(piv, "Value of limping to the BTN (milli-bb per hand)", "Version", "Effective stack (bb)",
            OUT / "limp_value_versions.png", fmt="num", figsize=(6.5, 5))
    hl = hd[hd.pot_type == "limp"]
    for col, title in (("btn_bets_flop", "BTN bets the flop when checked to, limped pots"),
                       ("bb_leads_flop", "BB leads the flop, limped pots"),
                       ("bb_folds_to_bet", "BB folds to a flop bet, limped pots")):
        heatmap(hl.pivot(index="hand", columns="stack_bb", values=col).reindex(list(CLASSES)), title,
                "Effective stack (bb)", "Hand", OUT / f"limped_{col}.png", fmt="pct", figsize=(7, 4.4))

    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
    print(cal.round(3).to_string())
    print(cmp.round(3).to_string())
    print(hd.round(2).to_string())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "train":
        for name in sys.argv[2:] or MODELS:
            train(name)
    else:
        report()


if __name__ == "__main__":
    main()
