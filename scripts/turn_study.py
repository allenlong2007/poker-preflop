"""Experiment 11: the BB called a flop c-bet -- what should it do on the turn?

Uses the two flop policies saved by cbet_frequency_test.py (learned c-bets, and the
fixed 50-75% c-bets) and learns the turn game for each (preflop/turngame.py).
Reports the BB's turn play -- lead, check-call, check-fold, check-raise -- by turn
card, flop wetness and hand strength.

Usage:  python scripts/turn_study.py      (~6 min on an idle multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.plots import grouped_bars, heatmap, stacked_bars
from preflop.turngame import BUCKETS, COLUMNS, FLOPS, TEXTURES, TURN_CARDS, learn_and_play

OPEN = 2.25
OUT = ROOT / "output" / "turn_study"
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = ROOT / "output" / "cbet_frequency" / "flop_policies.npz"
MIN_WEIGHT = 300           # minimum (reach-weighted) hands for a heatmap cell
COLORS = {"lead": "#eb6834", "check, BTN checks back": "#e4e3df", "check-call": "#2a78d6",
          "check-fold": "#9a9893", "check-raise": "#1baf7a"}


def wavg(x, w):
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def outcomes(h):
    """Per-hand probability of each BB turn line (sums to 1)."""
    chk = 1 - h.bb_lead
    return pd.DataFrame({"lead": h.bb_lead, "check, BTN checks back": chk * (1 - h.btn_bet),
                         "check-call": chk * h.btn_bet * h.bb_call, "check-fold": chk * h.btn_bet * h.bb_fold,
                         "check-raise": chk * h.btn_bet * h.bb_xr})


def features(h):
    tex = np.array(TEXTURES, dtype=object)
    h = h.copy()
    h["turn card"] = [tex[int(t)][0] for t in h.texture]
    h["flop"] = [tex[int(t)][1] for t in h.texture]
    h["bb hand"] = [BUCKETS[int(b)] for b in h.bb_bucket]
    h["w_vs_bet"] = h.weight * (1 - h.bb_lead) * h.btn_bet          # reach of "BB checks, BTN bets"
    return h


def report(h, label):
    rows = []
    groups = [("all", "all", np.ones(len(h), bool))]
    groups += [("turn card", v, (h["turn card"] == v).to_numpy()) for v in TURN_CARDS]
    groups += [("flop", v, (h["flop"] == v).to_numpy()) for v in FLOPS]
    groups += [("bb hand", v, (h["bb hand"] == v).to_numpy()) for v in BUCKETS]
    oc = outcomes(h)
    for feat, v, m in groups:
        w, wb = h.weight[m], h.w_vs_bet[m]
        row = {"scenario": label, "feature": feat, "value": v, "share_of_turns": w.sum() / h.weight.sum(),
               "bb_lead_pct": wavg(h.bb_lead[m], w), "btn_bets_after_check_pct": wavg(h.btn_bet[m], w * (1 - h.bb_lead[m])),
               "bb_fold_vs_bet_pct": wavg(h.bb_fold[m], wb), "bb_call_vs_bet_pct": wavg(h.bb_call[m], wb),
               "bb_xr_vs_bet_pct": wavg(h.bb_xr[m], wb),
               "btn_fold_vs_lead_pct": wavg(h.btn_fold_vs_lead[m], w * h.bb_lead[m]),
               "btn_ev_bb_per_hand": wavg(h.share[m] * 2 * OPEN - OPEN, w)}
        for name in oc:
            row[name] = wavg(oc[name][m], w)
        rows.append(row)
    return pd.DataFrame(rows)


def grid(h, col, weight):
    rows = {}
    for feat, values in (("turn card", TURN_CARDS), ("flop", FLOPS)):
        for v in values:
            m = h[feat] == v
            rows[v] = [wavg(h[col][m & (h.bb_bucket == b)], weight[m & (h.bb_bucket == b)])
                       if weight[m & (h.bb_bucket == b)].sum() >= MIN_WEIGHT else np.nan for b in range(len(BUCKETS))]
    g = pd.DataFrame(rows, index=BUCKETS).T
    return g.loc[:, g.notna().any()]


def main():
    d = np.load(POLICIES)
    rules = {"table": d["table"], "bluff_probs": d["bluff_probs"]}
    summaries = []
    for key, label in (("eq", "learned flop c-bets"), ("fx", "fixed 50-75% flop c-bets")):
        print(f"Turn game after {label}:")
        tag = "learned" if key == "eq" else "fixed"
        cache = OUT / f"hands_{tag}.npz"        # local cache (gitignored); delete to re-learn
        if cache.exists():
            hands = np.load(cache)["hands"]
        else:
            out = learn_and_play(d["btn_range"], d["bb_range"], (d[f"{key}_cbet"], d[f"{key}_bb"]), rules, s=OPEN)
            hands = out["hands"]
            np.savez_compressed(cache, hands=hands)
            np.savez(OUT / f"turn_policy_{tag}.npz", *out["policy"])
        h = features(pd.DataFrame(hands, columns=COLUMNS))
        summary = report(h, label)
        summaries.append(summary)
        lines = summary.set_index(["feature", "value"])
        tc = lines.loc["turn card"]
        stacked_bars(list(tc.index), {k: tc[k] for k in COLORS},
                     f"What the BB does on the turn, by turn card ({label})", "Share of turns",
                     OUT / f"bb_turn_lines_by_card_{tag}.png", colors=COLORS)
        bh = lines.loc["bb hand"].dropna(subset=["bb_lead_pct"])
        stacked_bars(list(bh.index), {k: bh[k] for k in COLORS},
                     f"What the BB does on the turn, by hand ({label})", "Share of turns",
                     OUT / f"bb_turn_lines_by_hand_{tag}.png", colors=COLORS)
        for col, weight, title, fname in (
                ("bb_lead", h.weight, "BB leads the turn", "lead"),
                ("bb_xr", h.w_vs_bet, "BB check-raises the turn (of BTN bets)", "xr"),
                ("bb_fold", h.w_vs_bet, "BB folds to a turn bet", "fold")):
            g = grid(h, col, weight)
            g.round(3).to_csv(OUT / f"bb_{fname}_heatmap_{tag}.csv")
            heatmap(g, f"{title}: turn card / flop x hand ({label})", "BB hand on the turn",
                    "Turn card / flop", OUT / f"bb_{fname}_heatmap_{tag}.png", fmt="pct", figsize=(9, 5.2))
        print(summary[summary.feature != "bb hand"][["value", "share_of_turns", "bb_lead_pct",
              "btn_bets_after_check_pct", "bb_fold_vs_bet_pct", "bb_call_vs_bet_pct", "bb_xr_vs_bet_pct"]]
              .round(2).to_string(index=False))
        print(summary[summary.feature == "bb hand"][["value", "share_of_turns", "bb_lead_pct", "bb_fold_vs_bet_pct",
              "bb_call_vs_bet_pct", "bb_xr_vs_bet_pct"]].round(2).to_string(index=False))

    allsum = pd.concat(summaries)
    allsum.round(4).to_csv(OUT / "summary.csv", index=False)
    tc = allsum[allsum.feature.isin(["all", "turn card"])]
    for col, title, fname in (("bb_lead_pct", "BB leads the turn", "compare_lead"),
                              ("bb_xr_vs_bet_pct", "BB check-raises a turn bet", "compare_xr"),
                              ("bb_fold_vs_bet_pct", "BB folds to a turn bet", "compare_fold")):
        grouped_bars(list(tc[tc.scenario == "learned flop c-bets"].value),
                     {s: list(tc[tc.scenario == s][col]) for s in tc.scenario.unique()},
                     f"{title}: after learned vs fixed 50-75% flop c-bets", "Frequency", OUT / f"{fname}.png")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
