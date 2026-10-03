"""Experiment 12: the BB called the flop c-bet and the turn bet -- what should it do on the river?

Uses the flop policies (cbet_frequency_test.py) and turn policies (turn_study.py)
to decide which hands reach this river, then learns the river game
(preflop/rivergame.py) after both flop strategies (learned and fixed 50-75%).

Also checks two river benchmarks:
  * MDF: vs a 2/3-pot bet the BB must continue (call or raise) 60% of the time,
    or the BTN profits by betting any two cards.
  * Bluff share: a balanced bettor's 2/3-pot bets lose at showdown 28.6% of the
    time (bet / (pot + 2*bet)).

Usage:  python scripts/river_study.py      (~5 min on an idle multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.plots import grouped_bars, heatmap, stacked_bars
from preflop.rivergame import BET, BUCKETS, COLUMNS, TEXTURES, learn_and_play

OPEN = 2.25
OUT = ROOT / "output" / "river_study"
OUT.mkdir(parents=True, exist_ok=True)
FLOP = ROOT / "output" / "cbet_frequency" / "flop_policies.npz"
TURN = ROOT / "output" / "turn_study"
MIN_WEIGHT = 200
MDF = 1 / (1 + BET)                        # 60% vs a 2/3-pot bet
BALANCED_BLUFFS = BET / (1 + 2 * BET)      # 28.6%
COLORS = {"lead": "#eb6834", "check, BTN checks back": "#e4e3df", "check-call": "#2a78d6",
          "check-fold": "#9a9893", "check-raise": "#1baf7a"}


def wavg(x, w):
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def outcomes(h):
    chk = 1 - h.bb_lead
    return pd.DataFrame({"lead": h.bb_lead, "check, BTN checks back": chk * (1 - h.btn_bet),
                         "check-call": chk * h.btn_bet * h.bb_call, "check-fold": chk * h.btn_bet * h.bb_fold,
                         "check-raise": chk * h.btn_bet * h.bb_xr})


def report(h, label):
    oc = outcomes(h)
    h = h.assign(w_check=h.weight * (1 - h.bb_lead), w_bet=h.weight * (1 - h.bb_lead) * h.btn_bet)
    groups = [("all", "all", np.ones(len(h), bool))]
    groups += [("river card", v, (h.texture == k).to_numpy()) for k, v in enumerate(TEXTURES)]
    groups += [("bb hand", v, (h.bb_bucket == k).to_numpy()) for k, v in enumerate(BUCKETS)]
    rows = []
    for feat, v, m in groups:
        w, wb = h.weight[m], h.w_bet[m]
        row = {"scenario": label, "feature": feat, "value": v, "share_of_rivers": w.sum() / h.weight.sum(),
               "bb_lead_pct": wavg(h.bb_lead[m], w), "btn_bets_after_check_pct": wavg(h.btn_bet[m], h.w_check[m]),
               "bb_fold_vs_bet_pct": wavg(h.bb_fold[m], wb), "bb_call_vs_bet_pct": wavg(h.bb_call[m], wb),
               "bb_xr_vs_bet_pct": wavg(h.bb_xr[m], wb),
               # Of the BTN's bets, how many lose at showdown (bluffs)?
               "btn_bluff_share": wavg((h.btn_wins[m] == 0).astype(float), wb * 1.0) if feat != "bb hand" else np.nan,
               # When the BB calls a bet, how often does it win?
               "bb_wins_when_calling": wavg(1 - h.btn_wins[m], wb * h.bb_call[m]),
               "btn_ev_bb_per_hand": wavg(h.share[m] * 2 * OPEN - OPEN, w)}
        for name in oc:
            row[name] = wavg(oc[name][m], w)
        rows.append(row)
    return pd.DataFrame(rows)


def grid(h, col, weight):
    rows = {}
    for k, v in enumerate(TEXTURES):
        m = h.texture == k
        rows[v] = [wavg(h[col][m & (h.bb_bucket == b)], weight[m & (h.bb_bucket == b)])
                   if weight[m & (h.bb_bucket == b)].sum() >= MIN_WEIGHT else np.nan for b in range(len(BUCKETS))]
    g = pd.DataFrame(rows, index=BUCKETS).T
    return g.loc[:, g.notna().any()]


def main():
    f = np.load(FLOP)
    summaries = []
    for key, tag, label in (("eq", "learned", "learned flop c-bets"), ("fx", "fixed", "fixed 50-75% flop c-bets")):
        t = np.load(TURN / f"turn_policy_{tag}.npz")
        turn_pol = (t["arr_0"], t["arr_1"], t["arr_2"])
        cache = OUT / f"hands_{tag}.npz"          # local cache (gitignored); delete to re-learn
        if cache.exists():
            hands = np.load(cache)["hands"]
        else:
            print(f"River game after {label}:")
            out = learn_and_play(f["btn_range"], f["bb_range"], (f[f"{key}_cbet"], f[f"{key}_bb"]), turn_pol, s=OPEN)
            hands = out["hands"]
            np.savez_compressed(cache, hands=hands)
            np.savez(OUT / f"river_policy_{tag}.npz", *out["policy"])
        h = pd.DataFrame(hands, columns=COLUMNS)
        summary = report(h, label)
        summaries.append(summary)
        lines = summary.set_index(["feature", "value"])
        rc = lines.loc["river card"]
        stacked_bars(list(rc.index), {k: rc[k] for k in COLORS},
                     f"What the BB does on the river, by river card ({label})", "Share of rivers",
                     OUT / f"bb_river_lines_by_card_{tag}.png", colors=COLORS)
        bh = lines.loc["bb hand"].dropna(subset=["bb_lead_pct"])
        bh = bh[bh.share_of_rivers >= 0.005]
        stacked_bars(list(bh.index), {k: bh[k] for k in COLORS},
                     f"What the BB does on the river, by hand ({label})", "Share of rivers",
                     OUT / f"bb_river_lines_by_hand_{tag}.png", colors=COLORS)
        w_bet = h.weight * (1 - h.bb_lead) * h.btn_bet
        for col, weight, title, fname in (("bb_lead", h.weight, "BB leads the river", "lead"),
                                          ("bb_xr", w_bet, "BB check-raises the river (of BTN bets)", "xr"),
                                          ("bb_fold", w_bet, "BB folds to a river bet", "fold")):
            g = grid(h, col, weight)
            g.round(3).to_csv(OUT / f"bb_{fname}_heatmap_{tag}.csv")
            heatmap(g, f"{title}: river card x hand ({label})", "BB hand on the river", "River card",
                    OUT / f"bb_{fname}_heatmap_{tag}.png", fmt="pct", figsize=(9, 4.4))
        show = summary[summary.feature != "bb hand"]
        print(f"\n{label}")
        print(show[["value", "share_of_rivers", "bb_lead_pct", "btn_bets_after_check_pct", "bb_fold_vs_bet_pct",
                    "bb_call_vs_bet_pct", "bb_xr_vs_bet_pct", "btn_bluff_share", "bb_wins_when_calling"]]
              .round(3).to_string(index=False))
        print(summary[summary.feature == "bb hand"][["value", "share_of_rivers", "bb_lead_pct", "bb_fold_vs_bet_pct",
              "bb_call_vs_bet_pct", "bb_xr_vs_bet_pct", "bb_wins_when_calling"]].round(3).to_string(index=False))

    allsum = pd.concat(summaries)
    allsum.round(4).to_csv(OUT / "summary.csv", index=False)
    a = allsum[allsum.feature.isin(["all", "river card"])]
    learned, fixed = a[a.scenario == "learned flop c-bets"], a[a.scenario == "fixed 50-75% flop c-bets"]
    grouped_bars(list(learned.value),
                 {"BB defends vs a bet (learned flop)": list(1 - learned.bb_fold_vs_bet_pct),
                  "BB defends vs a bet (fixed 50-75% flop)": list(1 - fixed.bb_fold_vs_bet_pct)},
                 f"BB river defense vs a 2/3-pot bet (MDF = {MDF:.0%})", "Call or check-raise",
                 OUT / "defense_vs_mdf.png")
    grouped_bars(list(learned.value),
                 {"learned flop": list(learned.btn_bluff_share), "fixed 50-75% flop": list(fixed.btn_bluff_share)},
                 f"Share of BTN river bets that are bluffs (balanced = {BALANCED_BLUFFS:.1%})",
                 "Bets that lose at showdown", OUT / "btn_bluff_share.png")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
