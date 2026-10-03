"""Experiment 10: a Button that c-bets 75% on boards that favor it and 50% on boards that don't.

1. Run the learned flop game (experiment 8) at a 2.25bb open. From it, measure
   each board texture's "range advantage": the BTN's average equity vs the BB's
   calling range on that texture. Textures above the overall average favor the BTN.
2. Fix the BTN's c-bet frequency per texture -- 75% if the texture favors it,
   50% if not -- filling that frequency with hands in a simple priority order:
   value (two pair+, overpair, top pair), then draws, then air (bluffs), then two
   overcards, and weak pairs last. The BB (and the BTN's answer to a check-raise)
   learn their best response to it.
3. Compare both scenarios: BB response, check-raises, and who wins how much.

Saves both flop policies (output/cbet_frequency/flop_policies.npz) for the turn study.

Usage:  python scripts/cbet_frequency_test.py     (~5 min on an idle multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bb_study import solve_at
from preflop.flopgame import BUCKETS, COLUMNS, NB, NT, TEXTURES, learn_and_play
from preflop.plots import grouped_bars, stacked_bars

OPEN = 2.25
FAVORED, UNFAVORED = 0.75, 0.50
PRIORITY = ["two pair+", "overpair", "top pair, T+ kicker", "top pair, weak kicker", "draw", "air",
            "two overcards", "weaker pair"]
OUT = ROOT / "output" / "cbet_frequency"
OUT.mkdir(parents=True, exist_ok=True)
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"


def wavg(x, w):
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def fixed_policy(h, favored):
    """C-bet probability per (texture, bucket) that hits the target frequency on each texture."""
    pol = np.zeros((NT, NB))
    overall = np.bincount(h.btn_bucket.astype(int), minlength=NB) / len(h)
    for t in range(NT):
        m = h.texture == t
        share = np.bincount(h.btn_bucket[m].astype(int), minlength=NB) / m.sum() if m.sum() >= 200 else overall
        left = FAVORED if favored[t] else UNFAVORED
        for name in PRIORITY:
            b = BUCKETS.index(name)
            take = min(share[b], left)
            pol[t, b] = take / share[b] if share[b] > 0 else float(left > 0)
            left -= take
    return pol


def summarize(h, favored, label, s):
    pc = h.p_cbet
    fav = favored[h.texture.astype(int)]
    rows = []
    for name, m in (("all flops", np.ones(len(h), bool)), ("favor BTN", fav), ("favor BB", ~fav)):
        rows.append({"scenario": label, "boards": name, "flops_pct": m.mean(), "btn_cbet_pct": pc[m].mean(),
                     "bb_fold_pct": wavg(h.bb_fold[m], pc[m]), "bb_call_pct": wavg(h.bb_call[m], pc[m]),
                     "bb_xr_pct": wavg(h.bb_raise[m], pc[m]),
                     "btn_calls_xr_pct": wavg(h.btn_call_vs_xr[m], (pc * h.bb_raise)[m]),
                     "btn_ev_bb_per_hand": float((h.share[m] * 2 * s - s).mean())})
    return rows


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    f = pd.read_csv(FIT, header=None, index_col=0)[1]
    res, _ = solve_at(E, W, OPEN, float(f["alpha"]), float(f["r_oop"]))
    btn_r, bb_c = res.btn_open[:, 1], res.bb_vs_open[:, 1]

    print("Learned c-bets (both players adapt):")
    eq = learn_and_play(btn_r, bb_c, s=OPEN, rounds=60, n=600_000, seed=1)
    h_eq = pd.DataFrame(eq["hands"], columns=COLUMNS)

    # Range advantage per texture: BTN's average showdown equity vs the BB's range on it.
    by_t = h_eq.groupby("texture").btn_showdown.agg(["mean", "size"])
    overall = h_eq.btn_showdown.mean()
    favored = np.zeros(NT, bool)
    favored[by_t.index.astype(int)] = by_t["mean"].to_numpy() > overall
    pd.DataFrame({"texture": [" / ".join(TEXTURES[int(t)]) for t in by_t.index],
                  "flops_pct": by_t["size"] / len(h_eq), "btn_range_equity": by_t["mean"],
                  "favors": np.where(favored[by_t.index.astype(int)], "BTN", "BB")}
                 ).sort_values("btn_range_equity", ascending=False).round(3).to_csv(
        OUT / "texture_range_advantage.csv", index=False)
    print(f"BTN range equity on the flop: {overall:.3f} overall; "
          f"{favored[h_eq.texture.astype(int)].mean():.0%} of flops favor the BTN")

    print("Fixed c-bets (75% on BTN boards, 50% on BB boards; BB adapts):")
    pol = fixed_policy(h_eq, favored)
    fx = learn_and_play(btn_r, bb_c, s=OPEN, rounds=40, n=600_000, seed=1, fixed_cbet=pol)
    h_fx = pd.DataFrame(fx["hands"], columns=COLUMNS)

    np.savez(OUT / "flop_policies.npz", btn_range=btn_r, bb_range=bb_c, favored=favored,
             eq_cbet=eq["policy"][0], eq_bb=eq["policy"][1], eq_xr=eq["policy"][2],
             fx_cbet=fx["policy"][0], fx_bb=fx["policy"][1], fx_xr=fx["policy"][2],
             table=eq["rules"]["table"], bluff_probs=eq["rules"]["bluff_probs"])
    pd.DataFrame(pol, index=[" / ".join(t) for t in TEXTURES], columns=BUCKETS).round(3).to_csv(
        OUT / "fixed_cbet_policy.csv")

    summary = pd.DataFrame(summarize(h_eq, favored, "learned", OPEN) + summarize(h_fx, favored, "fixed 50-75%", OPEN))
    summary.round(4).to_csv(OUT / "summary.csv", index=False)

    rows = []
    for label, h in (("learned", h_eq), ("fixed 50-75%", h_fx)):
        for b, name in enumerate(BUCKETS):
            mb, bb = h.btn_bucket == b, h.bb_bucket == b
            rows.append({"scenario": label, "bucket": name, "btn_cbet_pct": h.p_cbet[mb].mean(),
                         "bb_fold_pct": wavg(h.bb_fold[bb], h.p_cbet[bb]), "bb_call_pct": wavg(h.bb_call[bb], h.p_cbet[bb]),
                         "bb_xr_pct": wavg(h.bb_raise[bb], h.p_cbet[bb])})
    by_bucket = pd.DataFrame(rows)
    by_bucket.round(3).to_csv(OUT / "by_bucket.csv", index=False)

    # Charts.
    labels, series = [], {"BTN c-bets": [], "BB folds (of c-bets)": [], "BB check-raises (of c-bets)": []}
    for _, r in summary.iterrows():
        labels.append(f"{r.boards}\n{r.scenario}")
        series["BTN c-bets"].append(r.btn_cbet_pct)
        series["BB folds (of c-bets)"].append(r.bb_fold_pct)
        series["BB check-raises (of c-bets)"].append(r.bb_xr_pct)
    order = [0, 3, 1, 4, 2, 5]
    grouped_bars([labels[k] for k in order], {k: [v[i] for i in order] for k, v in series.items()},
                 "Learned c-bets vs fixed 50-75% c-bets: the BB's response", "Frequency",
                 OUT / "scenario_comparison.png",
                 groups=[("all flops", 2), ("boards that favor the BTN", 2), ("boards that favor the BB", 2)])
    fx_b = by_bucket[by_bucket.scenario == "fixed 50-75%"].dropna(subset=["bb_xr_pct"])
    stacked_bars(list(fx_b.bucket), {"fold": fx_b.bb_fold_pct, "call": fx_b.bb_call_pct,
                                     "check-raise": fx_b.bb_xr_pct},
                 "BB vs fixed 50-75% c-bets, by hand", "Share of c-bets faced", OUT / "bb_vs_fixed_by_bucket.png",
                 colors={"fold": "#e4e3df", "call": "#2a78d6", "check-raise": "#eb6834"})
    eq_b = by_bucket[by_bucket.scenario == "learned"].set_index("bucket")
    fx_all = by_bucket[by_bucket.scenario == "fixed 50-75%"].set_index("bucket")
    grouped_bars(BUCKETS, {"learned": list(eq_b.btn_cbet_pct), "fixed 50-75%": list(fx_all.btn_cbet_pct)},
                 "BTN c-bet frequency by hand: learned vs fixed 50-75%", "C-bet frequency",
                 OUT / "btn_cbet_by_bucket.png")

    print(summary.round(3).to_string(index=False))
    print(by_bucket.round(2).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
