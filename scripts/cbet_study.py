"""Experiment 8: if the BB always checks, what does the Button c-bet -- and when should the BB check-raise?

BTN opens 2.25bb, BB calls (ranges from the preflop solver with the pot-odds
realization fit). On the flop the BB always checks; the BTN chooses check or a
1/3-pot c-bet; the BB chooses fold / call / check-raise to 3x; the BTN answers a
check-raise with fold / call. All flop choices are learned per (board texture,
hand bucket) with CFR+ and Monte Carlo rollouts (preflop/flopgame.py); turn and
river use the pot-odds / balanced rules.

Usage:  python scripts/cbet_study.py [open size]     (~4 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bb_study import solve_at
from preflop.cards import HANDS
from preflop.flopgame import BUCKETS, COLUMNS, HIGH, SUITS, TEXTURES, learn_and_play
from preflop.plots import action_grid, grouped_bars, heatmap, range_grid, stacked_bars

OPEN = float(sys.argv[1]) if len(sys.argv) > 1 else 2.25
ROUNDS = 60
N_HANDS = 800_000
MIN_SAMPLES = 2000
OUT = ROOT / "output" / "cbet_study"
OUT.mkdir(parents=True, exist_ok=True)
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"

FEATURES = [("high card", 0, HIGH), ("pairing", 1, ["unpaired", "paired"]), ("suits", 2, SUITS),
            ("connectedness", 3, ["disconnected", "connected"])]
GRAY, BLUE, ORANGE = "#e4e3df", "#2a78d6", "#eb6834"


def wavg(x, w):
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    f = pd.read_csv(FIT, header=None, index_col=0)[1]
    res, t3 = solve_at(E, W, OPEN, float(f["alpha"]), float(f["r_oop"]))
    # Local cache this script writes itself (gitignored); delete it to re-learn.
    cache = OUT / f"hands_{OPEN:g}.npz"
    if cache.exists():
        hands = np.load(cache)["hands"]
    else:
        out = learn_and_play(res.btn_open[:, 1], res.bb_vs_open[:, 1], s=OPEN, rounds=ROUNDS, n=N_HANDS)
        hands = out["hands"]
        np.savez_compressed(cache, hands=hands)
        pd.DataFrame(out["history"]).to_csv(OUT / "learning_history.csv", index=False)
    h = pd.DataFrame(hands, columns=COLUMNS)
    tex = np.array(TEXTURES, dtype=object)
    for name, k, _ in FEATURES:
        h[name] = [tex[int(t)][k] for t in h.texture]
    h["cbet_and_xr"] = h.p_cbet * h.bb_raise                 # weight for BTN's answer to a check-raise
    pc = h.p_cbet

    overall = {"open_bb": OPEN, "btn_cbet_pct": pc.mean(), "bb_fold_vs_cbet": wavg(h.bb_fold, pc),
               "bb_call_vs_cbet": wavg(h.bb_call, pc), "bb_xr_vs_cbet": wavg(h.bb_raise, pc),
               "btn_calls_xr": wavg(h.btn_call_vs_xr, h.cbet_and_xr)}
    pd.Series(overall).to_csv(OUT / "overall.csv", header=False)

    # 1. By hand bucket.
    rows = []
    for b, name in enumerate(BUCKETS):
        mb, bb = h.btn_bucket == b, h.bb_bucket == b
        rows.append({"bucket": name, "btn_share_of_flops": mb.mean(), "btn_cbet_pct": pc[mb].mean(),
                     "btn_calls_xr_pct": wavg(h.btn_call_vs_xr[mb], h.cbet_and_xr[mb]),
                     "bb_share_of_flops": bb.mean(), "bb_fold_pct": wavg(h.bb_fold[bb], pc[bb]),
                     "bb_call_pct": wavg(h.bb_call[bb], pc[bb]), "bb_xr_pct": wavg(h.bb_raise[bb], pc[bb])})
    by_bucket = pd.DataFrame(rows)
    by_bucket.round(3).to_csv(OUT / "by_bucket.csv", index=False)
    shown = by_bucket.dropna(subset=["bb_xr_pct"])
    stacked_bars(list(by_bucket.bucket), {"c-bet": by_bucket.btn_cbet_pct, "check": 1 - by_bucket.btn_cbet_pct},
                 f"BTN flop c-bet frequency by hand (BB checked, {OPEN:g}bb pot)", "Share of flops",
                 OUT / "btn_cbet_by_bucket.png", colors={"c-bet": ORANGE, "check": GRAY})
    stacked_bars(list(shown.bucket), {"fold": shown.bb_fold_pct, "call": shown.bb_call_pct,
                                      "check-raise": shown.bb_xr_pct},
                 "BB response to a 1/3-pot c-bet, by hand", "Share of c-bets faced",
                 OUT / "bb_vs_cbet_by_bucket.png", colors={"fold": GRAY, "call": BLUE, "check-raise": ORANGE})

    # 2. By board texture feature.
    labels, cb, xr, groups, trows = [], [], [], [], []
    for name, _, values in FEATURES:
        groups.append((name, len(values)))
        for v in values:
            m = h[name] == v
            labels.append(v)
            cb.append(pc[m].mean())
            xr.append(wavg(h.bb_raise[m], pc[m]))
            trows.append({"feature": name, "value": v, "flops_pct": m.mean(), "btn_cbet_pct": cb[-1],
                          "bb_fold_pct": wavg(h.bb_fold[m], pc[m]), "bb_call_pct": wavg(h.bb_call[m], pc[m]),
                          "bb_xr_pct": xr[-1]})
    pd.DataFrame(trows).round(3).to_csv(OUT / "by_texture_feature.csv", index=False)
    grouped_bars(labels, {"BTN c-bets": cb, "BB check-raises (of c-bets)": xr},
                 "Flop texture: how often the BTN c-bets and the BB check-raises", "Frequency",
                 OUT / "texture_effects.png", groups=groups)

    # 3. When should the BB check-raise? Hand bucket x texture feature.
    grid = {}
    for name, _, values in FEATURES:
        for v in values:
            m = h[name] == v
            grid[f"{v}"] = [wavg(h.bb_raise[m & (h.bb_bucket == b)], pc[m & (h.bb_bucket == b)])
                            if (m & (h.bb_bucket == b)).sum() >= MIN_SAMPLES else np.nan
                            for b in range(len(BUCKETS))]
    xr_grid = pd.DataFrame(grid, index=BUCKETS).T
    xr_grid = xr_grid.loc[:, xr_grid.notna().any()]
    xr_grid.round(3).to_csv(OUT / "bb_xr_by_texture_and_hand.csv")
    heatmap(xr_grid, "BB check-raise frequency vs a c-bet: board feature x hand", "BB hand on the flop",
            "Board feature", OUT / "bb_xr_heatmap.png", fmt="pct", figsize=(9, 6))
    grid = {}
    for name, _, values in FEATURES:
        for v in values:
            m = h[name] == v
            grid[v] = [pc[m & (h.btn_bucket == b)].mean() if (m & (h.btn_bucket == b)).sum() >= MIN_SAMPLES
                       else np.nan for b in range(len(BUCKETS))]
    cb_grid = pd.DataFrame(grid, index=BUCKETS).T
    cb_grid.round(3).to_csv(OUT / "btn_cbet_by_texture_and_hand.csv")
    heatmap(cb_grid, "BTN c-bet frequency: board feature x hand", "BTN hand on the flop", "Board feature",
            OUT / "btn_cbet_heatmap.png", fmt="pct", figsize=(9, 6))

    # 4. Full textures ranked (enough samples only).
    full = []
    for k, t in enumerate(TEXTURES):
        m = h.texture == k
        if m.sum() < 2000:
            continue
        full.append({"texture": " / ".join(t), "flops_pct": m.mean(), "btn_cbet_pct": pc[m].mean(),
                     "bb_xr_pct": wavg(h.bb_raise[m], pc[m]), "bb_fold_pct": wavg(h.bb_fold[m], pc[m])})
    full = pd.DataFrame(full).sort_values("bb_xr_pct", ascending=False)
    full.round(3).to_csv(OUT / "by_full_texture.csv", index=False)

    # 5. 13x13: the BTN's c-bet range and the BB's response, by preflop hand.
    cls_b, cls_bb = h.btn_class.astype(int), h.bb_class.astype(int)
    n_b = np.bincount(cls_b, minlength=169)
    cbet_hand = np.bincount(cls_b, pc, 169) / np.maximum(n_b, 1)
    cbet_hand[n_b < MIN_SAMPLES] = np.nan
    range_grid(cbet_hand, f"BTN c-bet range: how often each hand c-bets the flop ({OPEN:g}bb, BB checks)",
               OUT / "btn_cbet_range.png", fmt="pct",
               note="Averaged over all flops. Blank = hand not in the BTN's opening range.")
    w_bb = np.bincount(cls_bb, pc, 169)
    enough = np.bincount(cls_bb, minlength=169) >= MIN_SAMPLES     # hide hands the BB rarely calls with
    acts = {}
    for name, col in (("fold", "bb_fold"), ("call", "bb_call"), ("check-raise", "bb_raise")):
        acts[name] = np.where(enough & (w_bb > 0), np.bincount(cls_bb, pc * h[col], 169) / np.maximum(w_bb, 1e-9), 0)
    action_grid(acts, f"BB vs a flop c-bet, by preflop hand ({OPEN:g}bb open)", OUT / "bb_vs_cbet_range.png",
                subtitle="Averaged over all flops. Blank = not in the BB's calling range.")
    pd.DataFrame({"hand": HANDS, "btn_cbet_pct": cbet_hand, "bb_fold": acts["fold"], "bb_call": acts["call"],
                  "bb_check_raise": acts["check-raise"]}).round(3).to_csv(OUT / "by_preflop_hand.csv", index=False)

    print(pd.Series(overall).round(3).to_string())
    print(by_bucket.round(2).to_string(index=False))
    print(pd.DataFrame(trows).round(2).to_string(index=False))
    print("\nTop check-raise textures:\n", full.head(8).round(2).to_string(index=False))
    print("\nLowest check-raise textures:\n", full.tail(6).round(2).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
