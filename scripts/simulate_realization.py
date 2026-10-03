"""Experiment 6b: measure equity realization by playing simple postflop rules.

Loop (until alpha stops moving):
  1. Solve the preflop game at a 2.25bb open with the current (alpha, r_oop),
     letting the BB pick its best 3-bet size -> BTN opening range, BB calling range.
  2. Deal ~1.5M hands from those ranges and play them out with the rules in
     preflop/postflop.py.
  3. Fit alpha and r_oop to the simulated results:
        BTN realized share  ~  E^a / (E^a + r_oop * (1-E)^a)

Then: per-hand realization charts, and which BTN open size the simulated
(alpha, r_oop) prefers.

Usage:  python scripts/simulate_realization.py [pot_odds|fixed]   (~6 min on a multi-core machine)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preflop.cards import HANDS
from preflop.plots import fit_plot, line, range_grid
from preflop.postflop import BET_FRACTION, BLUFF, simulate
from preflop.sizing import bb_ev_grid, best_threebet
from preflop.solver import solve

OPEN = 2.25
N_HANDS = 1_500_000
ROUNDS = 4
START = {"alpha": 1.5, "r_oop": 0.85}
OPENS = np.round(np.arange(1.75, 3.501, 0.05), 2)
MIN_SAMPLES = 1500        # hands needed before we report a per-hand realization
CALLING = sys.argv[1] if len(sys.argv) > 1 else "pot_odds"   # or "fixed" (experiment 6b rules)

OUT = ROOT / "output" / ("simulated_realization" if CALLING == "fixed" else "simulated_realization_pot_odds")
OUT.mkdir(parents=True, exist_ok=True)


def model_share(E, alpha, r_oop):
    x = E ** alpha
    return x / (x + r_oop * (1 - E) ** alpha)


def fit(E, share, bins=60):
    """Grid-search (alpha, r_oop) to match binned average realized share."""
    edges = np.quantile(E, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    ok = n > 0
    e_mean = np.bincount(b, E, bins)[ok] / n[ok]
    s_mean = np.bincount(b, share, bins)[ok] / n[ok]
    alphas = np.arange(0.5, 3.0001, 0.01)
    r_oops = np.arange(0.5, 1.5001, 0.005)
    A, R = np.meshgrid(alphas, r_oops, indexing="ij")
    pred = model_share(e_mean[None, None, :], A[..., None], R[..., None])
    sse = ((pred - s_mean) ** 2 * n[ok]).sum(-1)
    k = np.unravel_index(sse.argmin(), sse.shape)
    return float(alphas[k[0]]), float(r_oops[k[1]]), e_mean, s_mean


def ranges(E, W, alpha, r_oop):
    """BTN raise frequencies and BB call frequencies at OPEN, with the BB's best 3-bet size."""
    ts = np.arange(2.5, 6.001, 0.25) * OPEN
    evs = [-solve(E, W, OPEN, t, iters=2000, alpha=alpha, r_oop=r_oop).btn_ev() for t in ts]
    t, _, _ = best_threebet(ts, np.array(evs))
    res = solve(E, W, OPEN, t, iters=4000, alpha=alpha, r_oop=r_oop)
    return res.btn_open[:, 1], res.bb_vs_open[:, 1], t


def main():
    d = np.load(ROOT / "data" / "equity_matrix.npz")
    E, W = d["E"], d["W"]
    alpha, r_oop = START["alpha"], START["r_oop"]
    history = []
    for rnd in range(1, ROUNDS + 1):
        btn_r, bb_c, t = ranges(E, W, alpha, r_oop)
        sim = simulate(btn_r, bb_c, n=N_HANDS, s=OPEN, seed=rnd, calling=CALLING,
                       bluff="balanced" if CALLING == "pot_odds" else BLUFF)
        bi, bj, share = sim["btn_class"], sim["bb_class"], sim["share"]
        eq = E[bi, bj]
        new_alpha, new_r_oop, e_mean, s_mean = fit(eq, share)
        history.append({"round": rnd, "alpha_in": alpha, "r_oop_in": r_oop, "bb_3bet_to": t,
                        "hands": len(share), "btn_mean_equity": eq.mean(), "btn_mean_share": share.mean(),
                        "alpha_fit": new_alpha, "r_oop_fit": new_r_oop})
        print(f"round {rnd}: in a={alpha:.2f} r_oop={r_oop:.3f} -> fit a={new_alpha:.2f} "
              f"r_oop={new_r_oop:.3f}  (BTN equity {eq.mean():.3f}, realized {share.mean():.3f})")
        done = abs(new_alpha - alpha) < 0.03 and abs(new_r_oop - r_oop) < 0.01
        alpha, r_oop = new_alpha, new_r_oop
        if done:
            break
    pd.DataFrame(history).round(4).to_csv(OUT / "fit_history.csv", index=False)
    if sim["bluff_rates"] is not None:
        pd.DataFrame(sim["bluff_rates"], index=["BB leads", "BTN after check"],
                     columns=["flop", "turn", "river"]).round(3).to_csv(OUT / "bluff_rates.csv")
    if sim["table"] is not None:
        pd.DataFrame(sim["table"], index=["flop", "turn", "river"],
                     columns=[f"hs_{k / 20:.2f}" for k in range(20)]).round(3).to_csv(OUT / "win_chance_table.csv")

    # Fit chart: simulated realized share vs equity, fitted curve, and calibrated curve if present.
    grid_e = np.linspace(0.05, 0.95, 91)
    curves = {f"fitted: a={alpha:.2f}, r_oop={r_oop:.2f}": (grid_e, model_share(grid_e, alpha, r_oop))}
    cal = ROOT / "output" / "calibration" / "calibrated_fit.csv"
    if cal.exists():
        c = pd.read_csv(cal, header=None, index_col=0)[1]
        ca, cr = float(c["alpha"]), float(c["r_oop"])
        curves[f"calibrated to solver: a={ca:g}, r_oop={cr:g}"] = (grid_e, model_share(grid_e, ca, cr))
    fit_plot(e_mean, s_mean, curves, "BTN's realized share of the pot vs its raw equity",
             "BTN raw equity vs the BB's hand", "BTN realized share of the flop pot",
             OUT / "realization_fit.png")

    # Per-hand realization from the last round.
    rows = []
    for who, cls, num, den in (("btn", bi, share, eq), ("bb", bj, 1 - share, 1 - eq)):
        n = np.bincount(cls, minlength=169)
        real = np.bincount(cls, num, 169) / np.maximum(np.bincount(cls, den, 169), 1e-9)
        real[n < MIN_SAMPLES] = np.nan
        for h, k, r in zip(HANDS, n, real):
            rows.append({"player": who, "hand": h, "hands_played": int(k), "realization": round(r, 3)})
        title = {"btn": "BTN (in position)", "bb": "BB (out of position)"}[who]
        range_grid(real, f"{title}: realized ÷ raw equity, simulated", OUT / f"realization_{who}.png",
                   fmt="ratio", vmin=0.5, vmax=1.5,
                   note="1.00 = wins exactly its equity. Blank = hand not in range (too few samples).")
    pd.DataFrame(rows).to_csv(OUT / "realization_by_hand.csv", index=False)

    # Which open size does the simulated realization prefer?
    table = bb_ev_grid(OPENS, lambda s: np.arange(2.5, 6.001, 0.25) * s,
                       {"sim": {"alpha": alpha, "r_oop": r_oop}})
    btn_ev = np.array([-best_threebet(*table[("sim", s)])[1] for s in OPENS])
    best_open = float(OPENS[btn_ev.argmax()])
    good = OPENS[btn_ev >= btn_ev.max() - 0.001]
    line(list(OPENS), list(btn_ev * 1000),
         f"BTN profit by open size, simulated realization (a {alpha:.2f}, r_oop {r_oop:.2f})",
         "BTN open size (bb)", "BTN EV (milli-bb per hand)", OUT / "simulated_open_size.png",
         highlight=(best_open, float(btn_ev.max() * 1000)))
    pd.Series({"alpha": alpha, "r_oop": r_oop, "best_open_bb": best_open,
               "good_open_range_bb": f"{good.min():g}-{good.max():g}",
               "cost_of_2.25_bb_per_100": (btn_ev.max() - btn_ev[np.isclose(OPENS, 2.25)][0]) * 100,
               "bluffing": "balanced" if CALLING == "pot_odds" else BLUFF, "calling": CALLING,
               "bet_fraction": BET_FRACTION}).to_csv(OUT / "simulated_fit.csv", header=False)
    print(f"\nSimulated alpha {alpha:.2f}, r_oop {r_oop:.3f} -> best BTN open {best_open:g}bb "
          f"(within 1 mbb: {good.min():g}-{good.max():g}bb)")


if __name__ == "__main__":
    main()
