"""Experiment 24: preflop ranges from postflop values (one consistent realization), 2.25bb open, 100bb.

The preflop solver values every called pot with one realization formula (alpha = 1.25, r_oop = 0.915) for
all 169 hands. Here the whole-hand model measures what each starting hand actually wins after the flop,
and those values go back into the preflop solver as per-hand adjustments:

    called pot (BTN hand i, BB hand j) = formula[i, j] + pot x (a[i] - b[j])

a[i] is how much better the BTN's hand i does postflop than the formula says (as a share of the pot),
b[j] the same for the BB's hand j. (A per-hand realization multiplier can't do this: it caps a hand's value
at the whole pot, while strong hands win more than the pot postflop.) Each round:

    preflop solve with a, b (BB picks its best 3-bet size)
    -> whole-hand model on the resulting ranges (pot-sized raises, 250 batches)
    -> evaluate 1.5M hands, average the BTN's result by each side's preflop hand
    -> fit a, b so the formula plus adjustments matches those averages

Hands a player never plays (no data) get adjustments from a smooth fit on hand features, and noisy hands
are shrunk toward it. 3-bet pots get the same adjustments scaled to their pot (an approximation: the
whole-hand model only plays single-raised pots).

Round 0 has no adjustments (the usual preflop model) and reuses experiment 18's 100bb model. Later rounds
move halfway from the previous adjustments to the new fit (damping), so the loop settles instead of
oscillating. Two versions:

    absolute  the full whole-hand values: the BTN's overall postflop edge comes along too
    shape     only how hands differ from each other: each side's adjustments are centred on 0, so the
              formula's overall BTN-vs-BB level stays and only which hands play well postflop changes

Usage:
    python scripts/realization_loop.py run shape [rounds]      (~25 min per round, cached)
    python scripts/realization_loop.py report shape
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
from open_size_study import OPTS_POT
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.plots import action_grid, range_grid
from preflop.sizing import best_threebet
from preflop.solver import Payoffs, solve

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("realization_loop_quick" if QUICK else "realization_loop")
CACHE = OUT / "cache"                          # gitignored
CACHE.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 20_000) if QUICK else (250, 60_000, 1_500_000)
EXP18_100 = ROOT / "output" / "donk_sizes_stacks" / "cache" / "stack_100.npz"
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
S, STACK = 2.25, 100
SHRINK = 500                                   # hands of data that weigh as much as the smooth fit
DAMP = 0.5                                     # each round moves this far from the old adjustments to the new fit
_EQ = np.load(ROOT / "data" / "equity_matrix.npz")
E, W = _EQ["E"], _EQ["W"]
COMBOS = np.array([n_combos(h) for h in HANDS])
RANKS = "23456789TJQKA"
ALPHA, R_OOP = (float(v) for v in pd.read_csv(FIT, header=None, index_col=0)[1][["alpha", "r_oop"]])


def pct(p, w=None):
    w = COMBOS if w is None else COMBOS * w
    return float((p * w).sum() / max(w.sum(), 1e-12))


# ----------------------------------------------------------------------------- preflop

def preflop(a, b):
    """Equilibrium at the 2.25 open with per-hand postflop adjustments; the BB picks its best 3-bet size."""
    kw = dict(stack=STACK, alpha=ALPHA, r_oop=R_OOP, adj_ip=a, adj_oop=b)
    ts = np.arange(2.5, 6.001, 0.25) * S
    evs = [-solve(E, W, S, t, iters=2000, **kw).btn_ev() for t in ts]
    t, _, edge = best_threebet(ts, np.array(evs))
    res = solve(E, W, S, t, iters=6000, **kw)
    fold, call, three = res.bb_vs_open.T
    return {"btn_range": res.btn_open[:, 1], "bb_fold": fold, "bb_call": call, "bb_3bet": three,
            "btn_vs_3bet": res.btn_vs_3bet, "threebet_to": t, "threebet_on_edge": edge, "btn_ev": res.btn_ev(),
            "exploitability": res.exploitability()}


def threebet_bluffs(p):
    """BB 3-bets with under 50% equity vs the BTN's opening range (experiment 15's definition)."""
    w = W * p["btn_range"][:, None]
    eq_bb = 1 - (E * w).sum(0) / np.maximum(w.sum(0), 1e-12)
    bluff = eq_bb < 0.5
    return pct(p["bb_3bet"] * bluff), [h for h, t, x in zip(HANDS, p["bb_3bet"], bluff) if t > 0.3 and x]


# ----------------------------------------------------------------------------- adjustment fit

def features():
    rows = []
    for h in HANDS:
        hi, lo = RANKS.index(h[0]), RANKS.index(h[1])
        pair, suited = h[0] == h[1], h.endswith("s")
        gap = 0 if pair else min(hi - lo - 1, 4) / 4
        rows.append([1.0, pair, suited, hi / 12, lo / 12, gap, pair * hi / 12, suited * (1 - gap)])
    return np.array(rows, dtype=float)


def fit_adjustments(m, btn_range, bb_range, mode="absolute"):
    """a (BTN hands), b (BB hands), as shares of the pot, so formula + adjustments matches the whole-hand model."""
    F = Payoffs(E, S, S, STACK, 1.0, R_OOP, ALPHA).call_open           # formula: BTN net per called pot
    w = W * btn_range[:, None] * bb_range[None, :]
    n = {q: m["n_by_hand_" + q] for q in ("btn", "bb")}
    T = {q: m["value_by_hand_" + q] / np.maximum(n[q], 1) - S for q in ("btn", "bb")}   # BTN net, by hand
    pot = 2 * S
    a, b = np.zeros(169), np.zeros(169)
    rows, cols = np.maximum(w.sum(1), 1e-12), np.maximum(w.sum(0), 1e-12)
    for _ in range(200):                               # alternating least squares on the two sets of averages
        a = np.where(n["btn"] > 0, (T["btn"] - (w * (F - pot * b[None, :])).sum(1) / rows) / pot, 0.0)
        b = np.where(n["bb"] > 0, ((w * (F + pot * a[:, None])).sum(0) / cols - T["bb"]) / pot, 0.0)
    c = (b * n["bb"]).sum() / n["bb"].sum()            # only a - b matters: centre the BB's at 0
    a, b = a - c, b - c
    if mode == "shape":                                # keep the formula's overall level: centre the BTN's too
        a = a - (a * n["btn"]).sum() / n["btn"].sum()
    out, X = {}, features()
    for q, raw in (("btn", a), ("bb", b)):
        has = n[q] > 0
        sw = np.sqrt(n[q][has])
        coef, *_ = np.linalg.lstsq(X[has] * sw[:, None], raw[has] * sw, rcond=None)
        smooth = X @ coef
        k = n[q] / (n[q] + SHRINK)
        out[q] = k * np.where(has, raw, 0.0) + (1 - k) * smooth
        out[q + "_raw"] = np.where(has, raw, np.nan)
        out[q + "_n"] = n[q]
    # How well the fit reproduces the per-hand averages.
    G = F + pot * (out["btn"][:, None] - out["bb"][None, :])
    for q, r in (("btn", (w * G).sum(1) / rows - T["btn"]), ("bb", (w * G).sum(0) / cols - T["bb"])):
        if mode == "shape":
            r = r - (r * n[q]).sum() / n[q].sum()      # the level is left out on purpose
        out[q + "_resid"] = np.where(n[q] > 0, r, np.nan)
    return out


# ----------------------------------------------------------------------------- the loop

def model_round(k, p, cache):
    path = (CACHE if k == 0 else cache) / f"model_{k}.npz"     # round 0 is the same in every version
    if path.exists():
        return np.load(path)
    if k == 0 and not QUICK and EXP18_100.exists():
        d = np.load(EXP18_100)              # experiment 18's 100bb model: no adjustments, near-identical ranges
        sigma = {(int(x.split("_")[1]), "_".join(x.split("_")[2:])): d[x] for x in d.files if x.startswith("sigma_")}
    else:
        print(f"   training the whole-hand model, round {k}", flush=True)
        sigma, _ = fg.learn(p["btn_range"], p["bb_call"], OPTS_POT, s=S, stack=STACK, batches=BATCHES,
                            hands_per_batch=HANDS_PER_BATCH, seed=1, verbose=False)
    value, acc = fg.evaluate(p["btn_range"], p["bb_call"], sigma, OPTS_POT, s=S, stack=STACK, n=EVAL_HANDS)
    np.savez_compressed(path, value_btn=value, **{f"value_by_hand_{q}": acc.value_by_hand[q] for q in ("btn", "bb")},
                        **{f"n_by_hand_{q}": acc.n_by_hand[q] for q in ("btn", "bb")})
    return np.load(path)


def wavg_abs(r, n):
    m = n > 0
    return float(np.nansum(np.abs(r[m]) * n[m]) / n[m].sum())


def run(mode, rounds):
    cache = CACHE / mode
    cache.mkdir(exist_ok=True)
    a, b = np.zeros(169), np.zeros(169)
    for k in range(rounds + 1):
        path = cache / f"round_{k}.npz"
        if not path.exists():
            t0 = time.time()
            p = preflop(a, b)
            print(f"== round {k}: BTN opens {pct(p['btn_range']):.1%}, BB fold / call / 3-bet "
                  f"{pct(p['bb_fold']):.1%} / {pct(p['bb_call']):.1%} / {pct(p['bb_3bet']):.1%} "
                  f"(3-bet to {p['threebet_to']:.2f})", flush=True)
            m = model_round(k, p, cache)
            fit = fit_adjustments(m, p["btn_range"], p["bb_call"], mode)
            np.savez(path, adj_btn=a, adj_bb=b, value_btn=m["value_btn"], **p, **{f"fit_{x}": v for x, v in fit.items()})
            print(f"   BB net {S - float(m['value_btn']):+.3f}bb per called pot; fit residual (bb, weighted by "
                  f"hands) BTN {wavg_abs(fit['btn_resid'], fit['btn_n']):.3f}, BB {wavg_abs(fit['bb_resid'], fit['bb_n']):.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        d = np.load(path)
        a, b = a + DAMP * (d["fit_btn"] - a), b + DAMP * (d["fit_bb"] - b)
    final = cache / "final.npz"                        # the ranges implied by the last round's adjustments
    if not final.exists():
        np.savez(final, adj_btn=a, adj_bb=b, **preflop(a, b))


# ----------------------------------------------------------------------------- report

def report(mode):
    cache, out = CACHE / mode, OUT / mode
    out.mkdir(exist_ok=True)
    rounds = sorted(int(f.stem.split("_")[1]) for f in cache.glob("round_*.npz"))
    stages = [(f"round {k}", np.load(cache / f"round_{k}.npz")) for k in rounds]
    if (cache / "final.npz").exists():
        stages.append(("final", np.load(cache / "final.npz")))
    rows = []
    for name, d in stages:
        bpct, bl = threebet_bluffs(d)
        rows.append({"stage": name, "btn_opens": pct(d["btn_range"]), "bb_fold": pct(d["bb_fold"]),
                     "bb_call": pct(d["bb_call"]), "bb_3bet": pct(d["bb_3bet"]), "threebet_to": float(d["threebet_to"]),
                     "btn_folds_vs_3bet": pct(d["btn_vs_3bet"][:, 0], d["btn_range"]),
                     "bb_3bet_bluffs": bpct, "bluff_hands": " ".join(bl), "btn_ev_mbb": 1000 * float(d["btn_ev"]),
                     "bb_net_called_pot": S - float(d["value_btn"]) if "value_btn" in d.files else np.nan})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "rounds.csv", index=False)

    def main_action(d):
        return np.argmax(np.stack([d["bb_fold"], d["bb_call"], d["bb_3bet"]]), 0)
    changes = pd.DataFrame([{"from": x, "to": y, "bb_hands_changed": pct((main_action(dx) != main_action(dy)).astype(float)),
                             "btn_open_change": pct(np.abs(dx["btn_range"] - dy["btn_range"]))}
                            for (x, dx), (y, dy) in zip(stages, stages[1:])])
    changes.round(4).to_csv(out / "range_changes.csv", index=False)

    first, last = stages[0][1], stages[-1][1]
    fit = stages[-1][1] if "adj_btn" in stages[-1][1].files else np.load(cache / f"round_{rounds[-1]}.npz")
    adj = {"btn": fit["adj_btn"], "bb": fit["adj_bb"]}           # the adjustments behind the final ranges
    nfit = np.load(cache / f"round_{rounds[-1]}.npz")
    pot = 2 * S
    for q, who in (("btn", "BTN"), ("bb", "BB")):
        v = adj[q] * pot
        lim = float(np.nanpercentile(np.abs(v), 95))
        range_grid(v, f"{who}'s postflop value vs the formula, by hand (bb per called pot)",
                   out / f"adjustment_{q}.png", fmt="signed", vmin=-lim, vmax=lim)
    for name, d in (("before", first), ("after", last)):
        bpct, _ = threebet_bluffs(d)
        action_grid({"fold": d["bb_fold"], "call": d["bb_call"], "3-bet": d["bb_3bet"]},
                    f"BB vs a 2.25bb open, 100bb, {name} (3-bet to {float(d['threebet_to']):.1f})",
                    out / f"bb_range_{name}.png",
                    subtitle=f"Fold {pct(d['bb_fold']):.0%}   Call {pct(d['bb_call']):.0%}   "
                             f"3-bet {pct(d['bb_3bet']):.0%} (bluffs {bpct:.1%} of hands)")
        range_grid(d["btn_range"], f"BTN opening range, {name} ({pct(d['btn_range']):.0%})", out / f"btn_open_{name}.png")
    hd = pd.DataFrame([{"hand": h, "adj_btn_bb": adj["btn"][i] * pot, "adj_bb_bb": adj["bb"][i] * pot,
                        "n_btn": nfit["fit_btn_n"][i], "n_bb": nfit["fit_bb_n"][i],
                        "btn_open_before": first["btn_range"][i], "btn_open_after": last["btn_range"][i],
                        "bb_fold_before": first["bb_fold"][i], "bb_fold_after": last["bb_fold"][i],
                        "bb_call_before": first["bb_call"][i], "bb_call_after": last["bb_call"][i],
                        "bb_3bet_before": first["bb_3bet"][i], "bb_3bet_after": last["bb_3bet"][i]}
                       for i, h in enumerate(HANDS)])
    hd.round(3).to_csv(out / "by_hand.csv", index=False)

    pd.set_option("display.width", 250)
    print(df.drop(columns="bluff_hands").round(3).to_string(index=False))
    for r in rows:
        print(f"{r['stage']} 3-bet bluffs: {r['bluff_hands']}")
    print(changes.round(3).to_string(index=False))
    for col, nc in (("adj_btn_bb", "n_btn"), ("adj_bb_bb", "n_bb")):
        g = hd[hd[nc] > 0].sort_values(col)
        print(col, "lowest:", " ".join(f"{h} {v:+.2f}" for h, v in zip(g.hand[:10], g[col][:10])))
        print(col, "highest:", " ".join(f"{h} {v:+.2f}" for h, v in zip(g.hand[-10:], g[col][-10:])))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    mode = sys.argv[2] if len(sys.argv) > 2 else "shape"
    if cmd == "run":
        run(mode, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    else:
        report(mode)


if __name__ == "__main__":
    main()
