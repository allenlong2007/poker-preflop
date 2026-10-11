"""Experiment 22: why does the model's BB check-raise about twice as often as TexasSolver?

Experiment 16's whole-hand model (2.25bb open, 100bb) check-raised a 1/3-pot c-bet 29% vs the solver's
18%, with middle pair and weak top pair raising 54-64% vs 7-18%. Two of the suspected causes are about
the game the model plays, so each is changed to match the solver's medium tree:

    exp16        experiment 16's model: raises of 2.5x or 4x the bet on every street, flop donks allowed
    pot raise    one raise size, the solver's: raise by 50% of the pot after calling
                 (3.5x a 1/3-pot bet, 2.7x a 3/4-pot bet); everything else as exp16
    solver tree  the solver's whole tree: no flop donk, flop bets 1/3 / 3/4, one turn size (2/3 pot)
                 and one river size (3/4), raises (50% pot) on the flop only

Each model is compared with the solver on the 12 solved 2.25bb flops.

Usage:
    python scripts/xr_level_study.py train      (~25-35 min per model, cached)
    python scripts/xr_level_study.py report
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
import open_size_study as oss
from bb_short_stack_study import cell_mask, node_freq
from preflop import fullgame as fg
from preflop.plots import grouped_bars, heatmap

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("xr_level_quick" if QUICK else "xr_level")
CACHE = OUT / "cache"                          # gitignored
CACHE.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 100_000)
EXP16 = ROOT / "output" / "open_sizes" / "cache" / "model_2_25.npz"
VARIANTS = {
    "exp16": None,
    "pot raise": {**oss.OPTS, "raise_pot": (0.5,)},
    "solver tree": {"donk": False, "raise_pot": (0.5,), "raise_streets": (0,),
                    "sizes": ({"lead": (1 / 3,), "bet": (1 / 3, 3 / 4)},
                              {"lead": (2 / 3,), "bet": (2 / 3,)},
                              {"lead": (3 / 4,), "bet": (3 / 4,)})},
}
FOCUS = ["two pair", "overpair", "top pair, T+ kicker", "top pair, weak kicker", "middle pair", "weak pair",
         "strong draw", "weak draw", "overcards", "air"]


def path_for(name):
    return EXP16 if name == "exp16" else CACHE / f"{name.replace(' ', '_')}.npz"


def train(name):
    path = path_for(name)
    if path.exists():
        print(f"{name}: cached")
        return
    opts = VARIANTS[name]
    r = oss.ranges_for(2.25)
    print(f"== {name}", flush=True)
    t0 = time.time()
    sigma, _ = fg.learn(r["btn_range"], r["bb_range"], opts, s=2.25, stack=100, batches=BATCHES,
                        hands_per_batch=HANDS_PER_BATCH, seed=1)
    value, acc = fg.evaluate(r["btn_range"], r["bb_range"], sigma, opts, s=2.25, stack=100, n=EVAL_HANDS)
    out = {"value_btn": value}
    for (st, kind), v in sigma.items():
        out[f"sigma_{st}_{kind}"] = v
        out[f"strat_{st}_{kind}"] = acc.strat[(st, kind)]
        out[f"reach_{st}_{kind}"] = acc.reach[(st, kind)]
    np.savez_compressed(path, **out)
    print(f"   BB net {2.25 - value:+.3f}bb per called pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


def model_stats(name, d):
    shp = d["reach_0_bb_vs_bet"].shape
    row = {"variant": name, "bb_net": 2.25 - float(d["value_btn"])}
    f = node_freq(d, 0, "btn_vs_check")
    row.update({"btn_cbet": 1 - f[0], "btn_cbet_small": f[1]})
    row["btn_air_cbet"] = 1 - node_freq(d, 0, "btn_vs_check", cell_mask(d["reach_0_btn_vs_check"].shape,
                                                                        buckets=[10, 11]))[0]
    for k, n in enumerate(("1/3", "3/4")):
        f = node_freq(d, 0, "bb_vs_bet", cell_mask(shp, size=k))
        row.update({f"bb_fold_vs_{n}": f[0], f"bb_call_vs_{n}": f[1], f"bb_xr_vs_{n}": f[2:].sum()})
    return row


def report():
    import solver_compare as sc
    stats, agree, buckets = [], [], []
    for name in VARIANTS:
        if not path_for(name).exists():
            continue
        d = np.load(path_for(name))
        stats.append(model_stats(name, d))
        pf, bk = oss.model_vs_solver(2.25, d, sc.FLOPS)
        for m in ("btn_check", "btn_small", "bb_fold_vs_small", "bb_raise_vs_small", "bb_call_vs_small",
                  "bb_fold_vs_big", "bb_raise_vs_big", "bb_call_vs_big"):
            agree.append({"variant": name, "decision": m, "solver": pf[f"solver_{m}"].mean(),
                          "model": pf[f"model_{m}"].mean(), "corr": pf[f"solver_{m}"].corr(pf[f"model_{m}"])})
        bk = bk[bk.player == "bb_vs_small"]
        g = bk.groupby("bucket").apply(lambda x: pd.Series({
            "solver_raise": (x.solver_raise * x.weight).sum() / x.weight.sum(),
            "model_raise": (x.model_raise * x.weight).sum() / x.weight.sum(),
            "solver_fold": (x.solver_fold * x.weight).sum() / x.weight.sum(),
            "model_fold": (x.model_fold * x.weight).sum() / x.weight.sum()}), include_groups=False)
        g["variant"] = name
        buckets.append(g.reset_index())
    st = pd.DataFrame(stats)
    ag = pd.DataFrame(agree)
    bk = pd.concat(buckets)
    st.round(4).to_csv(OUT / "model_stats.csv", index=False)
    ag.round(4).to_csv(OUT / "solver_agreement.csv", index=False)
    bk.round(4).to_csv(OUT / "xr_by_bucket.csv", index=False)

    # Charts: check-raise vs a 1/3 c-bet, overall and by hand.
    raise_rows = ag[ag.decision == "bb_raise_vs_small"].set_index("variant")
    labels = list(raise_rows.index)
    grouped_bars(["TexasSolver"] + labels, {"check-raise vs 1/3": [raise_rows.solver.iloc[0]] + list(raise_rows.model),
                                            "check-raise vs 3/4": [ag[(ag.decision == "bb_raise_vs_big")].solver.iloc[0]]
                                            + list(ag[ag.decision == "bb_raise_vs_big"].model)},
                 "BB check-raise frequency on the 12 solved flops", "Share of the BB's checking range",
                 OUT / "xr_level.png")
    piv = bk.pivot(index="bucket", columns="variant", values="model_raise")
    piv.insert(0, "TexasSolver", bk[bk.variant == bk.variant.iloc[0]].set_index("bucket").solver_raise)
    piv = piv.reindex([b for b in FOCUS if b in piv.index])[["TexasSolver"] + labels]
    heatmap(piv, "BB check-raises a 1/3-pot c-bet, by hand (12 solved flops)", "", "BB hand",
            OUT / "xr_by_bucket.png", fmt="pct", figsize=(7.5, 5.2))

    pd.set_option("display.width", 220)
    print(st.round(3).to_string(index=False))
    print(ag.round(3).to_string(index=False))
    print(piv.round(2).to_string())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "train":
        for name in sys.argv[2:] or [v for v in VARIANTS if v != "exp16"]:
            train(name.replace("_", " "))
    else:
        report()


if __name__ == "__main__":
    main()
