"""Experiment 25: is a 2/3-pot c-bet a mistake for the BTN? (2.25bb open, 100bb)

The whole-hand model (pot-sized raises) is trained with different flop c-bet sizes for the BTN; everything
else is unchanged. The BB re-learns its best response in each version, so the BTN's value per called pot
shows what each set of sizes is worth:

    single sizes  1/3, 2/3, 3/4, pot             (the cost of using one size everywhere)
    two sizes     1/3 + 3/4                      (the current model)
    three sizes   1/3 + 2/3 + pot                (does the BTN use 2/3 when it has both neighbours?)
    overbet       1/3 + 3/4 + 1.25x pot

Values are also split by board, to see which single size is best where. TexasSolver's view (flop sizes
33 / 66 / 100 on six flops) is read from `solver_compare.py` with SOLVER_FLOP_SIZES=33,66,100 and
SOLVER_CHIP_SCALE=10.

Usage:
    python scripts/cbet_size_study.py train [names]    (~12 min per version, cached)
    python scripts/cbet_size_study.py report
"""
import json
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
from preflop.cards import HANDS
from preflop.flopgame import TEXTURES as FLOP_TEXTURES
from preflop.plots import grouped_bars, heatmap

QUICK = os.environ.get("STUDY_QUICK") == "1"
OUT = ROOT / "output" / ("cbet_sizes_quick" if QUICK else "cbet_sizes")
CACHE = OUT / "cache"                          # gitignored
CACHE.mkdir(parents=True, exist_ok=True)
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 400_000)
S, STACK = 2.25, 100
VERSIONS = {"1/3 only": (1 / 3,), "2/3 only": (2 / 3,), "3/4 only": (3 / 4,), "pot only": (1.0,),
            "1/3 + 3/4": (1 / 3, 3 / 4), "1/3 + 2/3 + pot": (1 / 3, 2 / 3, 1.0),
            "1/3 + 3/4 + 1.25x": (1 / 3, 3 / 4, 1.25)}
NAMES = {1 / 3: "1/3", 2 / 3: "2/3", 3 / 4: "3/4", 1.0: "pot", 1.25: "1.25x"}
BOARDS = [("A-high", 0, "A-high"), ("K-high", 0, "K-high"), ("Q/J-high", 0, "Q/J-high"),
          ("T-high or lower", 0, "T-high or lower"), ("paired", 1, "paired"), ("monotone", 2, "monotone"),
          ("connected", 3, "connected")]
CLASSES = {"two pair+": [0, 1, 2], "overpair / top pair": [3, 4, 5], "middle / weak pair": [6, 7],
           "draws": [8, 9], "overcards / air": [10, 11]}
SOLVER_TAG = "medium_f33-66-100_x10"          # 10 chips per bb, so bet sizes are not rounded


def opts_for(name):
    sizes = list(oss.SIZES)
    sizes[0] = {"lead": oss.SIZES[0]["lead"], "bet": VERSIONS[name]}
    return {**oss.OPTS_POT, "sizes": tuple(sizes)}


def path_for(name):
    return CACHE / (name.replace(" ", "").replace("/", "-").replace("+", "_") + ".npz")


def train(name):
    path = path_for(name)
    if path.exists():
        print(f"{name}: cached")
        return
    r = oss.ranges_for(S)
    opts = opts_for(name)
    print(f"== {name}", flush=True)
    t0 = time.time()
    sigma, _ = fg.learn(r["btn_range"], r["bb_range"], opts, s=S, stack=STACK, batches=BATCHES,
                        hands_per_batch=HANDS_PER_BATCH, seed=1, verbose=False)
    value, acc = fg.evaluate(r["btn_range"], r["bb_range"], sigma, opts, s=S, stack=STACK, n=EVAL_HANDS)
    out = {"value_btn": value, "value_by_flop": acc.value_by_flop, "n_by_flop": acc.n_by_flop}
    for key in sigma:
        out[f"strat_{key[0]}_{key[1]}"] = acc.strat[key]
        out[f"reach_{key[0]}_{key[1]}"] = acc.reach[key]
    np.savez_compressed(path, **out)
    print(f"   BTN net {value - S:+.4f}bb per called pot ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _tex_mask(shape, ts):
    m = np.zeros(shape)
    m[ts] = 1
    return m


def board_groups():
    return {name: [t for t, tex in enumerate(FLOP_TEXTURES) if tex[k] == v] for name, k, v in BOARDS}


def solver_usage():
    """TexasSolver's flop c-bet split with 33 / 66 / 100% sizes available, by flop."""
    import solver_compare as sc
    raw = ROOT / "output" / "solver_compare" / "raw" / SOLVER_TAG
    w = dict(zip(HANDS, np.load(sc.POLICIES)["btn_range"]))   # the BTN range the solves used
    rows = []
    for f in sorted(raw.glob("*.json")):
        d = json.load(open(f))
        node = d["childrens"]["CHECK"] if d["actions"] == ["CHECK"] else d
        acts = node["strategy"]["actions"]
        tot = {a: 0.0 for a in acts}
        wsum = 0.0
        for combo, probs in node["strategy"]["strategy"].items():
            wt = w[sc.class_of(combo)]
            wsum += wt
            for a, p in zip(acts, probs):
                tot[a] += wt * p
        row = {"flop": f.stem}
        for a, v in tot.items():
            row[a.split(" ")[0].lower() if a == "CHECK" else f"bet {float(a.split()[1]) / (2 * S * 10) * 100:.0f}%"] = v / wsum
        rows.append(row)
    return pd.DataFrame(rows)


def report():
    rows, by_board, usage = [], [], []
    groups = board_groups()
    for name in VERSIONS:
        path = path_for(name)
        if not path.exists():
            continue
        d = np.load(path)
        row = {"version": name, "btn_net": float(d["value_btn"]) - S}
        f = node_freq(d, 0, "btn_vs_check")
        row["btn_cbets"] = 1 - f[0]
        for k, sz in enumerate(VERSIONS[name]):
            row[f"bet {NAMES[sz]}"] = f[1 + k]
        g = node_freq(d, 0, "bb_vs_bet")
        row.update({"bb_fold": g[0], "bb_call": g[1], "bb_raise": g[2:].sum()})
        rows.append(row)
        vb, nb = d["value_by_flop"], d["n_by_flop"]
        for gname, ts in groups.items():
            by_board.append({"version": name, "board": gname, "btn_net": float(vb[ts].sum() / nb[ts].sum()) - S})
        if len(VERSIONS[name]) > 1:
            shp = d["reach_0_btn_vs_check"].shape
            for cls, b in CLASSES.items():
                f = node_freq(d, 0, "btn_vs_check", cell_mask(shp, buckets=b))
                usage.append({"version": name, "hand": cls, "check": f[0],
                              **{NAMES[sz]: f[1 + k] for k, sz in enumerate(VERSIONS[name])}})
            for gname, ts in groups.items():
                f = node_freq(d, 0, "btn_vs_check", _tex_mask(shp, ts))
                usage.append({"version": name, "hand": f"board: {gname}", "check": f[0],
                              **{NAMES[sz]: f[1 + k] for k, sz in enumerate(VERSIONS[name])}})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "versions.csv", index=False)
    bb = pd.DataFrame(by_board)
    bb.round(4).to_csv(OUT / "by_board.csv", index=False)
    us = pd.DataFrame(usage)
    us.round(3).to_csv(OUT / "size_usage.csv", index=False)

    base = df.set_index("version").btn_net
    ref = base.get("1/3 + 3/4", base.max())
    grouped_bars(list(df.version), {"BTN net vs the current 1/3 + 3/4 (bb / 100 called pots)":
                                    list((df.btn_net - ref) * 100)},
                 "What each c-bet size set is worth to the BTN", "bb per 100 called pots",
                 OUT / "value_by_version.png", pct=False)
    piv = bb.pivot(index="board", columns="version", values="btn_net")
    piv = (piv.sub(piv["1/3 + 3/4"], axis=0) * 100) if "1/3 + 3/4" in piv else piv * 100
    heatmap(piv[[v for v in VERSIONS if v in piv.columns]], "BTN net vs the current sizes, by board (bb / 100 called pots)",
            "Version", "Board", OUT / "value_by_board.png", fmt="signed", diverging=True, figsize=(9.5, 5))
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(piv.round(1).to_string())
    print(us.round(2).to_string(index=False))
    try:
        su = solver_usage()
        su.round(3).to_csv(OUT / "solver_usage.csv", index=False)
        print(su.round(3).to_string(index=False))
    except Exception as e:                                  # solver runs are optional (and slow)
        print("solver usage unavailable:", e)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "train":
        for name in (sys.argv[2:] or list(VERSIONS)):
            train(name)
    else:
        report()


if __name__ == "__main__":
    main()
