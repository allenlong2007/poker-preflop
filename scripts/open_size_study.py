"""Experiment 16: best sizings and actions vs different BTN open sizes, the BTN's 2.25bb strategy, and TexasSolver.

For each BTN open (2.0 / 2.25 / 2.5 / 3.0bb):
  * preflop ranges from the preflop solver (pot-odds realization fit), BB's best 3-bet size;
  * the whole-hand model (preflop/fullgame.py) with a bigger sizing menu, so it can pick its sizes:
        flop:  BB donk 1/3 or 3/4      BTN bet 1/3 or 3/4
        turn:  BB lead 1/2 or pot      BTN bet 1/2 or pot
        river: BB lead 1/2 or pot      BTN bet 1/2, pot or 1.5x
        raises (either player): 2.5x or 4x
Then:
  * the BB's chosen actions and sizes at each open size;
  * the BTN's 2.25bb plan: c-bet, turn barrel after a called c-bet, river bets and bluffs;
  * the most probable full betting lines;
  * a flop comparison with TexasSolver at 2.25bb (12 flops) and 3.0bb (6 flops).

Usage:
    python scripts/open_size_study.py train     # ranges + whole-hand model per open size (~60 min, cached)
    python scripts/open_size_study.py solve     # TexasSolver at a 3.0bb open, 6 flops (~30 min; needs tools/)
    python scripts/open_size_study.py report    # tables + charts
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import eval7
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preflop import fullgame as fg
from preflop.cards import HANDS, n_combos
from preflop.flopgame import TEXTURES as FLOP_TEXTURES, texture as flop_texture
from preflop.plots import grouped_bars, heatmap, scatter_compare, stacked_bars

OPENS = [2.0, 2.25, 2.5, 3.0]
SIZES = ({"lead": (1 / 3, 3 / 4), "bet": (1 / 3, 3 / 4)},
         {"lead": (1 / 2, 1.0), "bet": (1 / 2, 1.0)},
         {"lead": (1 / 2, 1.0), "bet": (1 / 2, 1.0, 3 / 2)})
RAISES = (2.5, 4.0)
OPTS = {"donk": True, "sizes": SIZES, "raises": RAISES}
QUICK = os.environ.get("OPEN_STUDY_QUICK") == "1"          # tiny end-to-end test run
BATCHES, HANDS_PER_BATCH, EVAL_HANDS = (2, 15_000, 3_000) if QUICK else (250, 60_000, 150_000)
SOLVER_OPEN = 3.0
SOLVER_FLOPS = ["As 7d 2c", "Kh 8d 3c", "Td 9h 6c", "7c 5d 3h", "Kd Kc 4s", "Jh 7h 2h"]
OUT = ROOT / "output" / ("open_sizes_quick" if QUICK else "open_sizes")
CACHE = OUT / "cache"                          # gitignored
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)
FIT = ROOT / "output" / "simulated_realization_pot_odds" / "simulated_fit.csv"
GRAY, BLUE, ORANGE, DARK_ORANGE, AQUA = "#e4e3df", "#2a78d6", "#f6a07a", "#c4501f", "#1baf7a"
SIZE_NAMES = {1 / 3: "1/3", 3 / 4: "3/4", 1 / 2: "1/2", 1.0: "pot", 3 / 2: "1.5x"}
CLASSES = ["strong (two pair+)", "one pair", "draw", "air"]


def tag(s):
    return f"{s:g}".replace(".", "_")


# ----------------------------------------------------------------------------- train

def ranges_for(s):
    path = CACHE / f"ranges_{tag(s)}.npz"
    if not path.exists():
        from bb_study import solve_at
        d = np.load(ROOT / "data" / "equity_matrix.npz")
        f = pd.read_csv(FIT, header=None, index_col=0)[1]
        res, t3 = solve_at(d["E"], d["W"], s, float(f["alpha"]), float(f["r_oop"]))
        combos = np.array([n_combos(h) for h in HANDS])
        pct = lambda p: float((p * combos).sum() / combos.sum())
        fold, call, three = res.bb_vs_open.T
        np.savez(path, btn_range=res.btn_open[:, 1], bb_range=call, threebet_to=t3, bb_fold=pct(fold),
                 bb_call=pct(call), bb_3bet=pct(three), btn_raise=pct(res.btn_open[:, 1]))
    return np.load(path)


def train():
    for s in OPENS:
        r = ranges_for(s)
        path = CACHE / f"model_{tag(s)}.npz"
        if path.exists():
            print(f"open {s}: cached")
            continue
        print(f"== open {s}bb (BB folds {float(r['bb_fold']):.0%}, calls {float(r['bb_call']):.0%}, "
              f"3-bets {float(r['bb_3bet']):.0%} to {float(r['threebet_to']):.1f})", flush=True)
        t0 = time.time()
        sigma, _ = fg.learn(r["btn_range"], r["bb_range"], OPTS, s=s, batches=BATCHES,
                            hands_per_batch=HANDS_PER_BATCH, seed=1)
        value, acc = fg.evaluate(r["btn_range"], r["bb_range"], sigma, OPTS, s=s, n=EVAL_HANDS)
        out = {"value_btn": value, "callchain": acc.callchain, "n_flop_bucket": acc.n_flop_bucket,
               "line_keys": np.array(list(acc.lines.keys())), "line_vals": np.array(list(acc.lines.values()))}
        for (st, kind), v in sigma.items():
            out[f"sigma_{st}_{kind}"] = v
            out[f"strat_{st}_{kind}"] = acc.strat[(st, kind)]
            out[f"reach_{st}_{kind}"] = acc.reach[(st, kind)]
        np.savez_compressed(path, **out)
        print(f"   open {s}: BB net {s - value:+.3f}bb per called pot ({(time.time() - t0) / 60:.1f} min)",
              flush=True)


# ----------------------------------------------------------------------------- solve (TexasSolver at 3.0bb)

def solve():
    r = ranges_for(SOLVER_OPEN)
    env = dict(os.environ, SOLVER_TREE="medium", SOLVER_OPEN=str(SOLVER_OPEN),
               SOLVER_RANGES=str(CACHE / f"ranges_{tag(SOLVER_OPEN)}.npz"), SOLVER_FLOPS=",".join(SOLVER_FLOPS),
               SOLVER_PARALLEL="3")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "solver_compare.py"), "solve"], env=env, check=True)


# ----------------------------------------------------------------------------- report helpers

def load(s):
    return np.load(CACHE / f"model_{tag(s)}.npz")


def node_freq(d, st, kind, mask_fn=None):
    """Reach-weighted action frequencies at a node (optionally restricted by a mask over the cell axes)."""
    strat, reach = d[f"strat_{st}_{kind}"], d[f"reach_{st}_{kind}"]
    if mask_fn is not None:
        m = mask_fn(reach.shape)
        strat, reach = strat * m[..., None], reach * m
    flat = strat.reshape(-1, strat.shape[-1]).sum(0)
    return flat / max(float(reach.sum()), 1e-12), float(reach.sum())


def mask(shape, tex=None, lines=None, buckets=None, size=None):
    m = np.ones(shape)
    if tex is not None:
        keep = np.zeros(shape[0], bool)
        keep[list(tex)] = True
        m[~keep] = 0
    if lines is not None:
        keep = np.zeros(shape[1], bool)
        keep[list(lines)] = True
        m[:, ~keep] = 0
    if buckets is not None:
        keep = np.zeros(shape[2], bool)
        keep[list(buckets)] = True
        m[:, :, ~keep] = 0
    if size is not None:
        keep = np.zeros(shape[3], bool)
        keep[size] = True
        m[:, :, :, ~keep] = 0
    return m


def line_text(seq):
    """'xb0c/xx/xb2f' -> readable text using the sizes in SIZES."""
    out = []
    for st, code in enumerate(seq.split("/")):
        if not code:
            continue
        leads, bets = SIZES[st]["lead"], SIZES[st]["bet"]
        name = fg.STREETS[st]
        if code == "xx":
            out.append(f"{name}: check-check")
            continue
        if code.startswith("xb"):
            k = int(code[2])
            txt = f"{name}: BTN bets {SIZE_NAMES[bets[k]]}"
            rest = code[3:]
            who_raises = "BB raises"
        else:
            j = int(code[1])
            txt = f"{name}: BB leads {SIZE_NAMES[leads[j]]}"
            rest = code[2:]
            who_raises = "BTN raises"
        if rest.startswith("r"):
            txt += f", {who_raises} {RAISES[int(rest[1])]:g}x"
            rest = rest[2:]
            caller = "BTN" if who_raises == "BB raises" else "BB"
        else:
            caller = "BB" if code.startswith("xb") else "BTN"
        txt += f", {caller} calls" if rest == "c" else f", {caller} folds"
        out.append(txt)
    ends_in_fold = seq.split("/")[-1].endswith("f")
    return " | ".join(out) + ("" if ends_in_fold else " | showdown")


def flop_feature_groups():
    groups = {}
    for name, k, values in [("high card", 0, ["A-high", "K-high", "Q/J-high", "T-high or lower"]),
                            ("pairing", 1, ["paired"]), ("suits", 2, ["monotone"]),
                            ("connectedness", 3, ["connected"])]:
        for v in values:
            groups[v] = [t for t, tex in enumerate(FLOP_TEXTURES) if tex[k] == v]
    return groups


# ----------------------------------------------------------------------------- solver comparison

def model_vs_solver(s, d, flops):
    """Per-flop and per-bucket flop strategies: TexasSolver vs the whole-hand model at open s."""
    os.environ.update(SOLVER_TREE="medium", SOLVER_OPEN=str(s),
                      SOLVER_RANGES=str(CACHE / f"ranges_{tag(s)}.npz") if s != 2.25 else
                      str(ROOT / "output" / "cbet_frequency" / "flop_policies.npz"))
    import importlib
    import solver_compare as sc
    importlib.reload(sc)
    r = ranges_for(s)
    wbtn, wbb = dict(zip(HANDS, r["btn_range"])), dict(zip(HANDS, r["bb_range"]))
    if s == 2.25:                                       # the 2.25 solves used the experiment-10 ranges
        pol = np.load(ROOT / "output" / "cbet_frequency" / "flop_policies.npz")
        wbtn, wbb = dict(zip(HANDS, pol["btn_range"])), dict(zip(HANDS, pol["bb_range"]))
    btn_sig, bb_sig, first = d["sigma_0_btn_vs_check"], d["sigma_0_bb_vs_bet"], d["sigma_0_bb_first"]
    hole = lambda c: (eval7.Card(c[:2]), eval7.Card(c[2:]))
    rows, bucket_rows = [], []
    for flop in flops:
        if not (sc.RAW / f"{sc.slug(flop)}.json").exists():
            continue
        s_btn, s_bb = sc.solver_flop(flop)
        cards = [eval7.Card(c) for c in flop.split()]
        t = flop_texture(cards)
        btn_b = np.array([fg.fine_bucket(hole(c), cards) for c in s_btn.combo])
        m_btn = btn_sig[t, 0, btn_b]                                  # check, 1/3, 3/4
        bo = np.array([fg.fine_bucket(hole(c), cards) for c in s_bb.combo])
        m_bb = bb_sig[t, 0, bo, s_bb["size"].to_numpy()]              # fold, call, raise 2.5x, raise 4x
        w_btn = s_btn["class"].map(wbtn).to_numpy()
        w_bb = s_bb["class"].map(wbb).to_numpy() * first[t, 0, bo, 0]  # model BB range that checked
        w_bb_solver = s_bb["class"].map(wbb).to_numpy()
        row = {"open": s, "flop": flop}
        for name, sv, mv in (("check", s_btn.check, m_btn[:, 0]), ("small", s_btn.small, m_btn[:, 1]),
                             ("big", s_btn.big, m_btn[:, 2])):
            row[f"solver_btn_{name}"] = float((sv * w_btn).sum() / w_btn.sum())
            row[f"model_btn_{name}"] = float((mv * w_btn).sum() / w_btn.sum())
        for k, size in enumerate(("small", "big")):
            sel = (s_bb["size"] == k).to_numpy()
            for a, sv, mv in (("fold", s_bb["fold"], m_bb[:, 0]), ("call", s_bb["call"], m_bb[:, 1]),
                              ("raise", s_bb["raise"], m_bb[:, 2] + m_bb[:, 3])):
                row[f"solver_bb_{a}_vs_{size}"] = float((sv[sel] * w_bb_solver[sel]).sum() / w_bb_solver[sel].sum())
                row[f"model_bb_{a}_vs_{size}"] = float((mv[sel] * w_bb[sel]).sum() / w_bb[sel].sum())
        rows.append(row)
        for b in range(fg.NB):
            mb = btn_b == b
            if w_btn[mb].sum() > 0:
                bucket_rows.append({"open": s, "flop": flop, "player": "btn", "bucket": fg.BUCKETS[b],
                                    "weight": w_btn[mb].sum(),
                                    "solver_bet": float(((1 - s_btn.check[mb]) * w_btn[mb]).sum() / w_btn[mb].sum()),
                                    "model_bet": float(((1 - m_btn[mb, 0]) * w_btn[mb]).sum() / w_btn[mb].sum())})
            for k in (0, 1):
                mo = (bo == b) & (s_bb["size"] == k).to_numpy()
                if w_bb_solver[mo].sum() > 0:
                    bucket_rows.append({"open": s, "flop": flop, "player": f"bb_vs_{'small' if k == 0 else 'big'}",
                                        "bucket": fg.BUCKETS[b], "weight": w_bb_solver[mo].sum(),
                                        "solver_fold": float((s_bb["fold"][mo] * w_bb_solver[mo]).sum() / w_bb_solver[mo].sum()),
                                        "model_fold": float((m_bb[mo, 0] * w_bb[mo]).sum() / max(w_bb[mo].sum(), 1e-12)),
                                        "solver_raise": float((s_bb["raise"][mo] * w_bb_solver[mo]).sum() / w_bb_solver[mo].sum()),
                                        "model_raise": float(((m_bb[mo, 2] + m_bb[mo, 3]) * w_bb[mo]).sum() / max(w_bb[mo].sum(), 1e-12))})
    return pd.DataFrame(rows), pd.DataFrame(bucket_rows)


# ----------------------------------------------------------------------------- report

def report():
    pd.set_option("display.width", 250)
    models = {s: load(s) for s in OPENS if (CACHE / f"model_{tag(s)}.npz").exists()}

    # 1. BB vs each open size: preflop and flop / turn / river choices with sizes.
    rows = []
    for s, d in models.items():
        r = ranges_for(s)
        row = {"open_bb": s, "bb_fold": float(r["bb_fold"]), "bb_call": float(r["bb_call"]),
               "bb_3bet": float(r["bb_3bet"]), "bb_3bet_to": float(r["threebet_to"]),
               "bb_net_per_called_pot": s - float(d["value_btn"])}
        for st, street in enumerate(fg.STREETS):
            f, _ = node_freq(d, st, "bb_first")
            leads = SIZES[st]["lead"]
            row[f"{street}_bb_leads"] = 1 - f[0]
            for j, sz in enumerate(leads):
                row[f"{street}_bb_lead_{SIZE_NAMES[sz]}"] = f[1 + j]
            for k, sz in enumerate(SIZES[st]["bet"]):
                f, _ = node_freq(d, st, "bb_vs_bet", lambda sh, k=k: mask(sh, size=k))
                row[f"{street}_vs_{SIZE_NAMES[sz]}_fold"] = f[0]
                row[f"{street}_vs_{SIZE_NAMES[sz]}_call"] = f[1]
                row[f"{street}_vs_{SIZE_NAMES[sz]}_raise_2.5x"] = f[2]
                row[f"{street}_vs_{SIZE_NAMES[sz]}_raise_4x"] = f[3]
            f, _ = node_freq(d, st, "btn_vs_check")
            row[f"{street}_btn_bets_when_checked_to"] = 1 - f[0]
        rows.append(row)
    bbt = pd.DataFrame(rows)
    bbt.round(4).to_csv(OUT / "bb_by_open_size.csv", index=False)

    labels = [f"{s:g}bb" for s in bbt.open_bb]
    for street, sz in (("flop", "1/3"), ("flop", "3/4"), ("turn", "1/2"), ("turn", "pot"), ("river", "pot")):
        p = f"{street}_vs_{sz}"
        stacked_bars(labels, {"fold": bbt[f"{p}_fold"], "call": bbt[f"{p}_call"], "raise 2.5x": bbt[f"{p}_raise_2.5x"],
                              "raise 4x": bbt[f"{p}_raise_4x"]},
                     f"BB vs a {sz}-pot BTN bet on the {street}, by open size", "Share of bets faced",
                     OUT / f"bb_vs_{street}_{sz.replace('/', '')}.png",
                     colors={"fold": GRAY, "call": BLUE, "raise 2.5x": ORANGE, "raise 4x": DARK_ORANGE})
    leads = {}
    for st, street in enumerate(fg.STREETS):
        for sz in SIZES[st]["lead"]:
            leads[f"{street} lead {SIZE_NAMES[sz]}"] = list(bbt[f"{street}_bb_lead_{SIZE_NAMES[sz]}"])
    heatmap(pd.DataFrame(leads, index=labels).T, "How often the BB leads (and how big), by BTN open size",
            "BTN open size", "BB lead", OUT / "bb_leads_by_open.png", fmt="pct", figsize=(7, 4))

    # 2. BTN at 2.25: c-bet, turn barrel after a called c-bet, river bets after a called barrel.
    d = models[2.25]
    btn_rows = []
    groups = flop_feature_groups()
    for gname, texs in [("all flops", None)] + list(groups.items()):
        f, w = node_freq(d, 0, "btn_vs_check", lambda sh, texs=texs: mask(sh, tex=texs))
        btn_rows.append({"street": "flop", "group": gname, "check": f[0], "bet small": f[1], "bet big": f[2]})
    for b, name in enumerate(fg.BUCKETS):
        f, w = node_freq(d, 0, "btn_vs_check", lambda sh, b=b: mask(sh, buckets=[b]))
        btn_rows.append({"street": "flop", "group": name, "check": f[0], "bet small": f[1], "bet big": f[2]})
    btn_lines = [3, 4, 5]                                    # previous street: BTN was the aggressor
    for b, name in enumerate(fg.BUCKETS):
        f, w = node_freq(d, 1, "btn_vs_check", lambda sh, b=b: mask(sh, lines=btn_lines, buckets=[b]))
        btn_rows.append({"street": "turn (after a called c-bet)", "group": name, "check": f[0],
                         "bet small": f[1], "bet big": f[2], "weight": w})
    for b, name in enumerate(fg.RIVER_BUCKETS):
        f, w = node_freq(d, 2, "btn_vs_check", lambda sh, b=b: mask(sh, lines=btn_lines, buckets=[b]))
        btn_rows.append({"street": "river (after a called barrel)", "group": name, "check": f[0],
                         "bet 1/2": f[1], "bet pot": f[2], "bet 1.5x": f[3], "weight": w})
    bt = pd.DataFrame(btn_rows)
    bt.round(3).to_csv(OUT / "btn_225_plan.csv", index=False)
    fl = bt[(bt.street == "flop") & bt.group.isin(fg.BUCKETS)]
    stacked_bars(list(fl.group), {"check": fl.check, "bet 1/3": fl["bet small"], "bet 3/4": fl["bet big"]},
                 "BTN flop c-bet at 2.25bb, by hand", "Share of flops", OUT / "btn_flop_cbet_by_hand.png",
                 colors={"check": GRAY, "bet 1/3": "#86b6ef", "bet 3/4": "#104281"})
    fb = bt[(bt.street == "flop") & ~bt.group.isin(fg.BUCKETS)]
    stacked_bars(list(fb.group), {"check": fb.check, "bet 1/3": fb["bet small"], "bet 3/4": fb["bet big"]},
                 "BTN flop c-bet at 2.25bb, by board", "Share of flops", OUT / "btn_flop_cbet_by_board.png",
                 colors={"check": GRAY, "bet 1/3": "#86b6ef", "bet 3/4": "#104281"})
    tu = bt[(bt.street.str.startswith("turn")) & (bt.weight > 50)]
    if len(tu):
        stacked_bars(list(tu.group), {"check": tu.check, "barrel 1/2": tu["bet small"], "barrel pot": tu["bet big"]},
                     "BTN turn barrel after its c-bet was called (2.25bb)", "Share of turns",
                     OUT / "btn_turn_barrel.png", colors={"check": GRAY, "barrel 1/2": "#86b6ef", "barrel pot": "#104281"})
    rv = bt[(bt.street.str.startswith("river")) & (bt.weight > 50)]
    if len(rv):
        stacked_bars(list(rv.group), {"check": rv.check, "bet 1/2": rv["bet 1/2"], "bet pot": rv["bet pot"],
                                      "bet 1.5x": rv["bet 1.5x"]},
                     "BTN river bet after its turn barrel was called (2.25bb)", "Share of rivers",
                     OUT / "btn_river_bets.png",
                     colors={"check": GRAY, "bet 1/2": "#86b6ef", "bet pot": "#2a78d6", "bet 1.5x": "#104281"})

    # 3. Most probable full betting lines at 2.25 (share of all called pots), with who holds what.
    keys, vals = d["line_keys"], d["line_vals"]
    total = vals.sum()
    lines = pd.DataFrame({"line": keys, "prob": vals.sum(1) / total,
                          **{f"btn_{c}": vals[:, i] / np.maximum(vals.sum(1), 1e-12) for i, c in enumerate(CLASSES)}})
    lines = lines.sort_values("prob", ascending=False)
    lines["readable"] = [line_text(k) for k in lines.line]
    lines.head(40).round(4).to_csv(OUT / "top_lines_225.csv", index=False)
    # Lines where the BTN c-bets, gets called, barrels the turn and gets called: what happens on the river?
    parts = lines.line.str.split("/")
    barrel = lines[(parts.str.len() == 3) & parts.str[0].str.match(r"^xb\dc$") & parts.str[1].str.match(r"^xb\dc$")]
    barrel = barrel.assign(share_of_barrel_lines=barrel.prob / barrel.prob.sum())
    barrel.head(15).round(4).to_csv(OUT / "top_cbet_barrel_lines_225.csv", index=False)
    print(barrel.head(10)[["share_of_barrel_lines", "readable"] + [f"btn_{c}" for c in CLASSES]].round(3)
          .to_string(index=False))
    air_rank = pd.DataFrame({"line": keys, "prob_air": vals[:, 3] / vals[:, 3].sum()}).sort_values("prob_air",
                                                                                                   ascending=False)
    air_rank["readable"] = [line_text(k) for k in air_rank.line]
    air_rank.head(15).round(4).to_csv(OUT / "top_btn_bluff_lines_225.csv", index=False)

    # 4. TexasSolver comparisons.
    import solver_compare as sc0
    flops225 = sc0.FLOPS
    pf1, bk1 = model_vs_solver(2.25, d, flops225)
    pfs, bks = [pf1], [bk1]
    if 3.0 in models:
        pf2, bk2 = model_vs_solver(3.0, models[3.0], SOLVER_FLOPS)
        pfs.append(pf2)
        bks.append(bk2)
    pf, bk = pd.concat(pfs), pd.concat(bks)
    pf["solver_btn_bet"] = 1 - pf.solver_btn_check
    pf["model_btn_bet"] = 1 - pf.model_btn_check
    pf.round(3).to_csv(OUT / "solver_per_flop.csv", index=False)
    bk.round(3).to_csv(OUT / "solver_per_bucket.csv", index=False)
    agree = []
    for o, g in pf.groupby("open"):
        for name, col in [("BTN c-bets", "btn_bet"), ("BTN bets big", "btn_big"), ("BTN bets small", "btn_small"),
                          ("BB folds vs 1/3", "bb_fold_vs_small"), ("BB calls vs 1/3", "bb_call_vs_small"),
                          ("BB raises vs 1/3", "bb_raise_vs_small"), ("BB folds vs 3/4", "bb_fold_vs_big"),
                          ("BB calls vs 3/4", "bb_call_vs_big"), ("BB raises vs 3/4", "bb_raise_vs_big")]:
            sv, mv = g[f"solver_{col}"], g[f"model_{col}"]
            agree.append({"open": o, "decision": name, "solver_avg": sv.mean(), "model_avg": mv.mean(),
                          "mean_abs_diff": (sv - mv).abs().mean(),
                          "correlation": sv.corr(mv) if len(g) > 2 else np.nan})
    ag = pd.DataFrame(agree)
    ag.round(3).to_csv(OUT / "solver_agreement.csv", index=False)
    p225 = pf[pf.open == 2.25]
    scatter_compare(p225.solver_btn_bet, p225.model_btn_bet, list(p225.flop),
                    "BTN c-bet per flop at 2.25bb: TexasSolver vs whole-hand model", "TexasSolver", "Whole-hand model",
                    OUT / "solver_cbet_scatter_225.png")
    if (pf.open == 3.0).any():
        p3 = pf[pf.open == 3.0]
        scatter_compare(p3.solver_btn_bet, p3.model_btn_bet, list(p3.flop),
                        "BTN c-bet per flop at 3.0bb: TexasSolver vs whole-hand model", "TexasSolver",
                        "Whole-hand model", OUT / "solver_cbet_scatter_3.png")
    bb_small = bk[(bk.player == "bb_vs_small") & (bk.open == 2.25)]
    agg = bb_small.groupby("bucket").apply(lambda g: pd.Series({
        c: (g[c] * g.weight).sum() / g.weight.sum() for c in ("solver_fold", "model_fold", "solver_raise", "model_raise")}))
    agg = agg.reindex([b for b in fg.BUCKETS if b in agg.index])
    grouped_bars(list(agg.index), {"solver": list(agg.solver_raise), "whole-hand model": list(agg.model_raise)},
                 "BB check-raise vs a 1/3-pot c-bet by hand (2.25bb): solver vs whole-hand model",
                 "Check-raise frequency", OUT / "solver_xr_by_bucket_225.png")
    grouped_bars(list(agg.index), {"solver": list(agg.solver_fold), "whole-hand model": list(agg.model_fold)},
                 "BB fold vs a 1/3-pot c-bet by hand (2.25bb): solver vs whole-hand model", "Fold frequency",
                 OUT / "solver_fold_by_bucket_225.png")

    print(bbt.round(3).T.to_string())
    print(bt.round(2).to_string(index=False))
    print(lines.head(15)[["prob", "readable"] + [f"btn_{c}" for c in CLASSES]].round(3).to_string(index=False))
    print(air_rank.head(8).round(3).to_string(index=False))
    print(pf.round(2).to_string(index=False))
    print(ag.round(3).to_string(index=False))
    print(agg.round(2).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"train": train, "solve": solve, "report": report}[cmd]()
