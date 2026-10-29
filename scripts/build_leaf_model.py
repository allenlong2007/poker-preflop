"""Fit and validate the flop-leaf value model (preflop/leafvalue.py) on recorded legacy-engine rollouts.

    python scripts/build_leaf_model.py --out output/exact/leaf [--samples 600000] [--holdout 60000]
           [--pinned 200] [--pinned-rollouts 200] [--workers 12] [--seed 29102026]

Leaf states: a random deal, a stack log-uniform in [6, 110]bb and a random common preflop line that ends at the flop
(limp-check, open-call, limp-raise-call, 3-bet-call, 4-bet-call; sizes drawn around the usual live ranges). Each
sample is ONE legacy rollout from that leaf (both seats legacy, as in exactdepth). Validation:
  holdout   calibration of predicted vs realised g by prediction decile and by aggressor (rollout noise averages out)
  pinned    fixed leaf states with many rollouts each: model error vs the states' own rollout means (and their se)
Writes leaf_model.json (model + validation + continuation fingerprint) and rows.npz.
"""
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflop.exactdepth import ENGINE, continuation_fingerprint, legacy_rollout  # noqa: E402
from preflop.exactgame import DECK, Config, State  # noqa: E402
from preflop.exactpreflop import legacy_fingerprint  # noqa: E402
from preflop.leafvalue import (AGGRESSORS, aggressor, class_combos, features, fit, hand_class)  # noqa: E402

CONFIG = Config()
_E = {}


def _equity_row(args):
    i, draws, seed = args
    import eval7
    rng = np.random.default_rng(seed)
    row = np.zeros(169)
    for j in range(169):
        wins = 0.0
        pairs = [(a, b) for a in class_combos(i) for b in class_combos(j) if not set(a) & set(b)]
        for _ in range(draws):
            a, b = pairs[int(rng.integers(len(pairs)))]
            rest = [c for c in DECK if c not in a and c not in b]
            board = [eval7.Card(rest[k]) for k in rng.choice(len(rest), 5, replace=False)]
            x = eval7.evaluate([eval7.Card(c) for c in a] + board)
            y = eval7.evaluate([eval7.Card(c) for c in b] + board)
            wins += 1.0 if x > y else 0.5 if x == y else 0.0
        row[j] = wins / draws
    return i, row


def class_equity(workers, draws, seed):
    table = np.zeros((169, 169))
    with Pool(workers) as pool:
        for i, row in pool.imap_unordered(_equity_row, [(i, draws, seed + i) for i in range(169)]):
            table[i] = row
    return (table + (1 - table.T)) / 2                       # symmetrise the two Monte Carlo estimates


def sample_leaf(rng):
    """A random flop leaf (both seats still in, nobody all-in), or None if the drawn line is not legal here."""
    stack_bb = round(float(np.exp(rng.uniform(np.log(6), np.log(110)))), 2)
    state = State.deal(CONFIG, rng, stack_bb=stack_bb)
    chips = CONFIG.chips
    line = rng.choice(["limp", "open", "limp_raise", "three_bet", "four_bet"], p=[.15, .35, .1, .28, .12])
    open_to = chips(round(float(rng.uniform(2, 4)) * 4) / 4)
    try:
        if line == "limp":
            plan = ["call", "check"]
        elif line == "open":
            plan = [f"to:{open_to}", "call"]
        elif line == "limp_raise":
            plan = ["call", f"to:{chips(round(float(rng.uniform(3, 5.5)) * 2) / 2)}", "call"]
        else:
            three = int(open_to * rng.uniform(3, 4.5))
            plan = [f"to:{open_to}", f"to:{three}"]
            if line == "four_bet":
                plan.append(f"to:{int(three * rng.uniform(2.1, 2.6))}")
            plan.append("call")
        for action in plan:
            if action.startswith("to:") and int(action[3:]) >= state.cap:
                return None
            state = state.step(action, observed=True)
    except ValueError:
        return None
    if state.terminal or state.street != 1:
        return None
    return state


def _init():
    from bot.engine import Engine
    _E["engines"] = tuple(Engine(seed=0, **ENGINE) for _ in range(2))


def _row(state):
    return {"btn": hand_class(state.holes[0]), "bb": hand_class(state.holes[1]),
            "spr": (state.stack - state.spent[0]) / state.pot, "aggressor": aggressor(state),
            "pot_bb": state.pot * CONFIG.chip_bb}


def _samples(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        state = sample_leaf(rng)
        if state is None:
            continue
        end = legacy_rollout(_E["engines"], state, int(rng.integers(2 ** 63 - 1)))
        rows.append({**_row(state), "g": end.utility(0) / (state.pot * CONFIG.chip_bb)})
    return rows


def _pinned(args):
    seed, rollouts = args
    rng = np.random.default_rng(seed)
    state = None
    while state is None:
        state = sample_leaf(rng)
    g = [legacy_rollout(_E["engines"], state, int(rng.integers(2 ** 63 - 1))).utility(0) / (state.pot * CONFIG.chip_bb)
         for _ in range(rollouts)]
    return {**_row(state), "g_mean": float(np.mean(g)), "g_se": float(np.std(g, ddof=1) / math.sqrt(rollouts))}


def collect(workers, total, seed, chunk=500):
    jobs = [(seed + k, min(chunk, total - k * chunk)) for k in range(math.ceil(total / chunk))]
    rows, started = [], time.time()
    with Pool(workers, initializer=_init) as pool:
        for part in pool.imap_unordered(_samples, jobs):
            rows.extend(part)
            print(f"{len(rows)}/{total} rollouts, {len(rows) / (time.time() - started):.0f}/s", flush=True)
    return rows


def predict(coefficients, equity, rows):
    out = np.zeros(len(rows))
    for name in AGGRESSORS:
        idx = [i for i, r in enumerate(rows) if r["aggressor"] == name]
        if idx:
            part = [rows[i] for i in idx]
            b, c = [r["btn"] for r in part], [r["bb"] for r in part]
            out[idx] = features(b, c, equity[b, c], [r["spr"] for r in part]) @ np.asarray(coefficients[name])
    return out


def calibration(predicted, actual):
    order = np.argsort(predicted)
    bins = []
    for chunk in np.array_split(order, 10):
        p, a = predicted[chunk], actual[chunk]
        bins.append({"predicted": round(float(p.mean()), 4), "realised": round(float(a.mean()), 4),
                     "se": round(float(a.std(ddof=1) / math.sqrt(len(a))), 4)})
    return bins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=600_000)
    p.add_argument("--holdout", type=int, default=60_000)
    p.add_argument("--pinned", type=int, default=200)
    p.add_argument("--pinned-rollouts", type=int, default=200)
    p.add_argument("--equity-draws", type=int, default=600)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=29_102_026)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("class equity table", flush=True)
    equity = class_equity(args.workers, args.equity_draws, args.seed)
    np.save(args.out / "class_equity.npy", equity)
    train = collect(args.workers, args.samples, args.seed + 1_000)
    holdout = collect(args.workers, args.holdout, args.seed + 5_000_000)
    for r in train + holdout:
        r["equity"] = float(equity[r["btn"], r["bb"]])
    np.savez_compressed(args.out / "rows.npz", train=json.dumps(train), holdout=json.dumps(holdout))
    coefficients = fit(train)
    spr = [r["spr"] for r in train]
    predicted = predict(coefficients, equity, holdout)
    actual = np.array([r["g"] for r in holdout])
    by_aggressor = {}
    for name in AGGRESSORS:
        mask = np.array([r["aggressor"] == name for r in holdout])
        by_aggressor[name] = {"n": int(mask.sum()), "predicted": round(float(predicted[mask].mean()), 4),
                              "realised": round(float(actual[mask].mean()), 4),
                              "se": round(float(actual[mask].std(ddof=1) / math.sqrt(mask.sum())), 4)}
    with Pool(args.workers, initializer=_init) as pool:
        pinned = pool.map(_pinned, [(args.seed + 9_000_000 + k, args.pinned_rollouts) for k in range(args.pinned)])
    for r in pinned:
        r["equity"] = float(equity[r["btn"], r["bb"]])
    pin_pred = predict(coefficients, equity, pinned)
    pin_mean = np.array([r["g_mean"] for r in pinned])
    pin_se = np.array([r["g_se"] for r in pinned])
    error = pin_pred - pin_mean
    from bot.engine import Engine
    validation = {
        "holdout_r2": round(float(1 - ((actual - predicted) ** 2).mean() / actual.var()), 4),
        "holdout_calibration": calibration(predicted, actual),
        "holdout_by_aggressor": by_aggressor,
        "pinned_states": args.pinned, "pinned_rollouts": args.pinned_rollouts,
        "pinned_bias": round(float(error.mean()), 4),
        "pinned_rmse": round(float(np.sqrt((error ** 2).mean())), 4),
        "pinned_rmse_noise_corrected": round(float(np.sqrt(max(0.0, (error ** 2).mean() - (pin_se ** 2).mean()))), 4),
        "pinned_mean_se": round(float(pin_se.mean()), 4),
        "pinned_value_sd": round(float(pin_mean.std()), 4),
        "pinned_rmse_bb": round(float(np.sqrt(((error * np.array([r["pot_bb"] for r in pinned])) ** 2).mean())), 3)}
    record = {"schema": 1, "kind": "legacy_leaf_regression", "class_equity": equity.round(5).tolist(),
              "coefficients": coefficients, "spr_range": [float(np.percentile(spr, 0.5)), float(np.percentile(spr, 99.5))],
              "samples": len(train), "holdout": len(holdout), "seed": args.seed,
              "continuation": continuation_fingerprint(Engine(seed=0, **ENGINE)), "legacy": legacy_fingerprint(),
              "validation": validation}
    (args.out / "leaf_model.json").write_text(json.dumps(record))
    (args.out / "pinned.json").write_text(json.dumps([{**r, "predicted": float(v)} for r, v in zip(pinned, pin_pred)]))
    print(json.dumps(validation, indent=1))


if __name__ == "__main__":
    main()
