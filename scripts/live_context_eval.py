"""Frozen evaluation on the live decision contexts the exact candidate was trained for.

The whole-game evaluation (train_exact.py evaluate) deals random hands from the blinds, so it almost never reaches the
literal live contexts a coverage candidate holds (0.02% exact coverage for ready-2026-10-26). This evaluation starts
each match at those contexts instead:

    for each coverage target (hero seat, hero's literal hole cards, board, depth, exact action history):
        deal the villain's hole cards and the runout uniformly from the unseen cards
        play A: hero = candidate (exact policy where ready, legacy elsewhere), villain = frozen legacy baseline
        play B: the same deal, hero = legacy baseline, villain = the same baseline
        result = hero's chips in A minus B (bb)

All three engines are re-seeded identically before each play, so a deal where the candidate serves no exact decision
scores exactly 0: every non-zero difference comes from exact decisions. Units are bb per visit of the context.

Two estimates:
    uniform    villain hands uniform over compatible combos (the training prior: coverage_prior in the artifact)
    posterior  `draws` uniform deals, each weighted by the baseline villain's probability of the observed villain
               actions with its dealt hand (action kinds: fold / check / call / bet-or-raise), then `deals` of them
               resampled by weight: the baseline villain's range for the observed line

Targets are the unit of analysis: each target gets equal weight and the standard error is over target means.
This report is not wired into `train_exact.py promote`; its criteria are proposals for the user to accept or change.

Usage: python scripts/live_context_eval.py --path ready.sqlite --targets targets.json --output report.json
       [--deals 200] [--draws 2000] [--seed 2000003] [--workers 8] [--limit N]
"""
import argparse
import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflop.exactgame import Config, SEATS, from_context
from preflop.exacttrain import _metadata, _runtime, file_hash

ROOT = Path(__file__).resolve().parents[1]
AGGRESSIVE = ("bet", "raise")

# Proposed pass rule (not enforced anywhere): enough evidence, the candidate actually serves the contexts, no illegal
# actions, and at the contexts it serves it is not worse than the legacy policy by more than 0.05 bb per visit.
PROPOSED = {"min_deals_per_target": 200, "min_root_exact_share": 0.9, "illegal": 0,
            "posterior_lower_bound_bb": -0.05}

_W = {}


def _kind(kind):
    return "aggressive" if kind in AGGRESSIVE else kind


def _engines(path, seed):
    from bot.engine import Engine
    candidate = Engine(exact_required=False, seed=seed, pot_odds=True, exact_policy_path=path,
                       exact_candidate=True, exact_min_updates=1000)
    hero = Engine(exact_required=False, seed=seed, pot_odds=True, exact_policy_path=False)
    villain = Engine(exact_required=False, seed=seed, pot_odds=True, exact_policy_path=False)
    return candidate, hero, villain


def _init(path, config):
    _W["engines"] = _engines(path, 0)
    _W["config"] = Config(**config)


def _history(state):
    return [[[SEATS[p], kind, to * state.config.chip_bb] for p, kind, to in street] for street in state.history]


def _reseed(engines, seed):
    for e in engines:
        e.rng = np.random.default_rng(seed)


def _weight(state0, villain, context, engines, seed):
    """Baseline villain's probability of the observed villain action kinds, holding its dealt hand."""
    v = 1 - SEATS.index(context["hero_position"])
    weight = 1.0
    hole = "".join(state0.holes[v])
    played = [[] for _ in range(4)]
    for street, actions in enumerate(context["history"]):
        for p, kind, to in actions:
            if p == SEATS[v]:
                _reseed(engines, seed)
                board = "".join(state0.runout[:(0, 3, 4, 5)[street]])
                d = villain.decide(p, hole, board, played, sample=False,
                                   stack=state0.stack * state0.config.chip_bb)
                opts = villain._options
                weight *= sum(pr for label, pr in d.probs.items()
                              if label in opts and _kind(opts[label][0]) == _kind(kind))
                if weight <= 0:
                    return 0.0
            played[street].append([p, kind, to])
    return weight


def _play(state, hero_seat, hero, villain):
    """Play to the end; hero's result in bb, the hero's decisions served exact, and whether the first one was."""
    exact, first, illegal = 0, None, 0
    while not state.terminal:
        engine = hero if state.actor == hero_seat else villain
        d = engine.decide(SEATS[state.actor], "".join(state.holes[state.actor]), "".join(state.board),
                          _history(state), stack=state.stack * state.config.chip_bb)
        if state.actor == hero_seat:
            served = d.source == "exact"
            exact += served
            first = served if first is None else first
        if d.action in ("bet", "raise"):
            raw = round(d.to / state.config.chip_bb)
            legal = min(state.cap, max(state.minimum, raw))
            if d.source == "exact" and raw != legal:
                illegal += 1
            label = f"to:{legal}"
        else:
            label = d.action
        state = state.step(label, observed=True)
    return state.utility(hero_seat), exact, bool(first), illegal


def run_target(job):
    """Draw `draws` deals; the uniform sample is the first `deals`, the posterior sample `deals` resampled in
    proportion to their weights (systematic resampling). Each distinct deal is played once."""
    index, context, deals, draws, seed = job
    candidate, hero, villain = _W["engines"]
    engines = (candidate, hero, villain)
    config = _W["config"]
    hero_seat = SEATS.index(context["hero_position"])
    seeds, roots, weights = [], [], []
    for k in range(draws):
        s = int(np.random.SeedSequence([seed, index, k]).generate_state(1)[0])
        root = from_context(config, context, np.random.default_rng(s), known_hole=True)
        seeds.append(s)
        roots.append(root)
        weights.append(_weight(root, villain, context, engines, s))
    w = np.asarray(weights)
    picked = []
    if w.sum() > 0:
        u = (np.random.default_rng([seed, index, 1]).random() + np.arange(deals)) / deals
        picked = np.searchsorted(np.cumsum(w / w.sum()), u, side="right").clip(0, draws - 1).tolist()
    played = {}

    def play(k):
        if k not in played:
            _reseed(engines, seeds[k])
            a, exact_a, first_a, illegal_a = _play(roots[k], hero_seat, candidate, villain)
            _reseed(engines, seeds[k])
            b, _, _, _ = _play(roots[k], hero_seat, hero, villain)
            played[k] = (a - b, exact_a, first_a, illegal_a)
        return played[k]

    uniform = [play(k) for k in range(deals)]
    posterior = [play(k) for k in picked]
    ess = float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0
    return index, {"uniform": uniform, "posterior": posterior, "ess": ess,
                   "weighted_draws": int((w > 0).sum()), "illegal": sum(r[3] for r in played.values())}


def summarise(targets, results, deals, report):
    diff_u, diff_p, street_rows, street_post = [], [], {}, {}
    exact_root, exact_total, illegal, ess, no_weight = 0, 0, 0, [], 0
    for index, res in sorted(results.items()):
        rows = res["uniform"]
        d = np.array([r[0] for r in rows])
        exact_root += sum(r[2] for r in rows)
        exact_total += sum(r[1] for r in rows)
        illegal += res["illegal"]
        diff_u.append(d.mean())
        street = len(targets[index].get("board", "")) // 2
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[street]
        street_rows.setdefault(street, []).append(d.mean())
        if res["posterior"]:
            m = float(np.mean([r[0] for r in res["posterior"]]))
            diff_p.append(m)
            street_post.setdefault(street, []).append(m)
            ess.append(res["ess"])
        else:
            no_weight += 1

    def stats(values):
        v = np.asarray(values, float)
        se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        return {"targets": len(v), "mean_bb": float(v.mean()), "se_bb": se,
                "lower95_bb": float(v.mean() - 1.96 * se)}

    plays = len(results) * deals
    report.update({
        "deals_per_target": deals, "targets_evaluated": len(results), "plays": plays,
        "root_exact_share": exact_root / plays, "hero_exact_decisions": exact_total, "illegal": illegal,
        "uniform": stats(diff_u),
        "posterior": stats(diff_p) | {"median_effective_draws": float(np.median(ess)),
                                      "targets_without_weight": no_weight},
        "uniform_by_street": {k: stats(v) for k, v in street_rows.items()},
        "posterior_by_street": {k: stats(v) for k, v in street_post.items()},
    })
    errors = []
    if deals < PROPOSED["min_deals_per_target"]:
        errors.append(f"need at least {PROPOSED['min_deals_per_target']} deals per target")
    if report["root_exact_share"] < PROPOSED["min_root_exact_share"]:
        errors.append("candidate must serve the target context on at least 90% of plays")
    if illegal:
        errors.append("illegal actions observed")
    if not report["posterior"]["lower95_bb"] > PROPOSED["posterior_lower_bound_bb"]:
        errors.append("posterior 95% lower bound must exceed -0.05 bb per visit")
    report["proposed_criteria"] = PROPOSED
    report["proposed_errors"] = errors
    return report


def load_contexts(path, targets_path, seed):
    """The candidate's metadata and its coverage targets in canonical form, checked against its manifest hash."""
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        meta = _metadata(db)
    if seed == meta["seed"]:
        raise ValueError("use a held-out evaluation seed")
    targets = [t.get("context", t) for t in json.loads(Path(targets_path).read_text())["targets"]]
    config = Config(**meta["config"])
    # The targets must be the candidate's own coverage manifest (same canonical keys and hash as training).
    canonical = {}
    for c in targets:
        state = from_context(config, c, np.random.default_rng(0), known_hole=True)
        clean = {"hero_position": SEATS[state.actor], "hero_hole": "".join(state.holes[state.actor]),
                 "board": "".join(state.board), "effective_stack_bb": state.stack * config.chip_bb,
                 "history": _history(state)}
        canonical[state.key()] = json.dumps(clean, sort_keys=True)
    digest = hashlib.sha256(json.dumps(sorted(canonical.items())).encode()).hexdigest()
    if digest != meta.get("coverage_manifest"):
        raise ValueError("targets are not this candidate's coverage manifest")
    return meta, [json.loads(v) for _, v in sorted(canonical.items())], digest


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deals", type=int, default=200)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2_000_003)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    meta, contexts, digest = load_contexts(args.path, args.targets, args.seed)
    contexts = contexts[:args.limit]
    fingerprint = file_hash(args.path)
    jobs = [(i, c, args.deals, args.draws, args.seed) for i, c in enumerate(contexts)]
    results = {}
    with Pool(args.workers, initializer=_init, initargs=(str(args.path), meta["config"])) as pool:
        for n, (index, rows) in enumerate(pool.imap_unordered(run_target, jobs), 1):
            results[index] = rows
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} targets", flush=True)
    if file_hash(args.path) != fingerprint:
        raise ValueError("candidate changed during evaluation")
    report = {"evaluation": "live_context", "schema": 1, "candidate_sha256": fingerprint,
              "config_sha256": meta["config_sha256"], "iterations": meta["iterations"],
              "coverage_manifest": digest, "runtime": _runtime(), "seed": args.seed, "draws": args.draws,
              "baseline": "legacy engine (exact off, pot_odds=True), same seed as the hero each play",
              "interpretation": "candidate vs legacy at the live contexts it was trained for; bb per visit; "
                                "not exploitability, not whole-game strength"}
    summarise(contexts, results, args.deals, report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
