"""Bounded local exact-card training, immutable evaluation, and gated promotion.

Examples: train --hours 8; status; snapshot --output output/exact/frozen.sqlite;
evaluate --path output/exact/frozen.sqlite --output output/exact/evaluation.json.
"""
import argparse
import json
import math
import os
import shutil
import signal
import sqlite3
import sys
import time
from collections import deque
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflop.exactgame import Config, SEATS, State
from preflop.exacttrain import Trainer, _metadata, _runtime, file_hash, freeze, frozen_artifact, promotion_errors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/exact/candidate.sqlite"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_meta(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return _metadata(db)


def _focus(path, limit):
    if not path:
        return None
    contexts = deque(maxlen=limit)
    with Path(path).open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                context = row.get("context") or row.get("decision", {}).get("diagnostics", {}).get("context")
                if context and all(k in context for k in ("hero_position", "board", "history", "effective_stack_bb")):
                    contexts.append(context)
            except (ValueError, AttributeError):
                continue
    return list(contexts)


def _train(args):
    if (args.iterations is None and args.hours is None) or \
            (args.iterations is not None and args.iterations <= 0) or \
            (args.hours is not None and (not math.isfinite(args.hours) or args.hours <= 0)):
        raise ValueError("provide a finite positive iteration or hours budget")
    if args.batch < 1 or not math.isfinite(args.max_gb) or args.max_gb <= 0 or args.max_focus < 1:
        raise ValueError("invalid batch, disk budget, or focus limit")
    base = Config(**_read_meta(args.path)["config"]) if args.path.exists() else Config()
    changes = {}
    if args.chip_bb is not None:
        changes["chip_bb"] = args.chip_bb
    if args.stacks:
        changes["stacks_bb"] = tuple(map(float, args.stacks.split(",")))
    if args.fractions:
        changes["fractions"] = tuple(map(float, args.fractions.split(",")))
    if args.exhaustive:
        changes["exhaustive"] = True
    cfg = replace(base, **changes)
    focus = _focus(args.focus, args.max_focus)
    started = time.time()
    deadline = time.monotonic() + args.hours * 3600 if args.hours is not None else float("inf")
    stop = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda number, frame: stop.append(number))
    status_path = args.path.with_suffix(".status.json")
    try:
        with Trainer(args.path, cfg, seed=args.seed, focus=focus) as trainer:
            initial = trainer.status()["iterations"]
            state = {"state": "running", "pid": os.getpid(), "started_at": started,
                     "path": str(args.path.resolve()), "hours_budget": args.hours,
                     "iteration_budget": args.iterations, "max_candidate_gb": args.max_gb}
            _write_json(status_path, {**state, **trainer.status()})
            last_snapshot = last_print = time.monotonic()
            reason = "budget_complete"
            try:
                while time.monotonic() < deadline and not stop:
                    completed = trainer.status()["iterations"] - initial
                    if args.iterations is not None and completed >= args.iterations:
                        break
                    size = sum(p.stat().st_size for p in (args.path, Path(str(args.path) + "-wal")) if p.exists())
                    if size >= args.max_gb * 1024 ** 3:
                        reason = "candidate_disk_limit"
                        break
                    if shutil.disk_usage(args.path.parent).free < max(64 * 1024 ** 2, size * 1.2):
                        reason = "free_disk_limit"
                        break
                    count = min(args.batch, args.iterations - completed) if args.iterations is not None else args.batch
                    trainer.train(count)
                    trainer.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    state.update(trainer.status(), elapsed_seconds=round(time.time() - started, 2),
                                 candidate_bytes=args.path.stat().st_size)
                    _write_json(status_path, state)
                    if time.monotonic() - last_print >= 30:
                        print(json.dumps(state), flush=True)
                        last_print = time.monotonic()
                    if args.snapshot_every > 0 and time.monotonic() - last_snapshot >= args.snapshot_every:
                        try:
                            trainer.checkpoint(args.path.with_name("latest.sqlite"))
                        except BlockingIOError:
                            state["snapshot_note"] = "latest snapshot leased by an evaluator; skipped"
                        last_snapshot = time.monotonic()
                if stop:
                    reason = "signal_stopped"
                try:
                    trainer.checkpoint(args.path.with_name("latest.sqlite"))
                except BlockingIOError:
                    state["snapshot_note"] = "latest snapshot leased; candidate remains committed"
            except BaseException as error:
                state.update(state="failed", error=f"{type(error).__name__}: {error}", **trainer.status())
                _write_json(status_path, state)
                raise
            state.update(trainer.status(), state="stopped" if stop else "completed", stop_reason=reason,
                         elapsed_seconds=round(time.time() - started, 2))
            _write_json(status_path, state)
            print(json.dumps(state, indent=2))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _play(state, engines, candidate):
    counts = {"exact": 0, "total": 0, "focus_used": 0, "illegal": 0}
    while not state.terminal:
        actor, engine = state.actor, engines[state.actor]
        history = [[[SEATS[p], kind, to * state.config.chip_bb] for p, kind, to in street]
                   for street in state.history]
        d = engine.decide(SEATS[actor], "".join(state.holes[actor]), "".join(state.board), history,
                          stack=state.stack * state.config.chip_bb)
        if engine is candidate:
            counts["total"] += 1
            if d.source == "exact":
                counts["exact"] += 1
                counts["focus_used"] += d.diagnostics["exact"].get("root_id") != "whole_game"
        if d.action in ("bet", "raise"):
            raw = round(d.to / state.config.chip_bb)
            legal = min(state.cap, max(state.minimum, raw))
            if d.source == "exact" and raw != legal:
                raise ValueError("exact policy returned an illegal size")
            label = f"to:{legal}"
        else:
            label = d.action
        state = state.step(label, observed=True)
    return state.utility(0), counts


def evaluate(path, deals, seed, minimum):
    with frozen_artifact(path) as fingerprint:
        return _evaluate_frozen(path, deals, seed, minimum, fingerprint)


def _evaluate_frozen(path, deals, seed, minimum, fingerprint):
    from bot.audit import AuditLog
    from bot.engine import Engine
    if deals < 2 or minimum < 1:
        raise ValueError("at least two evaluation deals and positive update threshold required")
    meta = _read_meta(path)
    if seed == meta["seed"]:
        raise ValueError("use a held-out evaluation seed")
    cfg = Config(**meta["config"])
    candidate = Engine(seed=seed + 1, pot_odds=True, exact_policy_path=path,
                       exact_candidate=True, exact_min_updates=minimum)
    baseline = Engine(seed=seed + 2, pot_odds=True, exact_policy_path=False)
    provenance = AuditLog(Path(path).parent / "_evaluation_provenance", baseline).run
    rng = np.random.default_rng(seed)
    results, counts = [], {"exact": 0, "total": 0, "focus_used": 0, "illegal": 0}
    try:
        for _ in range(deals):
            state = State.deal(cfg, rng)
            first, c1 = _play(state, (candidate, baseline), candidate)
            second, c2 = _play(state, (baseline, candidate), candidate)
            results.append(first - second)
            for key in counts:
                counts[key] += c1[key] + c2[key]
    finally:
        if candidate.exact_policy is not None:
            candidate.exact_policy.close()
    values = np.asarray(results)
    report = {"evaluation_schema": 1, "candidate_sha256": fingerprint, "runtime": _runtime(),
              "config_sha256": meta["config_sha256"], "iterations": meta["iterations"],
              "baseline_policy_sha256": provenance["policy_sha256"],
              "evaluation_source_sha256": provenance["source_sha256"], "seed": seed, "deals": deals,
              "min_updates": minimum, "bb100": float(values.mean() * 50),
              "se_bb100": float(values.std(ddof=1) / math.sqrt(deals) * 50),
              "coverage": counts["exact"] / max(1, counts["total"]), **counts,
              "interpretation": "frozen candidate vs legacy baseline; not exploitability or equilibrium"}
    report["promotion_errors"] = promotion_errors(report)
    return report


def _promote(args):
    with frozen_artifact(args.path) as fingerprint:
        _promote_frozen(args, fingerprint)


def _promote_frozen(args, fingerprint):
    from bot.audit import AuditLog
    from bot.engine import Engine
    report = json.loads(args.report.read_text())
    if args.path.resolve() == args.output.resolve():
        raise ValueError("promotion destination must differ from evaluation artifact")
    if report.get("candidate_sha256") != fingerprint or report.get("evaluation_schema") != 1:
        raise ValueError("promotion report does not match the frozen artifact")
    errors = promotion_errors(report)
    if errors:
        raise ValueError("promotion rejected: " + "; ".join(errors))
    provenance = AuditLog(args.path.parent / "_promotion_provenance", Engine(exact_policy_path=False)).run
    if report.get("evaluation_source_sha256") != provenance["source_sha256"] or \
            report.get("baseline_policy_sha256") != provenance["policy_sha256"] or report.get("runtime") != _runtime():
        raise ValueError("promotion provenance changed; rerun frozen evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    shutil.copyfile(args.path, temporary)
    if file_hash(temporary) != fingerprint:
        temporary.unlink(missing_ok=True)
        raise ValueError("promotion source changed while copying")
    with closing(sqlite3.connect(temporary)) as db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("promotion", json.dumps(report)))
        db.commit()
    os.replace(temporary, args.output)
    print(f"Promoted {args.output}; restart the server to load it.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--path", type=Path, default=DEFAULT)
    train.add_argument("--iterations", type=int)
    train.add_argument("--hours", type=float)
    train.add_argument("--batch", type=int, default=100)
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument("--chip-bb", type=float)
    train.add_argument("--stacks")
    train.add_argument("--fractions")
    train.add_argument("--exhaustive", action="store_true")
    train.add_argument("--max-gb", type=float, default=4)
    train.add_argument("--focus", type=Path)
    train.add_argument("--max-focus", type=int, default=32)
    train.add_argument("--snapshot-every", type=float, default=900)
    status = sub.add_parser("status")
    status.add_argument("--path", type=Path, default=DEFAULT)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--path", type=Path, default=DEFAULT)
    snapshot.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--path", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--deals", type=int, default=2000)
    evaluate_parser.add_argument("--seed", type=int, default=1000003)
    evaluate_parser.add_argument("--min-updates", type=int, default=1000)
    promote = sub.add_parser("promote")
    promote.add_argument("--path", type=Path, required=True)
    promote.add_argument("--report", type=Path, required=True)
    promote.add_argument("--output", type=Path, default=ROOT / "output/exact/active.sqlite")
    args = parser.parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "status":
        status_path = args.path.with_suffix(".status.json")
        print(status_path.read_text() if status_path.exists() else json.dumps(_read_meta(args.path), indent=2))
    elif args.command == "snapshot":
        freeze(args.path, args.output)
    elif args.command == "evaluate":
        report = evaluate(args.path, args.deals, args.seed, args.min_updates)
        _write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "promote":
        _promote(args)


if __name__ == "__main__":
    main()
