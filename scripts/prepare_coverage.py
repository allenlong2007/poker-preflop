"""Prepare exact missing-cell curriculum from the queue and action-time archives."""
import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.coverage import CoverageStore, canonical_context, identity
from preflop.exactgame import Config, DECK, cards, from_context

FRACTIONS = (.1, .2, .25, 1 / 3, .4, .5, .6, .75, 1., 1.25, 1.5, 2., 3., 4.)


def hand_contexts(record):
    if not record.get("hero_hole") or not record.get("hero_pos"):
        return
    hero = record["hero_pos"]
    history = record.get("history", [])
    for street, actions in enumerate(history):
        for i, (player, _, _) in enumerate(actions):
            if player != hero:
                continue
            yield {"hero_position": hero, "hero_hole": record["hero_hole"],
                   "board": record.get("board", "")[:0 if street == 0 else (street + 2) * 2],
                   "effective_stack_bb": min(record["stacks"].values()),
                   "history": history[:street] + [actions[:i]] + [[] for _ in range(3 - street)]}


def prepare(queue, decision_files, hand_files, chip_bb=.001, expand_private=False):
    errors, candidates, sources = Counter(), {}, {}
    with CoverageStore(queue) as store:
        for row in store.targets():
            candidates[row["id"]] = row["context"]
        for mode, paths in (("decisions", decision_files), ("hands", hand_files)):
            for path in map(Path, paths):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for line in stream:
                        digest.update(line)
                        try:
                            row = json.loads(line)
                            contexts = hand_contexts(row) if mode == "hands" else [
                                row.get("context") or row.get("decision", {}).get("diagnostics", {}).get("context")]
                            for c in contexts:
                                if not c or not c.get("hero_hole"):
                                    errors["missing_action_time_context"] += 1
                                    continue
                                key, text = identity(c)
                                candidates[key] = json.loads(text)
                        except (ValueError, KeyError, TypeError, AttributeError):
                            errors["invalid_archive_record"] += 1
                sources[str(path.resolve())] = digest.hexdigest()
        base = Config(chip_bb=chip_bb)
        valid = {}
        for c in candidates.values():
            try:
                c = canonical_context(c)
                from_context(base, c, np.random.default_rng(0), known_hole=True)
                key, _ = identity(c)
                valid[key] = c
            except (ValueError, KeyError, TypeError) as error:
                reason = f"unsupported:{error}"
                errors[reason] += 1
                # Keep reconstructable exact requests visible in the queue even
                # when this game model cannot train them.
                try:
                    key, _ = identity(c)
                    store.enqueue(c, reason, import_key=f"prepare:{key}")
                except (ValueError, KeyError, TypeError):
                    pass
        if expand_private:
            roots = {}
            for c in valid.values():
                public = {k: v for k, v in c.items() if k != "hero_hole"}
                roots[json.dumps(public, sort_keys=True)] = public
            for public in roots.values():
                deck = [c for c in DECK if c not in cards(public["board"])]
                for combo in itertools.combinations(deck, 2):
                    c = {**public, "hero_hole": "".join(combo)}
                    key, _ = identity(c)
                    valid[key] = c
        if not valid:
            raise ValueError(f"no supported exact targets: {dict(errors)}")
        for key, c in valid.items():
            store.enqueue(c, "archive_gap", import_key=f"prepare:{key}")
    sizes = sorted({float(to) for c in valid.values() for st in c["history"] for _, kind, to in st
                    if kind in ("bet", "raise")})
    config = Config(chip_bb=chip_bb, fractions=FRACTIONS,
                    stacks_bb=tuple(sorted({c["effective_stack_bb"] for c in valid.values()})), extra_sizes_bb=tuple(sizes))
    targets = [valid[key] for key in sorted(valid)]
    payload = {"config": asdict(config), "targets": targets}
    return {**payload, "schema_version": 1, "manifest_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "sources_sha256": sources, "rejected_or_missing": dict(errors), "full_game_complete": False,
            "scope": "all_compatible_private_combos_at_requested_public_roots" if expand_private else "queued_and_archived_exact_cells",
            "target_count": len(targets), "expand_private": expand_private,
            "board_universe_per_hero_combo": {"flop": 19600, "turn": 921200, "river": 42375200}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("output/pokernow/coverage.sqlite"))
    parser.add_argument("--decisions", type=Path, default=Path("output/pokernow/decisions.jsonl"))
    parser.add_argument("--hands", type=Path, default=Path("output/pokernow/hands"))
    parser.add_argument("--chip-bb", type=float, default=.001)
    parser.add_argument("--expand-private", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.queue, [args.decisions] if args.decisions.exists() else [],
                   sorted(args.hands.glob("*.jsonl")), args.chip_bb, args.expand_private)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: v for k, v in plan.items() if k not in ("targets", "sources_sha256")}, indent=2))


if __name__ == "__main__":
    main()
