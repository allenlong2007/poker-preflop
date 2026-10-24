"""Durable exact-cell training queue and observed serving/fallback frequencies."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from preflop.exactgame import DECK, cards


class ExactCoverageError(RuntimeError):
    """No automatic action: the exact state lacks a validated average policy."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Exact strategy unavailable ({diagnostics['exact']['status']}); training queued")


def canonical_context(context: dict) -> dict:
    """Lossless card/order normalization only; never board/hand categories."""
    hole, board = cards(context["hero_hole"]), cards(context.get("board", ""))
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or set(hole) & set(board):
        raise ValueError("invalid exact coverage cards")
    if context["hero_position"] not in ("btn", "bb"):
        raise ValueError("invalid hero position")
    return {"hero_position": context["hero_position"], "hero_hole": "".join(sorted(hole, key=DECK.index)),
            "board": "".join(sorted(board[:3], key=DECK.index) + list(board[3:])),
            "effective_stack_bb": float(context["effective_stack_bb"]),
            "history": [[[p, k, float(to or 0) if k in ("bet", "raise") else 0] for p, k, to in street]
                        for street in context["history"]]}


def identity(context: dict) -> tuple[str, str]:
    text = json.dumps(canonical_context(context), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest(), text


class CoverageStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (kind TEXT, name TEXT, n INTEGER NOT NULL, PRIMARY KEY(kind,name))")
        self.db.execute("CREATE TABLE IF NOT EXISTS targets (key TEXT PRIMARY KEY, context TEXT NOT NULL, "
                        "occurrences INTEGER NOT NULL, reason TEXT NOT NULL, first_seen REAL, last_seen REAL)")
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def _count(self, kind, name):
        self.db.execute("INSERT INTO counts VALUES (?,?,1) ON CONFLICT(kind,name) DO UPDATE SET n=n+1", (kind, name))

    def _enqueue(self, context, reason):
        key, text = identity(context)
        now = time.time()
        self.db.execute("INSERT INTO targets VALUES (?,?,1,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                        "occurrences=occurrences+1,reason=excluded.reason,last_seen=excluded.last_seen",
                        (key, text, reason, now, now))
        return key

    def enqueue(self, context, reason="archive_gap"):
        """Import a target without pretending it was a newly observed live attempt."""
        with self.db:
            return self._enqueue(context, reason)

    def record(self, context, status, route, fallbacks=()):
        with self.db:
            self._count("route", route)
            self._count("exact_status", status)
            self._count("street", str({0: 0, 3: 1, 4: 2, 5: 3}[len(cards(context.get("board", "")))]))
            self._count("stack_bb", str(context["effective_stack_bb"]))
            raises = sum(k in ("bet", "raise") for st in context["history"] for _, k, _ in st)
            self._count("raise_count", str(raises))
            for reason in set(fallbacks):
                self._count("fallback", reason)
            if route != "exact":
                self._enqueue(context, status)

    def targets(self):
        return [{"id": key, "context": json.loads(context), "occurrences": n, "reason": reason}
                for key, context, n, reason in self.db.execute(
                    "SELECT key,context,occurrences,reason FROM targets ORDER BY occurrences DESC,key")]

    def summary(self):
        groups = {}
        for kind, name, n in self.db.execute("SELECT kind,name,n FROM counts ORDER BY kind,name"):
            groups.setdefault(kind, {})[name] = n
        routes = groups.get("route", {})
        total = sum(routes.values())
        return {"attempts": total, "exact": routes.get("exact", 0), "withheld": routes.get("withheld", 0),
                "legacy": routes.get("legacy", 0), "slices": groups,
                "fallbacks": {reason: {"count": n, "fraction_of_attempts": n / total if total else 0}
                              for reason, n in groups.get("fallback", {}).items()},
                "queued_exact_cells": self.db.execute("SELECT count(*) FROM targets").fetchone()[0],
                "coverage_scope": "observed_attempts_and_queued_exact_cells_not_full_game"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("output/pokernow/coverage.sqlite"))
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    with CoverageStore(args.path) as store:
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(store.targets(), indent=2) + "\n")
        print(json.dumps(store.summary(), indent=2))


if __name__ == "__main__":
    main()
