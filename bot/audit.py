"""Versioned, append-only decision evidence; historical imports never inherit today's build."""
import hashlib
import json
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

from bot.version import NAME, VERSION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def _fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        # Include names so replacing/renaming files also changes the build identity.
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


class AuditLog:
    """One server run; retain decision links across restarts without relabeling old hands."""

    def __init__(self, directory: Path, engine, root: Path = ROOT):
        self.directory = Path(directory)
        sources = [p for folder in ("bot", "preflop", "scripts") for p in (root / folder).rglob("*.py")]
        policies = list(engine.policy_dir.glob("*.npz"))
        preflop = root / "output" / "match" / "policy" / "preflop.json"
        if preflop.exists():
            policies.append(preflop)
        policies.extend(p for p in (root / "output" / "match" / "bluff_prior.json",
                                    root / "output" / "match" / "baseline_profile.json",
                                    root / "data" / "equity_matrix.npz") if p.exists())
        if engine.use_solver_flops:
            policies.extend(engine.flops_dir.glob("*.json"))
        self.run = {
            "schema_version": SCHEMA_VERSION, "run_id": uuid.uuid4().hex,
            "started_at": time.time(), "bot_version": VERSION, "bot_name": NAME,
            "git_commit": _git_commit(root), "source_sha256": _fingerprint(sources, root),
            "policy_sha256": _fingerprint(policies, root),
            "config": {"policy_dir": str(engine.policy_dir), "flops_dir": str(engine.flops_dir),
                       "solver_flops": engine.use_solver_flops, "translate": engine.translate,
                       "pot_odds": engine.pot_odds,
                       **{key: getattr(engine, key, None) for key in
                          ("defend_keep", "river_shift", "river_lead", "river_jam")}},
        }
        self._links = {}
        self._lock = threading.Lock()
        self._manifest_written = False
        path = self.directory / "decisions.jsonl"
        if path.exists():
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        self._remember(json.loads(line))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        logging.getLogger(__name__).warning("Skipping invalid decision log line %s", number)

    def _remember(self, row: dict) -> None:
        if not all(row.get(k) is not None for k in ("hand_id", "hero_id", "run_id", "bot_version")):
            return  # Old logs cannot establish a build/version for the hand.
        key = (row["hand_id"], row["hero_id"])
        link = (row["run_id"], row["decision"]["id"], row["bot_version"])
        links = self._links.setdefault(key, [])
        if link not in links:
            links.append(link)

    def _append(self, filename: str, row: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / filename
        # A crash may leave a partial final line. Keep it isolated from the next valid event.
        with path.open("ab+") as stream:
            stream.seek(0, 2)
            if stream.tell():
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write((json.dumps(row, allow_nan=False) + "\n").encode())

    def write_decision(self, row: dict) -> None:
        with self._lock:
            if not self._manifest_written:
                self._append("runs.jsonl", self.run)
                self._manifest_written = True
            event = {**row, "schema_version": SCHEMA_VERSION, "run_id": self.run["run_id"],
                     "bot_version": self.run["bot_version"]}
            self._append("decisions.jsonl", event)
            self._remember(event)

    def hand_metadata(self, hand_id: str, hero_id: str) -> dict | None:
        with self._lock:
            links = self._links.get((hand_id, hero_id), [])
            if not links:
                return None
            versions = {version for _, _, version in links}
            return {"schema_version": SCHEMA_VERSION,
                    "bot_version": next(iter(versions)) if len(versions) == 1 else None,
                    "run_ids": list(dict.fromkeys(run for run, _, _ in links)),
                    "decision_ids": list(dict.fromkeys(decision for _, decision, _ in links))}
