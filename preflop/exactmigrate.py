"""Lossless schema-1 -> schema-2 migration of exact MCCFR artifacts, and the small ready-only export.

Design: docs/storage-bounded-continuation.md (B, F). The source is opened read-only and never modified. The
destination is built in a temporary file and appears only after every node has been verified bit-for-bit against
the source.
"""
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np

from preflop.exactstore import KEY_TEXT_FLOOR, StoreV1, StoreV2, storage_hash
from preflop.exacttrain import _code_hash, _metadata, file_hash, required_updates, stable

ROOT = Path(__file__).resolve().parents[1]
BATCH = 20000


def _ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _bits(values):
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).view(np.uint64)


def _code_at(commit):
    """The schema-1 training-code fingerprint of a git commit (to confirm which code produced the source)."""
    blobs = [subprocess.run(["git", "show", f"{commit}:preflop/{name}"], cwd=ROOT, capture_output=True,
                            check=True).stdout for name in ("exacttrain.py", "exactgame.py", "exactcoverage.py")]
    return hashlib.sha256(b"".join(blobs)).hexdigest()


def migrate(source, destination, key_text_floor=KEY_TEXT_FLOOR, source_commit=None, progress=None):
    """Convert a schema-1 artifact to schema 2 and verify it. Returns the verification report."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("destination exists; the migration never overwrites an artifact")
    if source.resolve() == destination.resolve():
        raise ValueError("destination must differ from the source")
    started = time.time()
    with closing(_ro(source)) as src:
        meta = _metadata(src)
        if meta.get("schema") != 1:
            raise ValueError("source is not a schema-1 artifact")
        if meta.get("promotion"):
            raise ValueError("refusing to migrate a promoted serving artifact")
        produced_by = None
        if source_commit:
            produced_by = _code_at(source_commit)
            if produced_by != meta.get("training_code_sha256"):
                raise ValueError("source was not produced by the training code of the given commit")
        source_sha = file_hash(source)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".migrating")
        try:
            with closing(sqlite3.connect(temporary, isolation_level=None)) as dst:   # explicit transactions only
                dst.execute("PRAGMA journal_mode=OFF")
                dst.execute("PRAGMA synchronous=OFF")
                dst.execute("PRAGMA cache_size=-1048576")        # 1 GiB page cache for random digest inserts
                dst.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                store = StoreV2(dst, key_text_floor)
                store.create(key_text_floor)
                if src.execute("SELECT name FROM sqlite_master WHERE name='coverage_targets'").fetchone():
                    dst.execute("CREATE TABLE coverage_targets (key TEXT PRIMARY KEY, context TEXT NOT NULL)")
                    dst.executemany("INSERT INTO coverage_targets VALUES (?,?)",
                                    src.execute("SELECT key,context FROM coverage_targets ORDER BY key"))
                if src.execute("SELECT name FROM sqlite_master WHERE name='baselines'").fetchone():
                    # Variance-reduction baselines are keyed by the same digest in both schemas; copied verbatim.
                    dst.execute("CREATE TABLE baselines (digest BLOB PRIMARY KEY, value BLOB) WITHOUT ROWID")
                    dst.executemany("INSERT INTO baselines VALUES (?,?)",
                                    src.execute("SELECT digest,value FROM baselines ORDER BY digest"))
                rows = 0
                dst.execute("BEGIN")
                for key, actions, regrets, total, visits, updates, street in src.execute(
                        "SELECT key,actions,regrets,total,visits,updates,street FROM nodes"):
                    store.put(key, tuple(json.loads(actions)), np.array(json.loads(regrets), dtype=float),
                              np.array(json.loads(total), dtype=float), visits, updates, street)
                    rows += 1
                    if rows % BATCH == 0:
                        dst.execute("COMMIT")
                        dst.execute("BEGIN")
                        if progress:
                            progress("write", rows)
                dst.execute("COMMIT")
                report = verify(src, dst, key_text_floor, progress)
                if report["rows"] != rows:
                    raise ValueError("row count changed during migration")
                carried = {k: v for k, v in meta.items() if k not in ("schema", "training_code_sha256")}
                carried.update(schema=2, algorithm_sha256=_code_hash(), storage_sha256=storage_hash(),
                               key_text_floor=key_text_floor)
                carried["migration"] = {
                    "from_schema": 1, "source_path": str(source.resolve()), "source_sha256": source_sha,
                    "source_iterations": meta["iterations"],
                    "source_training_code_sha256": meta.get("training_code_sha256"),
                    "source_commit": source_commit, "source_commit_verified": produced_by is not None,
                    "migration_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "verified": True, "verification": report, "seconds": round(time.time() - started, 1)}
                dst.execute("BEGIN")
                dst.executemany("INSERT INTO metadata VALUES (?,?)",
                                [(k, json.dumps(v, allow_nan=False, sort_keys=True)) for k, v in carried.items()])
                dst.execute("COMMIT")
                dst.execute("PRAGMA journal_mode=DELETE")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return carried["migration"]


def verify(src, dst, key_text_floor, progress=None):
    """Every source node must exist in the destination with the same action menu, bit-identical regrets and sums
    (signed zeros included), and the same visits / updates / street; nodes at or above the key-text floor must
    carry their literal key. Raises on the first mismatch."""
    old, new = StoreV1(src), StoreV2(dst, key_text_floor)
    rows = keyed = 0
    for key, actions, regrets, total, visits, updates, street in src.execute(
            "SELECT key,actions,regrets,total,visits,updates,street FROM nodes"):
        got = new.get(key)
        if got is None:
            raise ValueError(f"node missing after migration: {key[:80]}")
        g_actions, g_regrets, g_total, g_visits, g_updates, g_street = got
        if g_actions != tuple(json.loads(actions)) or (g_visits, g_updates, g_street) != (visits, updates, street) \
                or not np.array_equal(_bits(g_regrets), _bits(json.loads(regrets))) \
                or not np.array_equal(_bits(g_total), _bits(json.loads(total))):
            raise ValueError(f"node differs after migration: {key[:80]}")
        if updates >= key_text_floor:
            served = new.served(key)
            if not served[4]:
                raise ValueError(f"ready-eligible node lost its literal key: {key[:80]}")
            keyed += 1
        rows += 1
        if progress and rows % BATCH == 0:
            progress("verify", rows)
    if new.count() != rows or old.count() != rows:
        raise ValueError("node counts differ after migration")
    targets_src = src.execute("SELECT key,context FROM coverage_targets ORDER BY key").fetchall() \
        if src.execute("SELECT name FROM sqlite_master WHERE name='coverage_targets'").fetchone() else []
    targets_dst = dst.execute("SELECT key,context FROM coverage_targets ORDER BY key").fetchall() if targets_src else []
    if targets_src != targets_dst:
        raise ValueError("coverage targets differ after migration")
    if src.execute("SELECT name FROM sqlite_master WHERE name='baselines'").fetchone():
        if src.execute("SELECT digest,value FROM baselines ORDER BY digest").fetchall() != \
                dst.execute("SELECT digest,value FROM baselines ORDER BY digest").fetchall():
            raise ValueError("variance-reduction baselines differ after migration")
    return {"rows": rows, "keyed_rows": keyed, "targets": len(targets_src), "bit_identical": True}


def export_ready(source, destination, min_updates=1000, value_loss=None):
    """Small frozen serving/evaluation artifact: only nodes that pass the unchanged serving checks (literal key,
    >= min_updates, finite non-negative average with positive mass), average strategy only, no regrets. Everything
    else is absent, so the reader withholds it exactly as before.

    value_loss (bb, preflop-game candidates only): judge readiness by value instead of stability
    (exactpreflop.VALUE_READINESS); each kept node's loss travels in a value_checks table and the export records the
    value rule as its readiness."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("destination exists")
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".export")
    try:
        with closing(_ro(source)) as src, closing(sqlite3.connect(temporary)) as dst:
            meta = _metadata(src)
            if meta.get("promotion"):
                raise ValueError("export from a training candidate or snapshot, not a promoted artifact")
            floor = meta.get("key_text_floor", KEY_TEXT_FLOOR) if meta.get("schema") == 2 else 1
            if min_updates < max(1, floor):
                raise ValueError("min_updates is below the artifact's literal-key floor")
            old = StoreV1(src) if meta.get("schema") == 1 else StoreV2(src, floor)
            dst.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            new = StoreV2(dst, min_updates)
            new.create(min_updates)
            readiness = meta.get("readiness")
            losses = None
            if value_loss is not None:
                from preflop.exactpreflop import VALUE_READINESS, node_losses
                if "preflop_game" not in meta:
                    raise ValueError("value readiness needs a preflop-game candidate")
                losses = node_losses(src, old, meta, min_updates)
                meta["readiness_trained_under"] = readiness
                readiness = meta["readiness"] = {**VALUE_READINESS, "max_loss_bb": float(value_loss)}
                dst.execute("CREATE TABLE value_checks (key TEXT PRIMARY KEY, loss_bb REAL NOT NULL)")
            if readiness:
                dst.execute("CREATE TABLE target_checkpoints (key TEXT PRIMARY KEY, updates INTEGER NOT NULL, "
                            "probs TEXT NOT NULL, stability REAL)")
            kept = 0
            for key, street, updates in old.key_items(min_updates):
                actions, _, total, visits, n, _street = old.get(key)
                sums = np.asarray(total, float)
                if not (np.isfinite(sums).all() and (sums >= 0).all() and sums.sum() > 0) or not old.served(key)[4]:
                    continue
                # A candidate with a readiness rule exports only nodes it would serve: the per-action floor and a
                # stable average strategy; the checkpoint row travels with the node so the reader can re-check it.
                if n < required_updates(len(actions), min_updates, readiness):
                    continue
                if losses is not None:
                    if losses.get(key, math.inf) > value_loss:
                        continue
                    dst.execute("INSERT INTO value_checks VALUES (?,?)", (key, losses[key]))
                elif not stable(src, key, readiness):
                    continue
                if readiness and losses is None:
                    checkpoint = src.execute("SELECT updates,probs,stability FROM target_checkpoints WHERE key=?",
                                             (key,)).fetchone()
                    dst.execute("INSERT INTO target_checkpoints VALUES (?,?,?,?)", (key, *checkpoint))
                new.put(key, actions, np.zeros(len(actions)), sums, visits, n, street)
                kept += 1
            carried = {k: v for k, v in meta.items() if k not in ("schema", "training_code_sha256")}
            carried.update(schema=2, key_text_floor=min_updates, storage_sha256=storage_hash(),
                           algorithm_sha256=meta.get("algorithm_sha256"),
                           ready_export={"source_path": str(source.resolve()), "source_sha256": file_hash(source),
                                         "source_iterations": meta["iterations"], "min_updates": min_updates,
                                         "nodes": kept, "contains": "average_strategy_only"})
            dst.executemany("INSERT INTO metadata VALUES (?,?)",
                            [(k, json.dumps(v, allow_nan=False, sort_keys=True)) for k, v in carried.items()])
            dst.commit()
            dst.execute("PRAGMA journal_mode=DELETE")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return carried["ready_export"]
