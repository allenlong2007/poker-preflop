"""Node storage for exact MCCFR artifacts (design: docs/storage-bounded-continuation.md).

Schema 1 stores each information set's canonical key, action labels, regrets and average-strategy sums as JSON text.
Schema 2 stores the same values losslessly and ~5-7x smaller:
  - a 16-byte BLAKE2b digest of the canonical key (plus an independent 64-bit check value that turns any digest
    collision into a hard error instead of merged regrets);
  - deduplicated action menus;
  - regrets / sums as little-endian float64 BLOBs (bit-exact; NULL = all zero);
  - the full key text only for nodes at or above `key_text_floor` updates (every node that can ever be served) and
    for coverage targets, so a served policy is always matched against its literal key.
Both schemas expose the same NodeStore interface, so the training algorithm never depends on the storage format.
"""
import ctypes
import ctypes.util
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path

import numpy as np

KEY_TEXT_FLOOR = 100        # schema 2: nodes with this many updates keep their literal key text (served minimum >= this)


def digest(key):
    return hashlib.blake2b(key.encode(), digest_size=16).digest()


def check(key):
    """Independent 63-bit check value (SHA-256 based) stored next to the digest."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") >> 1


def pack(values):
    values = np.asarray(values, dtype="<f8")
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite value cannot be stored")
    return None if not values.view(np.uint64).any() else values.tobytes()   # NULL only when every bit is zero (keeps -0.0)


def unpack(blob, n):
    if blob is None:
        return np.zeros(n)
    values = np.frombuffer(blob, dtype="<f8")
    if values.shape != (n,):
        raise ValueError("stored vector length does not match its action menu")
    return values.astype(float)


class StoreV1:
    """Original JSON-text rows (kept so existing artifacts stay readable and migratable)."""
    schema = 1

    def __init__(self, db):
        self.db = db

    def create(self, key_text_floor=None):
        self.db.execute("CREATE TABLE IF NOT EXISTS nodes (key TEXT PRIMARY KEY, actions TEXT NOT NULL, "
                        "regrets TEXT NOT NULL, total TEXT NOT NULL, visits INTEGER NOT NULL, "
                        "updates INTEGER NOT NULL, street INTEGER NOT NULL)")

    def get(self, key):
        row = self.db.execute("SELECT actions,regrets,total,visits,updates,street FROM nodes WHERE key=?",
                              (key,)).fetchone()
        if row is None:
            return None
        return (tuple(json.loads(row[0])), np.array(json.loads(row[1]), dtype=float),
                np.array(json.loads(row[2]), dtype=float), row[3], row[4], row[5])

    def put(self, key, actions, regrets, total, visits, updates, street):
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
                        (key, json.dumps(list(actions)), json.dumps(np.asarray(regrets).tolist(), allow_nan=False),
                         json.dumps(np.asarray(total).tolist(), allow_nan=False), visits, updates, street))

    def served(self, key):
        """(actions, sums, visits, updates, key_verified) for serving, or None; schema 1 keys are literal text."""
        row = self.get(key)
        return None if row is None else (row[0], row[2], row[3], row[4], True)

    def count(self):
        return self.db.execute("SELECT count(*) FROM nodes").fetchone()[0]

    def validate(self):
        """Cheap layout check (no scan): the expected columns exist."""
        self.db.execute("SELECT key,actions,regrets,total,visits,updates,street FROM nodes LIMIT 0")

    def dump(self):
        """Every node, in a canonical order, as comparable plain values (tests and verification)."""
        return [(k, tuple(json.loads(a)), tuple(json.loads(r)), tuple(json.loads(t)), v, u, s) for k, a, r, t, v, u, s
                in self.db.execute("SELECT key,actions,regrets,total,visits,updates,street FROM nodes ORDER BY key")]

    def key_items(self, min_updates=0, street=None):
        """(key, street, updates) for nodes whose literal key is stored."""
        return self.db.execute("SELECT key,street,updates FROM nodes WHERE updates>=? AND (? IS NULL OR street=?) "
                               "ORDER BY key", (min_updates, street, street)).fetchall()


class StoreV2:
    schema = 2

    def __init__(self, db, key_text_floor=KEY_TEXT_FLOOR):
        self.db = db
        self.floor = key_text_floor
        self._menus, self._menu_ids = {}, {}

    def create(self, key_text_floor=None):
        if key_text_floor is not None:
            self.floor = key_text_floor
        self.db.execute("CREATE TABLE IF NOT EXISTS menus (id INTEGER PRIMARY KEY, actions TEXT NOT NULL UNIQUE)")
        self.db.execute("CREATE TABLE IF NOT EXISTS nodes (digest BLOB PRIMARY KEY, check_value INTEGER NOT NULL, "
                        "menu INTEGER NOT NULL, regrets BLOB, total BLOB, visits INTEGER NOT NULL, "
                        "updates INTEGER NOT NULL, street INTEGER NOT NULL) WITHOUT ROWID")
        self.db.execute("CREATE TABLE IF NOT EXISTS node_keys (digest BLOB PRIMARY KEY, key TEXT NOT NULL) WITHOUT ROWID")

    def _menu(self, menu_id):
        if menu_id not in self._menus:
            text = self.db.execute("SELECT actions FROM menus WHERE id=?", (menu_id,)).fetchone()
            if text is None:
                raise ValueError("node references a missing action menu")
            self._menus[menu_id] = tuple(json.loads(text[0]))
        return self._menus[menu_id]

    def _menu_id(self, actions):
        text = json.dumps(list(actions))
        if text not in self._menu_ids:
            row = self.db.execute("SELECT id FROM menus WHERE actions=?", (text,)).fetchone()
            if row is None:
                self.db.execute("INSERT INTO menus (actions) VALUES (?)", (text,))
                row = self.db.execute("SELECT id FROM menus WHERE actions=?", (text,)).fetchone()
            self._menu_ids[text] = row[0]
            self._menus[row[0]] = tuple(actions)
        return self._menu_ids[text]

    def rollback(self):
        """Menu ids cached from an uncommitted transaction are not valid after a rollback."""
        self._menus, self._menu_ids = {}, {}

    def get(self, key):
        row = self.db.execute("SELECT check_value,menu,regrets,total,visits,updates,street FROM nodes WHERE digest=?",
                              (digest(key),)).fetchone()
        if row is None:
            return None
        if row[0] != check(key):
            raise ValueError("information-set digest collision: two exact keys share one digest")
        actions = self._menu(row[1])
        return actions, unpack(row[2], len(actions)), unpack(row[3], len(actions)), row[4], row[5], row[6]

    def put(self, key, actions, regrets, total, visits, updates, street):
        d = digest(key)
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                        (d, check(key), self._menu_id(actions), pack(regrets), pack(total), visits, updates, street))
        if updates >= self.floor:
            self.db.execute("INSERT OR IGNORE INTO node_keys VALUES (?,?)", (d, key))

    def served(self, key):
        """(actions, sums, visits, updates, key_verified). Serving requires the literal key stored with the node:
        a digest match alone never serves a policy (the reader treats an unverified ready node as invalid)."""
        row = self.get(key)
        if row is None:
            return None
        text = self.db.execute("SELECT key FROM node_keys WHERE digest=?", (digest(key),)).fetchone()
        return row[0], row[2], row[3], row[4], text is not None and text[0] == key

    def count(self):
        return self.db.execute("SELECT count(*) FROM nodes").fetchone()[0]

    def validate(self):
        self.db.execute("SELECT digest,check_value,menu,regrets,total,visits,updates,street FROM nodes LIMIT 0")
        self.db.execute("SELECT id,actions FROM menus LIMIT 0")
        self.db.execute("SELECT digest,key FROM node_keys LIMIT 0")

    def dump(self):
        out = []
        for d, c, m, r, t, v, u, s in self.db.execute(
                "SELECT digest,check_value,menu,regrets,total,visits,updates,street FROM nodes ORDER BY digest"):
            actions = self._menu(m)
            out.append((d, actions, tuple(unpack(r, len(actions))), tuple(unpack(t, len(actions))), v, u, s))
        return out

    def key_items(self, min_updates=0, street=None):
        return self.db.execute("SELECT k.key,n.street,n.updates FROM node_keys k JOIN nodes n ON n.digest=k.digest "
                               "WHERE n.updates>=? AND (? IS NULL OR n.street=?) ORDER BY k.key",
                               (min_updates, street, street)).fetchall()


def open_store(db, schema, key_text_floor=KEY_TEXT_FLOOR):
    if schema == 1:
        return StoreV1(db)
    if schema == 2:
        return StoreV2(db, key_text_floor)
    raise ValueError(f"unknown exact storage schema {schema}")


def storage_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- snapshots

def _clonefile():
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fn = libc.clonefile
    except (OSError, AttributeError, TypeError):
        return None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def clone_snapshot(source, destination):
    """Copy-on-write clone (APFS) of a consistent candidate into a standalone DELETE-journal snapshot.
    The caller must have checkpointed the WAL (TRUNCATE) with no write in progress. Returns False when cloning is
    unsupported here (the caller falls back to a full sqlite3 backup)."""
    fn = _clonefile()
    if fn is None:
        return False
    destination = Path(destination)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".clone")
    try:
        if fn(os.fsencode(Path(source)), os.fsencode(temporary), 0) != 0:
            return False
        with sqlite3.connect(temporary) as db:
            if db.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() != "delete":
                raise ValueError("snapshot clone could not leave WAL mode")
        os.replace(temporary, destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(temporary) + suffix).unlink(missing_ok=True)


def allocated_bytes(path):
    """Blocks actually allocated (clones share blocks, so st_size overstates their cost)."""
    path = Path(path)
    return path.stat().st_blocks * 512 if path.exists() else 0
