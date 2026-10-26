"""Storage-bounded continuation (docs/storage-bounded-continuation.md): lossless schema 2, verified migration,
split resume fingerprints, phase-2 scheduling, clone snapshots and the ready-only export."""
import json
import sqlite3

import numpy as np
import pytest

from preflop import exactstore
from preflop.exactgame import Config
from preflop.exactmigrate import export_ready, migrate, verify
from preflop.exactstore import StoreV1, StoreV2, pack, unpack
from preflop.exacttrain import ExactPolicy, Trainer, frozen_artifact

CFG = Config(chip_bb=0.5, stacks_bb=(3,), fractions=(0.5, 1))


def context(stack=3):
    return {"hero_position": "bb", "hero_hole": "AhKd", "board": "Kh8d3c2s9h", "effective_stack_bb": stack,
            "history": [[["btn", "call", 0], ["bb", "check", 0]], [["bb", "check", 0], ["btn", "check", 0]],
                        [["bb", "check", 0], ["btn", "check", 0]], []]}


def dump(path):
    with sqlite3.connect(path) as db:
        schema = json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0])
        return (StoreV1 if schema == 1 else StoreV2)(db).dump()


def test_codec_is_bit_exact_and_stores_all_zero_vectors_as_null():
    values = np.array([304045.00000000006, -0.0, 1e-300, -608090.0000000001])
    assert np.array_equal(unpack(pack(values), 4).view(np.uint64), values.view(np.uint64))
    assert pack(np.zeros(3)) is None and pack([-0.0, 0.0]) is not None      # signed zero kept exactly
    assert np.array_equal(unpack(None, 3), np.zeros(3))
    with pytest.raises(FloatingPointError):
        pack([1.0, np.nan])
    with pytest.raises(ValueError, match="length"):
        unpack(pack([1.0, 2.0]), 3)


def test_digest_collision_is_an_error_not_merged_regrets(tmp_path, monkeypatch):
    with Trainer(tmp_path / "c.sqlite", CFG, seed=1) as trainer:
        trainer.train(5)
        key = trainer.store.key_items(0)[0][0] if trainer.store.key_items(0) else None
        trainer.store.put("A", ("fold", "call"), [1, 2], [3, 4], 1, 1, 0)
        monkeypatch.setattr(exactstore, "check", lambda k: 12345)
        with pytest.raises(ValueError, match="collision"):
            trainer.store.get("A")
        assert key is None or isinstance(key, str)


def test_menus_are_deduplicated_and_rows_are_much_smaller(tmp_path):
    a, b = tmp_path / "v1.sqlite", tmp_path / "v2.sqlite"
    with Trainer(a, CFG, seed=5, schema=1) as trainer:
        trainer.train(300)
    migrate(a, b)
    with sqlite3.connect(b) as db:
        menus = db.execute("SELECT count(*) FROM menus").fetchone()[0]
        nodes = db.execute("SELECT count(*) FROM nodes").fetchone()[0]
        v2 = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
    with sqlite3.connect(a) as db:
        db.execute("VACUUM")
        v1 = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
    assert menus < nodes / 5
    assert v2 < 0.6 * v1


def test_migration_then_resume_equals_uninterrupted_training(tmp_path):
    """The equivalence test: schema-1 training, migration, then schema-2 training is node-for-node bit-identical
    to the same number of iterations without a storage change (RNG state carries over)."""
    full, part, migrated, reference = (tmp_path / n for n in ("full.sqlite", "part.sqlite", "m.sqlite", "r.sqlite"))
    with Trainer(full, CFG, seed=41, schema=1) as trainer:
        trainer.train(20)
    with Trainer(part, CFG, seed=41, schema=1) as trainer:
        trainer.train(8)
    report = migrate(part, migrated)
    assert report["verified"] and report["verification"]["bit_identical"]
    with Trainer(migrated) as resumed:
        assert resumed.store.schema == 2
        resumed.train(12)
        assert resumed.status()["iterations"] == 20
        assert resumed.status()["infosets"] == resumed.store.count()
    migrate(full, reference)
    assert dump(migrated) == dump(reference)


def test_migration_never_overwrites_and_verification_catches_any_change(tmp_path):
    src, dst = tmp_path / "v1.sqlite", tmp_path / "v2.sqlite"
    with Trainer(src, CFG, seed=3, schema=1) as trainer:
        trainer.train(30)
    before = src.read_bytes()
    migrate(src, dst)
    assert src.read_bytes() == before                       # the source is never modified
    with pytest.raises(FileExistsError):
        migrate(src, dst)
    with pytest.raises(ValueError, match="schema-1"):
        migrate(dst, tmp_path / "again.sqlite")
    with sqlite3.connect(dst) as db:
        digest, total = db.execute("SELECT digest,total FROM nodes WHERE total IS NOT NULL LIMIT 1").fetchone()
        tampered = np.frombuffer(total, "<f8").copy()
        tampered[0] = np.nextafter(tampered[0], np.inf)     # one ulp
        db.execute("UPDATE nodes SET total=? WHERE digest=?", (tampered.tobytes(), digest))
        db.commit()
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d, pytest.raises(ValueError, match="differs"):
        verify(s, d, exactstore.KEY_TEXT_FLOOR)


def test_resume_guard_separates_algorithm_from_storage(tmp_path, monkeypatch):
    path = tmp_path / "g.sqlite"
    with Trainer(path, CFG, seed=1) as trainer:
        trainer.train(2)
    monkeypatch.setattr("preflop.exacttrain.storage_hash", lambda: "other")
    with pytest.raises(ValueError, match="storage code changed"):
        Trainer(path)
    monkeypatch.undo()
    monkeypatch.setattr("preflop.exacttrain._code_hash", lambda: "other")
    with pytest.raises(ValueError, match="training code changed"):
        Trainer(path)
    monkeypatch.undo()
    legacy = tmp_path / "legacy.sqlite"
    with Trainer(legacy, CFG, seed=1, schema=1) as trainer:
        trainer.train(1)
    with sqlite3.connect(legacy) as db:                     # an artifact from before the fingerprint split
        db.execute("DELETE FROM metadata WHERE key IN ('algorithm_sha256','storage_sha256')")
        db.execute("INSERT INTO metadata VALUES ('training_code_sha256', '\"old\"')")
    with pytest.raises(ValueError, match="migrate"):
        Trainer(legacy)


def test_reader_never_serves_a_ready_node_without_its_literal_key(tmp_path):
    path, frozen = tmp_path / "k.sqlite", tmp_path / "f.sqlite"
    from preflop.exactcoverage import CoverageTrainer
    with CoverageTrainer(path, CFG, targets=[context()], seed=22, key_text_floor=5) as trainer:
        trainer.train(20)
        key = trainer.targets[0][0]
        trainer.checkpoint(frozen)
    with ExactPolicy(frozen, min_updates=5, candidate=True) as policy:
        assert policy.lookup(context())["status"] == "ready"
    with sqlite3.connect(frozen) as db:
        db.execute("DELETE FROM node_keys")
        db.commit()
    with ExactPolicy(frozen, min_updates=5, candidate=True) as policy:
        answer = policy.lookup(context())
        assert answer["status"] == "invalid" and answer["reason"] == "unverified exact key"
    assert key


def test_phase2_guides_only_unready_targets_and_is_resume_deterministic(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    other = {**context(), "hero_hole": "QsQd"}
    paths = [tmp_path / f"p{i}.sqlite" for i in range(2)]
    for path in paths:
        with CoverageTrainer(path, CFG, targets=[context(), other], seed=9) as trainer:
            trainer.train(10)
            ready = trainer.targets[0][0]
            actions, regrets, _, visits, _, street = trainer.store.get(ready) or (
                trainer.target_actions[ready], np.zeros(len(trainer.target_actions[ready])), None, 1, 0, 3)
            trainer.store.put(ready, actions, regrets, np.ones(len(actions)), visits, 5000, street)
            trainer.db.commit()
    with CoverageTrainer(paths[0], phase2=True, min_updates=1000) as trainer:
        assert trainer.phase["phase"] == 2 and trainer.phase["start_iteration"] == 10
        assert trainer.unready(1000) == [trainer.targets[1][0]]
        guided = []
        original = trainer._target
        trainer._target = lambda: guided.append(original()[0]) or (guided[-1], trainer._contexts[guided[-1]])
        trainer.train(30)
        assert set(guided) == {trainer.targets[1][0]}      # the ready target is no longer guided
    with CoverageTrainer(paths[1], phase2=True, min_updates=1000) as trainer:
        trainer.train(7)
    with CoverageTrainer(paths[1]) as trainer:              # resume mid-block: the phase is recorded
        assert trainer.phase["phase"] == 2
        trainer.train(23)
    assert dump(paths[0]) == dump(paths[1])
    with pytest.raises(ValueError, match="readiness floor"):
        CoverageTrainer(paths[1], phase2=True, min_updates=500)


def test_clone_snapshot_is_standalone_frozen_and_readable(tmp_path):
    path, snap = tmp_path / "c.sqlite", tmp_path / "latest.sqlite"
    with Trainer(path, CFG, seed=2, key_text_floor=1) as trainer:
        trainer.train(20)
        mode = trainer.checkpoint(snap, clone=True)
        assert mode in ("clone", "backup")
        trainer.train(5)                                    # later training never changes the snapshot
    with sqlite3.connect(snap) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert json.loads(db.execute("SELECT value FROM metadata WHERE key='iterations'").fetchone()[0]) == 20
    with frozen_artifact(snap):
        with ExactPolicy(snap, min_updates=1, candidate=True) as policy:
            assert policy.store.count() > 0


def test_ready_export_keeps_only_servable_nodes_with_identical_answers(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    path, frozen, small = tmp_path / "t.sqlite", tmp_path / "f.sqlite", tmp_path / "ready.sqlite"
    with CoverageTrainer(path, CFG, targets=[context()], seed=22, key_text_floor=5) as trainer:
        trainer.train(20)
        trainer.checkpoint(frozen)
    report = export_ready(frozen, small, min_updates=5)
    assert report["nodes"] >= 1 and report["contains"] == "average_strategy_only"
    with ExactPolicy(frozen, min_updates=5, candidate=True) as full, \
            ExactPolicy(small, min_updates=5, candidate=True) as ready:
        assert ready.lookup(context()) == full.lookup(context())
        assert ready.store.count() == report["nodes"] < full.store.count()
    with sqlite3.connect(small) as db:
        assert db.execute("SELECT count(*) FROM nodes WHERE regrets IS NOT NULL").fetchone()[0] == 0
    with pytest.raises(ValueError, match="ready-only export"):
        Trainer(small)
    with pytest.raises(ValueError, match="floor"):
        export_ready(frozen, tmp_path / "x.sqlite", min_updates=1)


def test_coverage_candidate_migrates_and_resumes_identically(tmp_path):
    """The real candidate is a coverage trainer (targets table, manifest digest, RNG): migrate it mid-run."""
    from preflop.exactcoverage import CoverageTrainer
    other = {**context(), "hero_hole": "QsQd"}
    full, part, migrated, reference = (tmp_path / n for n in ("full.sqlite", "part.sqlite", "m.sqlite", "r.sqlite"))
    with CoverageTrainer(full, CFG, targets=[context(), other], seed=4, schema=1) as trainer:
        trainer.train(15)
    with CoverageTrainer(part, CFG, targets=[context(), other], seed=4, schema=1) as trainer:
        trainer.train(6)
    assert migrate(part, migrated)["verification"]["targets"] == 2
    with CoverageTrainer(migrated) as resumed:
        resumed.train(9)
        assert resumed.coverage(1)["targets"] == 2
    migrate(full, reference)
    assert dump(migrated) == dump(reference)
