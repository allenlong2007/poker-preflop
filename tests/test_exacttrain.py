"""Reproducible training, unbiased updates, immutable serving, and promotion contracts."""
import json
import sqlite3

import numpy as np
import pytest

from preflop.exactgame import Config
from preflop.exacttrain import ExactPolicy, Trainer, outcome_update, promotion_errors


def rows(path):
    with sqlite3.connect(path) as db:
        return db.execute("SELECT * FROM nodes ORDER BY key").fetchall()


def test_outcome_sampling_importance_weights():
    regret, average, value = outcome_update(np.array([0.25, 0.75]), 0, 0.5, 2.0, 0.1, 0.6, 0.2)
    assert regret == pytest.approx([9, -3])
    assert average == pytest.approx([0.125, 0.375])
    assert value == 1.0


def test_resume_is_identical_to_uninterrupted_training(tmp_path):
    cfg = Config(chip_bb=0.5, stacks_bb=(3,), fractions=(0.5, 1))
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    with Trainer(a, cfg, seed=41) as trainer:
        trainer.train(20)
    with Trainer(b, cfg, seed=41) as trainer:
        trainer.train(8)
    with Trainer(b) as resumed:
        resumed.train(12)
        assert resumed.status()["iterations"] == 20
        assert resumed.status()["infosets"] > 0
        assert resumed.status()["infosets"] == resumed.db.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert rows(a) == rows(b)
    with pytest.raises(ValueError, match="configuration"):
        Trainer(b, Config(chip_bb=0.5, stacks_bb=(4,)))


def test_failed_batch_rolls_back_nodes_iteration_and_rng(tmp_path, monkeypatch):
    cfg = Config(chip_bb=0.5, stacks_bb=(3,))
    with Trainer(tmp_path / "model.sqlite", cfg) as trainer:
        before = trainer.status()
        rng = json.dumps(trainer.rng.bit_generator.state, sort_keys=True)

        def fail(*args):
            trainer.rng.random()
            raise RuntimeError("interrupted")

        monkeypatch.setattr(trainer, "_walk", fail)
        with pytest.raises(RuntimeError):
            trainer.train(1)
        assert trainer.status() == before
        assert json.dumps(trainer.rng.bit_generator.state, sort_keys=True) == rng


def test_focus_roots_ignore_private_labels_and_reject_changed_dataset(tmp_path):
    cfg = Config(chip_bb=0.5, stacks_bb=(3,))
    public = {"hero_position": "bb", "board": "Kh8d3c", "effective_stack_bb": 3,
              "history": [[["btn", "call", 0], ["bb", "check", 0]]]}
    with Trainer(tmp_path / "a.sqlite", cfg, seed=7, focus=[{**public, "hero_hole": "AsAd"}]) as a:
        a.train(10)
    with Trainer(tmp_path / "b.sqlite", cfg, seed=7, focus=[{**public, "hero_hole": "2s2d"}]) as b:
        b.train(10)
    assert rows(tmp_path / "a.sqlite") == rows(tmp_path / "b.sqlite")
    assert any(json.loads(row[0]).get("root_id", "whole_game") != "whole_game"
               for row in rows(tmp_path / "a.sqlite"))
    with pytest.raises(ValueError, match="focus"):
        Trainer(tmp_path / "a.sqlite", focus=[{**public, "board": "Ks8d3c"}])


def test_policy_is_readonly_and_requires_coverage_and_promotion(tmp_path):
    path, frozen = tmp_path / "model.sqlite", tmp_path / "frozen.sqlite"
    cfg = Config(chip_bb=0.5, stacks_bb=(3,))
    with Trainer(path, cfg, seed=3) as trainer:
        trainer.train(30)
        trainer.checkpoint(frozen)
    before = frozen.read_bytes()
    with ExactPolicy(frozen, min_updates=1) as policy:
        assert policy.lookup({})["status"] == "unpromoted"
    with ExactPolicy(frozen, min_updates=1, candidate=True) as policy:
        with sqlite3.connect(frozen) as db:
            key = db.execute("SELECT key FROM nodes WHERE street=0 AND updates>0 LIMIT 1").fetchone()[0]
        info = json.loads(key)
        context = {"hero_position": ("btn", "bb")[info["actor"]], "hero_hole": "".join(info["hole"]),
                   "board": "".join(info["board"]), "effective_stack_bb": info["stack"] * cfg.chip_bb,
                   "history": [[[("btn", "bb")[p], k, to * cfg.chip_bb] for p, k, to in h]
                               for h in info["history"]]}
        answer = policy.lookup(context)
        assert answer["status"] == "ready"
        assert sum(answer["probs"].values()) == pytest.approx(1)
        assert set(answer["probs"]) == set(answer["options"])
        assert policy.lookup({**context, "effective_stack_bb": 100})["status"] == "unseen"
    assert before == frozen.read_bytes()


def test_promotion_rejects_sparse_small_or_losing_evaluations():
    good = {"deals": 2000, "coverage": 0.5, "bb100": 10, "se_bb100": 2, "illegal": 0}
    assert promotion_errors(good) == []
    for bad in ({"deals": 10}, {"coverage": 0.01}, {"bb100": -20}, {"illegal": 1}):
        assert promotion_errors({**good, **bad})


def test_promoted_reader_honors_the_evaluated_visit_threshold(tmp_path):
    source, frozen = tmp_path / "model.sqlite", tmp_path / "snapshot.sqlite"
    with Trainer(source, Config(chip_bb=0.5, stacks_bb=(3,))) as trainer:
        trainer.train(1)
        trainer.checkpoint(frozen)
    report = {"deals": 2000, "coverage": 0.5, "bb100": 10, "se_bb100": 2, "illegal": 0,
              "min_updates": 5000}
    with sqlite3.connect(frozen) as db:
        db.execute("INSERT INTO metadata VALUES ('promotion',?)", (json.dumps(report),))
    with ExactPolicy(frozen, min_updates=1000) as policy:
        assert policy.minimum == 5000


def test_resume_rejects_changed_dependency_runtime(tmp_path, monkeypatch):
    import preflop.exacttrain as training

    path = tmp_path / "runtime.sqlite"
    with Trainer(path, Config(chip_bb=0.5, stacks_bb=(3,))):
        pass
    monkeypatch.setattr(training, "_runtime", lambda: {"python": "different"})
    with pytest.raises(ValueError, match="runtime"):
        Trainer(path)
