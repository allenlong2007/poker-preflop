"""preflop/exactpreflop.py: whole preflop game on a stack grid with hand-class keys and a fitted leaf model."""
import json

import numpy as np
import pytest

from preflop import exactpreflop
from preflop.exactgame import State
from preflop.exactpreflop import (PreflopGameTrainer, canonical_key, grid_config, legacy_fingerprint, map_context)
from preflop.leafvalue import AGGRESSORS, class_name, features, hand_class


def _leaf(tmp_path, scale=0.0):
    width = features([0], [0], [0.5], [1.0]).shape[1]
    record = {"schema": 1, "kind": "legacy_leaf_regression", "class_equity": np.full((169, 169), 0.5).tolist(),
              "coefficients": {a: ([scale] + [0.0] * (width - 1)) for a in AGGRESSORS}, "spr_range": [0.1, 100.0],
              "legacy": legacy_fingerprint()}
    path = tmp_path / "leaf.json"
    path.write_text(json.dumps(record))
    return path


def _trainer(tmp_path, **kwargs):
    return PreflopGameTrainer(tmp_path / "g.sqlite", grid_config(stacks=(10.0,)), seed=5, leaf=_leaf(tmp_path),
                              key_text_floor=1, **kwargs)


def test_hand_classes_and_canonical_keys():
    assert class_name(hand_class(("As", "Ks"))) == "AKs" and class_name(hand_class(("Kd", "Ac"))) == "AKo"
    state = State.deal(grid_config(stacks=(10.0,)), np.random.default_rng(0), stack_bb=10, hole="AsKs")
    same = State.deal(grid_config(stacks=(10.0,)), np.random.default_rng(1), stack_bb=10, hole="AhKh")
    off = State.deal(grid_config(stacks=(10.0,)), np.random.default_rng(1), stack_bb=10, hole="AhKd")
    assert canonical_key(state) == canonical_key(same) != canonical_key(off)


def test_trains_preflop_only_with_class_keys_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(exactpreflop, "CHECKPOINT_FROM", 2)
    with _trainer(tmp_path) as trainer:
        trainer.train(400)
        rows = trainer.store.dump()
        assert {row[6] for row in rows} == {0}
        holes = {tuple(json.loads(key)["hole"]) for key, _, _ in trainer.store.key_items()}
        assert len(holes) <= 169 and all(len({c[1] for c in h}) <= 2 for h in holes)
        assert trainer.db.execute("SELECT COUNT(*) FROM target_checkpoints").fetchone()[0] > 0
        assert trainer.coverage()["scope"] == "preflop_game_grid"


def test_resume_guards(tmp_path, monkeypatch):
    with _trainer(tmp_path) as trainer:
        trainer.train(5)
    leaf = tmp_path / "leaf.json"
    record = json.loads(leaf.read_text())
    leaf.write_text(json.dumps({**record, "spr_range": [0.2, 100.0]}))
    with pytest.raises(ValueError, match="leaf model file changed"):
        PreflopGameTrainer(tmp_path / "g.sqlite")
    leaf.write_text(json.dumps(record))
    monkeypatch.setattr(exactpreflop, "legacy_fingerprint", lambda: {"source_sha256": "x"})
    with pytest.raises(ValueError, match="legacy engine changed"):
        PreflopGameTrainer(tmp_path / "g.sqlite")


def test_leaf_and_allin_values_come_from_the_model(tmp_path):
    with PreflopGameTrainer(tmp_path / "v.sqlite", grid_config(stacks=(10.0,)), leaf=_leaf(tmp_path, scale=0.25),
                            key_text_floor=1) as trainer:
        state = State.deal(trainer.config, np.random.default_rng(0), stack_bb=10)
        flop = state.step("call").step("check")                     # limped pot of 2bb, BTN value g * pot
        assert trainer._walk(flop, 0) == pytest.approx(0.5) and trainer._walk(flop, 1) == pytest.approx(-0.5)
        allin = state.step(f"to:{state.cap}").step("call")
        assert allin.terminal and trainer._walk(allin, 0) == pytest.approx(0.0)     # equity 0.5, even stacks


def test_live_contexts_map_to_the_grid():
    ctx = {"hero_position": "bb", "hero_hole": "AsKd", "board": "", "effective_stack_bb": 96.4,
           "history": [[["btn", "raise", 2.25]], [], [], []]}
    mapped, live = map_context(ctx, [10.0, 100.0])
    assert mapped["effective_stack_bb"] == 100.0 and live == 96.4
    shove = {**ctx, "history": [[["btn", "raise", 96.4]], [], [], []]}
    assert map_context(shove, [100.0])[0]["history"][0][0][2] == 100.0
    assert map_context({**ctx, "effective_stack_bb": 130.0}, [100.0])[0] is None
    assert map_context({**ctx, "effective_stack_bb": 110.0, "history": [[["btn", "raise", 105.0]], [], [], []]},
                       [100.0])[0] is None
    assert map_context({**ctx, "board": "AhKh2c"}, [100.0])[0] is None


def test_serving_maps_stack_and_returns_live_amounts(tmp_path):
    from preflop.exacttrain import ExactPolicy
    with _trainer(tmp_path) as trainer:
        trainer.train(20)
        root = State.deal(trainer.config, np.random.default_rng(0), stack_bb=10, hole="AhKh")
        key = canonical_key(root)
        actions = root.actions()
        trainer.store.put(key, actions, np.zeros(len(actions)), np.ones(len(actions)), 1, 300 * len(actions), 0)
        trainer.db.execute("INSERT OR REPLACE INTO target_checkpoints VALUES (?,?,?,?)",
                           (key, 64, json.dumps([1 / len(actions)] * len(actions)), 0.0))
        trainer.db.commit()
    with ExactPolicy(tmp_path / "g.sqlite", candidate=True) as policy:
        live = {"hero_position": "btn", "hero_hole": "AsKs", "board": "", "effective_stack_bb": 9.6,
                "history": [[], [], [], []]}
        result = policy.lookup(live)
        assert result["status"] == "ready" and result["grid_stack_bb"] == 10.0
        cap = f"to:{root.cap}"
        assert result["options"][cap] == ("raise", 9.6)
        assert policy.lookup({**live, "effective_stack_bb": 40.0})["status"] == "unsupported"
        assert policy.lookup({**live, "hero_hole": "AsKd"})["status"] in ("unseen", "insufficient")


def test_memoised_menus_match_the_game_rules(tmp_path):
    from preflop.exactpreflop import _GridState
    config = grid_config(stacks=(27.0,))
    rng = np.random.default_rng(3)
    for _ in range(200):
        state = _GridState.deal(config, rng)
        while not state.terminal and state.street == 0:
            assert state.actions() == State.actions(state)
            state = state.step(state.actions()[int(rng.integers(len(state.actions())))])
