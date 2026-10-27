"""Node-specific menus and the per-action / stability readiness gate (2026-10-26 live-context diagnosis)."""
import json

import numpy as np
import pytest

from preflop.exactgame import Config, from_context

RAISED = {"hero_position": "bb", "hero_hole": "AsQd", "board": "", "effective_stack_bb": 20,
          "history": [[["btn", "raise", 2.35]], [], [], []]}


def test_observed_sizes_join_only_their_own_node():
    from scripts.prepare_coverage import menu_config
    cfg = menu_config([RAISED], chip_bb=.05)
    assert cfg.fractions == pytest.approx((1 / 3, .75, 1.5)) and not cfg.extra_sizes_bb
    rng = np.random.default_rng(0)
    target = from_context(cfg, RAISED, rng, known_hole=True)            # reachable: 2.35 is on the BTN's menu
    assert len(target.actions()) <= 8
    opening = {**RAISED, "hero_position": "btn", "history": [[], [], [], []]}
    assert "to:47" in from_context(cfg, opening, rng, known_hole=True).actions()      # 2.35bb in 0.05bb chips
    deeper = Config(**{**_plain(cfg), "stacks_bb": (20, 30)})
    elsewhere = from_context(deeper, {**opening, "effective_stack_bb": 30}, rng, known_hole=True)
    assert "to:47" not in elsewhere.actions()                             # another depth is another node


def _plain(cfg):
    return json.loads(json.dumps({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}))


def test_config_round_trips_and_old_fingerprints_are_unchanged():
    from scripts.prepare_coverage import menu_config
    cfg = menu_config([RAISED], chip_bb=.05)
    assert Config(**_plain(cfg)).fingerprint() == cfg.fingerprint()      # JSON metadata reloads the same game
    # Configs without node sizes hash exactly as before the field existed (existing artifacts stay readable).
    import hashlib
    from dataclasses import asdict
    old = asdict(Config(chip_bb=.01))
    old.pop("extra_sizes_bb")
    old.pop("node_sizes_bb")
    assert Config(chip_bb=.01).fingerprint() == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()


def _trainer(tmp_path, readiness):
    from preflop.exactcoverage import CoverageTrainer
    from scripts.prepare_coverage import menu_config
    target = {"hero_position": "btn", "hero_hole": "AsQd", "board": "", "effective_stack_bb": 3,
              "history": [[], [], [], []]}
    cfg = menu_config([target], chip_bb=.5)
    return CoverageTrainer(tmp_path / "c.sqlite", cfg, targets=[target], readiness=readiness, key_text_floor=1)


def test_per_action_floor_and_stability_gate_readiness(tmp_path):
    with _trainer(tmp_path, {"per_action": 50, "stable_tv": 0.05}) as trainer:
        key = trainer.targets[0][0]
        n_actions = len(trainer.target_actions[key])
        trainer.train(20)
        assert trainer.coverage(2)["trained_targets"] == 0                # far below 50 updates per action
        cp = trainer.db.execute("SELECT updates,stability FROM target_checkpoints WHERE key=?", (key,)).fetchone()
        assert cp is not None and cp[0] >= 1
        actions, regrets, total, visits, n, street = trainer.store.get(key)
        trainer.store.put(key, actions, regrets, total, visits, 50 * n_actions, street)
        trainer.db.execute("UPDATE target_checkpoints SET stability=0.2 WHERE key=?", (key,))
        assert trainer.coverage(2)["trained_targets"] == 0                # enough updates, not stable
        trainer.db.execute("UPDATE target_checkpoints SET stability=0.01 WHERE key=?", (key,))
        assert trainer.coverage(2)["trained_targets"] == 1


def test_checkpoints_record_stability_at_each_doubling(tmp_path):
    with _trainer(tmp_path, {"per_action": 1, "stable_tv": 0.05}) as trainer:
        key = trainer.targets[0][0]
        seen = []
        for _ in range(30):
            trainer.train(5)
            seen.append(trainer.db.execute("SELECT updates,stability FROM target_checkpoints WHERE key=?",
                                           (key,)).fetchone())
        updates = [u for u, _ in seen]
        assert updates == sorted(updates) and updates[-1] >= 4 * updates[0]
        assert all(0 <= s <= 1 for _, s in seen if s is not None) and seen[-1][1] is not None


def test_readiness_rule_is_fixed_at_creation(tmp_path):
    with _trainer(tmp_path, {"per_action": 50, "stable_tv": 0.05}) as trainer:
        trainer.train(2)
    with pytest.raises(ValueError, match="readiness rule differs"):
        _trainer(tmp_path, {"per_action": 10, "stable_tv": 0.05})


def test_export_and_serving_apply_the_same_gate(tmp_path):
    from preflop.exactmigrate import export_ready
    from preflop.exacttrain import ExactPolicy
    with _trainer(tmp_path, {"per_action": 50, "stable_tv": 0.05}) as trainer:
        trainer.train(20)
        key, context = trainer.targets[0]
        actions, regrets, total, visits, n, street = trainer.store.get(key)
        trainer.store.put(key, actions, regrets, total, visits, 50 * len(actions), street)
        trainer.db.execute("UPDATE target_checkpoints SET stability=0.2 WHERE key=?", (key,))
        trainer.db.commit()
        trainer.checkpoint(tmp_path / "unstable.sqlite")
        trainer.db.execute("UPDATE target_checkpoints SET stability=0.01 WHERE key=?", (key,))
        trainer.db.commit()
        trainer.checkpoint(tmp_path / "stable.sqlite")
    assert export_ready(tmp_path / "unstable.sqlite", tmp_path / "u.sqlite", 2)["nodes"] == 0
    assert export_ready(tmp_path / "stable.sqlite", tmp_path / "s.sqlite", 2)["nodes"] == 1
    clean = context
    with ExactPolicy(tmp_path / "s.sqlite", min_updates=2, candidate=True) as policy:
        assert policy.lookup(clean)["status"] == "ready"
    with ExactPolicy(tmp_path / "unstable.sqlite", min_updates=2, candidate=True) as policy:
        assert policy.lookup(clean)["status"] != "ready"


def test_linear_averaging_weights_later_updates_more(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    from scripts.prepare_coverage import menu_config
    target = {"hero_position": "btn", "hero_hole": "AsQd", "board": "", "effective_stack_bb": 3,
              "history": [[], [], [], []]}
    cfg = menu_config([target], chip_bb=.5)
    runs = {}
    for mode in ("uniform", "linear"):
        with CoverageTrainer(tmp_path / f"{mode}.sqlite", cfg, targets=[target], key_text_floor=1, seed=5,
                             averaging=mode) as trainer:
            trainer.train(40)
            assert json.loads(trainer.db.execute("SELECT value FROM metadata WHERE key='averaging'").fetchone()[0]) \
                == mode
            runs[mode] = {k: trainer.store.get(k) for k, _, _ in trainer.store.key_items()}
    assert runs["uniform"].keys() == runs["linear"].keys()
    for key, (actions, regrets, total, visits, n, street) in runs["linear"].items():
        same = runs["uniform"][key]
        assert np.array_equal(regrets, same[1]) and (visits, n) == (same[3], same[4])   # the walk is unchanged
        assert total.sum() >= same[2].sum() - 1e-12                                     # weights t >= 1
    with pytest.raises(ValueError, match="weighting"):
        CoverageTrainer(tmp_path / "bad.sqlite", cfg, targets=[target], key_text_floor=1, averaging="square")
