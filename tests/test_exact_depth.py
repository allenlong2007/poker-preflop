"""preflop/exactdepth.py: exact preflop nodes with legacy-engine rollouts from the flop on."""
import json

import pytest

from preflop import exactdepth
from preflop.exactdepth import DepthLimitedTrainer

TARGET = {"hero_position": "btn", "hero_hole": "AsQd", "board": "", "effective_stack_bb": 20,
          "history": [[], [], [], []]}
FLOP = {**TARGET, "board": "Kh8d3c", "history": [[["btn", "raise", 2.5], ["bb", "call", 0.0]], [], [], []]}


def _trainer(path, **kwargs):
    from scripts.prepare_coverage import menu_config
    return DepthLimitedTrainer(path, menu_config([TARGET], chip_bb=.5), targets=[TARGET], key_text_floor=1, seed=3,
                               **kwargs)


def test_stops_at_the_flop_and_records_the_continuation(tmp_path):
    with _trainer(tmp_path / "d.sqlite") as trainer:
        trainer.train(30)
        streets = {row[6] for row in trainer.store.dump()}
        assert streets == {0}                                       # no postflop node is ever stored
        assert trainer.aggregates["leaf_rollouts"] > 0
        record = json.loads(trainer.db.execute("SELECT value FROM metadata WHERE key='continuation'").fetchone()[0])
        assert record["kind"] == "legacy_rollout" and record["leaf_street"] == 1


def test_training_is_reproducible(tmp_path):
    dumps = []
    for name in ("a", "b"):
        with _trainer(tmp_path / f"{name}.sqlite") as trainer:
            trainer.train(20)
            dumps.append(trainer.store.dump())
    assert dumps[0] == dumps[1]


def test_resume_guards(tmp_path, monkeypatch):
    path = tmp_path / "d.sqlite"
    with _trainer(path) as trainer:
        trainer.train(5)
    from preflop.exactcoverage import CoverageTrainer
    with pytest.raises(ValueError, match="depth-limited candidate"):
        CoverageTrainer(path)                                         # never resumed as a full-game walk
    original = exactdepth.continuation_fingerprint
    monkeypatch.setattr(exactdepth, "continuation_fingerprint", lambda e: {**original(e), "policy_sha256": "x"})
    with pytest.raises(ValueError, match="continuation changed"):
        _trainer(path)


def test_rejects_postflop_targets(tmp_path):
    from scripts.prepare_coverage import menu_config
    with pytest.raises(ValueError, match="preflop targets only"):
        DepthLimitedTrainer(tmp_path / "d.sqlite", menu_config([FLOP], chip_bb=.5), targets=[FLOP], key_text_floor=1)


def test_potodds_speedups_keep_the_same_results():
    import eval7
    import numpy as np
    from bot import potodds
    from preflop import fullgame as fg
    deck = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
    g = np.random.default_rng(1)
    for trial in range(20):
        idx = g.permutation(52)
        hole, board = [deck[i] for i in idx[:2]], [deck[i] for i in idx[2:6]]          # turn: exact enumeration
        rng = potodds.combos(g.random(169) ** 3, set(hole) | set(board))
        win = tot = 0.0
        rest = [c for c in deck if c not in set(hole) | set(board)]
        for c1, c2, w in rng:
            for r in rest:
                if r == c1 or r == c2:
                    continue
                m, v = eval7.evaluate(hole + board + [r]), eval7.evaluate([c1, c2] + board + [r])
                win += w * (1.0 if m > v else 0.5 if m == v else 0.0)
                tot += w
        assert potodds.equity(hole, board, rng) == win / tot
        assert all(potodds._draw(c1, c2, tuple(board[:3])) == fg._has_draw((c1, c2), board[:3]) for c1, c2, _ in rng)


def test_external_sampling_walks_every_updater_action_and_records_it(tmp_path):
    with _trainer(tmp_path / "e.sqlite", sampling="external", leaf_runouts=3) as trainer:
        trainer.train(10)
        recorded = json.loads(trainer.db.execute("SELECT value FROM metadata WHERE key='depth_sampling'").fetchone()[0])
        assert recorded == {"kind": "external", "leaf_runouts": 3}
        assert trainer.aggregates["leaf_rollouts"] % 3 == 0
        root = trainer.store.get(trainer.targets[0][0])
        assert 0 < root[4] <= 10                                     # at most one update per traversal
    with pytest.raises(ValueError, match="sampling differs"):
        _trainer(tmp_path / "e.sqlite", sampling="outcome")
    with _trainer(tmp_path / "e.sqlite") as trainer:                 # a resume keeps the recorded sampling
        assert trainer.sampling["kind"] == "external"


def test_external_sampling_is_reproducible(tmp_path):
    dumps = []
    for name in ("a", "b"):
        with _trainer(tmp_path / f"{name}.sqlite", sampling="external", leaf_runouts=2) as trainer:
            trainer.train(5)
            dumps.append(trainer.store.dump())
    assert dumps[0] == dumps[1]


def test_old_candidates_default_to_outcome_sampling(tmp_path):
    path = tmp_path / "o.sqlite"
    with _trainer(path) as trainer:
        trainer.train(3)
        assert trainer.sampling == {"kind": "outcome", "leaf_runouts": 1}
    with pytest.raises(ValueError, match="sampling differs"):
        _trainer(path, sampling="external")
