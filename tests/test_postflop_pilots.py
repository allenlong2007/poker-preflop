import json
import math

import numpy as np
import pytest

from bot.engine import Act, Engine, State, parse_cards


@pytest.mark.parametrize("stack, hi_weight", [(30, 0), (40, 0), (60, math.log(1.5) / math.log(2.5)), (100, 1)])
def test_depth_model_blends_postflop_rows(tmp_path, stack, hi_weight):
    engine = Engine(policy_dir=tmp_path / "policy_fine", exact_required=False, use_solver_flops=False,
                    postflop_depths=[40, 100])
    sizes = {"donk": True, "sizes": [{"lead": [1 / 3], "bet": [1 / 3]}] * 3}
    shape = (100, 1, 12, 2)
    engine._pots["srp_alt"] = {"pot_to": 3}
    engine._pots["srp"] = {
        "opts_json": json.dumps(sizes),
        "sigma_0_bb_first": np.broadcast_to([0.75, 0.25], shape),
    }
    folder = tmp_path / "policy_depth" / "40"
    folder.mkdir(parents=True)
    np.savez(folder / "srp.npz", opts_json=json.dumps(sizes),
             sigma_0_bb_first=np.broadcast_to([0.25, 0.75], shape))
    state = State([[Act("btn", "raise", 2.25), Act("bb", "call")]], stack)
    decision = engine._postflop("bb", parse_cards("AcQd"), parse_cards("Kh8d3c"), state)
    assert decision.diagnostics["base"]["probs"]["check"] == pytest.approx(0.25 + 0.5 * hi_weight)
    assert sum(decision.diagnostics["lookup"]["postflop_depths"].values()) == pytest.approx(1)


@pytest.fixture
def wide_engine(tmp_path):
    from scripts.open_size_study import OPTS_WIDE
    from preflop.fullgame import table_shapes

    menu = tmp_path / "srp.npz"
    arrays = {f"sigma_{s}_{kind}": np.full(shape, 1 / shape[-1])
              for (s, kind), shape in table_shapes(OPTS_WIDE).items()}
    arrays["sigma_0_btn_vs_xr"][..., 0, :] = [0.2, 0.8]
    arrays["sigma_0_btn_vs_xr"][..., 1, :] = [0.8, 0.2]
    np.savez(menu, opts_json=json.dumps(OPTS_WIDE), **arrays)
    return Engine(exact_required=False, use_solver_flops=False, policy_overrides={"srp": menu})


def test_wide_menu_assigns_distinct_raise_totals(wide_engine):
    state = State([[Act("btn", "raise", 2.25), Act("bb", "call")],
                   [Act("bb", "check"), Act("btn", "bet", 1.5)]])
    wide_engine._postflop("bb", parse_cards("AcQd"), parse_cards("Kh8d3c"), state)
    assert wide_engine._options["raise 0.5"] == ("raise", 5.25)
    assert wide_engine._options["raise 1"] == ("raise", 9.0)


@pytest.mark.parametrize("total, fold", [(5.25, 0.2), (7.125, 0.5428571428571429), (9.0, 0.8)])
def test_wide_menu_translates_faced_raise(wide_engine, total, fold):
    state = State([[Act("btn", "raise", 2.25), Act("bb", "call")],
                   [Act("bb", "check"), Act("btn", "bet", 1.5), Act("bb", "raise", total)]])
    decision = wide_engine._postflop("btn", parse_cards("AcQd"), parse_cards("Kh8d3c"), state)
    assert decision.diagnostics["base"]["probs"]["fold"] == pytest.approx(fold)


def test_wide_policy_preserves_v7_solver_labels(wide_engine, monkeypatch):
    from scripts.open_size_study import OPTS_FINE

    wide_engine.use_solver_flops = True
    wide_engine._solver_opts["srp"] = OPTS_FINE
    wide_engine.solved_open = 2.25

    def solver(hole, board, kind, labels, cell, bets):
        assert labels == ["check", "bet 0.33", "bet 0.67", "bet 1"]
        assert bets == (1 / 3, 2 / 3, 1.0)
        return dict.fromkeys(labels, 0.25), "fixture"

    monkeypatch.setattr(wide_engine, "_solver_flop", solver)
    state = State([[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check")]])
    decision = wide_engine._postflop("btn", parse_cards("AcQd"), parse_cards("Kh8d3c"), state)
    assert decision.source == "solver"
    assert list(wide_engine._options) == ["check", "bet 0.33", "bet 0.67", "bet 1"]


def test_training_snapshot_uses_same_run(monkeypatch):
    from preflop import fullgame as fg

    key = (0, "btn_vs_check")
    monkeypatch.setattr(fg, "table_shapes", lambda opts: {key: (2,)})
    monkeypatch.setattr(fg.os, "cpu_count", lambda: 1)

    def batch(fn, jobs):
        weight = jobs[0][6]
        return [({key: np.zeros(2)}, {key: np.array([1, 0]) if weight <= 2 else np.array([0, 1])})]

    monkeypatch.setattr(fg, "_pool_map", batch)
    snapshots = []
    final, _ = fg.learn(np.ones(169), np.ones(169), {}, batches=4, verbose=False,
                        midpoint_snapshot=lambda batch, sigma: snapshots.append((batch, sigma[key])))
    assert len(snapshots) == 1 and snapshots[0][0] == 2
    np.testing.assert_array_equal(snapshots[0][1], [1, 0])
    np.testing.assert_array_equal(final[key], [0.5, 0.5])


def test_wide_menu_has_unique_labels():
    from scripts.open_size_study import OPTS_WIDE

    for street in OPTS_WIDE["sizes"]:
        for sizes in street.values():
            assert len({f"{f:.2g}" for f in sizes}) == len(sizes)
    assert len({f"{f:.2g}" for f in OPTS_WIDE["raise_pot"]}) == len(OPTS_WIDE["raise_pot"])


def test_nearest_depth_mode_plays_one_model():
    from bot.engine import Engine
    near = Engine(exact_required=False, postflop_depths=(25, 40, 60, 100), postflop_nearest=True)
    assert [d for d, _, w in near._postflop_models("srp", 70)] == [60]
    assert [d for d, _, w in near._postflop_models("srp", 80)] == [100]
    blend = Engine(exact_required=False, postflop_depths=(25, 40, 60, 100))
    assert [d for d, _, w in blend._postflop_models("srp", 70)] == [60, 100]
