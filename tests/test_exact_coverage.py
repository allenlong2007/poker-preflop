"""Exact-only serving, durable gap accounting, and targeted CPU training."""
import json
from unittest.mock import patch

import pytest

from bot.engine import Engine
from preflop.exactgame import Config


def context(hole="AsQd", board="Kh8d3c2s9h", stack=10):
    return {"hero_position": "btn", "hero_hole": hole, "board": board,
            "effective_stack_bb": stack,
            "history": [[["btn", "call", 0], ["bb", "check", 0]],
                        [["bb", "check", 0], ["btn", "check", 0]],
                        [["bb", "check", 0], ["btn", "check", 0]], [["bb", "check", 0]]]}


def test_default_engine_withholds_and_queues_instead_of_using_buckets(tmp_path):
    from bot.coverage import CoverageStore, ExactCoverageError
    engine = Engine(exact_policy_path=False, coverage_path=tmp_path / "coverage.sqlite")
    c = context()
    with patch.object(engine, "_postflop", side_effect=AssertionError("bucket lookup forbidden")):
        with pytest.raises(ExactCoverageError) as raised:
            engine.decide(c["hero_position"], c["hero_hole"], c["board"], c["history"], stack=10)
    assert raised.value.diagnostics["exact"]["status"] == "disabled"
    with CoverageStore(tmp_path / "coverage.sqlite") as store:
        report = store.summary()
        assert report["attempts"] == report["withheld"] == 1
        assert report["exact"] == 0
        queued = store.targets()
    assert queued[0]["context"]["hero_hole"] == "QdAs" or queued[0]["context"]["hero_hole"] == "AsQd"
    assert queued[0]["context"]["board"][-4:] == "2s9h"


def test_queue_never_merges_blockers_kickers_runouts_or_raise_amounts(tmp_path):
    from bot.coverage import CoverageStore
    base = context()
    variants = [base, context("AcQd"), context("AsJd"), context(board="Kh8d3c2s9d"), context(stack=20)]
    changed = json.loads(json.dumps(base))
    changed["history"][-1] += [["btn", "bet", 2], ["bb", "raise", 6]]
    variants.append(changed)
    with CoverageStore(tmp_path / "coverage.sqlite") as store:
        for c in variants + [base]:
            store.record(c, "unseen", "withheld", ["exact:unseen"])
        assert len(store.targets()) == 6
        assert store.summary()["attempts"] == 7
        assert store.summary()["fallbacks"]["exact:unseen"]["count"] == 7


def test_all_postflop_streets_and_repeated_raises_have_exact_keys():
    from preflop.exactgame import from_context
    import numpy as np
    c = context(stack=100)
    c["history"][-1] += [["btn", "bet", 2], ["bb", "raise", 6], ["btn", "raise", 14], ["bb", "raise", 30]]
    s = from_context(Config(), c, np.random.default_rng(1), known_hole=True)
    assert s.actor == 0 and s.street == 3 and len(s.history[3]) == 5
    assert any(a.startswith("to:") for a in s.actions())
    assert s.key() != from_context(Config(), {**c, "hero_hole": "AcQd"}, np.random.default_rng(1), known_hole=True).key()


def test_server_returns_no_action_and_counts_cached_gap_once(tmp_path, monkeypatch):
    from bot.pokernow import server
    from bot.coverage import CoverageStore
    monkeypatch.setattr(server, "LOG_DIR", tmp_path)
    monkeypatch.setattr(server.ENGINE, "exact_required", True)
    monkeypatch.setattr(server.ENGINE, "exact_policy", None)
    monkeypatch.setattr(server, "CACHE", {})
    log = ['-- starting hand #9 (id: exactmissing)  No Limit Texas Hold\'em (dealer: "A @ a1") --',
           'Player stacks: #1 "A @ a1" (20.00) | #2 "B @ b2" (3.00)',
           "Your hand is A♠, A♥", '"A @ a1" posts a small blind of 0.10', '"B @ b2" posts a big blind of 0.20',
           '"A @ a1" raises to 0.60']
    for _ in range(2):
        result = server.decide({"log": json.dumps(log), "seat": 2})
        assert result["decision"] is None and "Exact" in result["wait"]
    with CoverageStore(tmp_path / "coverage.sqlite") as store:
        assert store.summary()["attempts"] == 1
        assert store.summary()["queued_exact_cells"] == 1


def test_coverage_training_updates_exact_gap_and_resumes(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    from preflop.exacttrain import ExactPolicy
    c = context(stack=3)
    cfg = Config(chip_bb=.5, stacks_bb=(3,), fractions=(.5, 1))
    path = tmp_path / "targeted.sqlite"
    with CoverageTrainer(path, cfg, targets=[c], seed=22) as trainer:
        trainer.train(20)
        report = trainer.coverage(minimum=5)
        assert report["targets"] == report["trained_targets"] == 1
        assert report["by_street"]["3"]["trained"] == 1
        assert report["full_game_complete"] is False
        trainer.checkpoint(tmp_path / "frozen.sqlite")
    with CoverageTrainer(path) as trainer:
        trainer.train(2)
        assert trainer.status()["iterations"] == 22
    with ExactPolicy(tmp_path / "frozen.sqlite", min_updates=5, candidate=True) as policy:
        assert policy.lookup(c)["status"] == "ready"
    with ExactPolicy(tmp_path / "frozen.sqlite", min_updates=5) as policy:
        assert policy.lookup(c)["status"] == "unpromoted"


def test_targeted_chance_mixture_preserves_uniform_prior():
    from preflop.exactcoverage import chance_weight
    n = 1081
    q_target, q_other = .8 + .2 / n, .2 / n
    assert q_target * chance_weight(True, n) == pytest.approx(1 / n)
    assert q_other * chance_weight(False, n) == pytest.approx(1 / n)
    assert q_target * chance_weight(True, n) + (n - 1) * q_other * chance_weight(False, n) == pytest.approx(1)


def test_extra_observed_raise_totals_are_legal_without_rounding_to_menu():
    cfg = Config(chip_bb=.01, stacks_bb=(100,), extra_sizes_bb=(2.37, 8.19))
    from preflop.exactgame import State
    import numpy as np
    s = State.deal(cfg, np.random.default_rng(1), stack_bb=100)
    assert "to:237" in s.actions() and "to:819" in s.actions()


def test_prepare_plan_and_bounded_cli_train_are_reproducible(tmp_path, capsys):
    from scripts.prepare_coverage import prepare
    from scripts.train_exact import main
    queue, data, plan = tmp_path / "queue.sqlite", tmp_path / "decisions.jsonl", tmp_path / "plan.json"
    data.write_text(json.dumps({"context": context(stack=3)}) + "\n")
    result = prepare(queue, [data], [], chip_bb=.5)
    assert len(result["targets"]) == 1
    assert result["full_game_complete"] is False
    plan.write_text(json.dumps(result))
    candidate = tmp_path / "coverage.sqlite"
    main(["train", "--path", str(candidate), "--coverage-targets", str(plan), "--iterations", "30",
          "--batch", "5", "--target-min-updates", "2", "--max-gb", ".1"])
    status = json.loads(candidate.with_suffix(".status.json").read_text())
    assert status["coverage"]["trained_targets"] == 1
    assert status["stop_reason"] == "queued_target_floor_reached"
    assert status["iterations"] <= 30
    result2 = prepare(queue, [data], [], chip_bb=.5)
    assert result["manifest_sha256"] == result2["manifest_sha256"]


def test_archived_targets_use_only_action_time_cards_and_no_revealed_opponent(tmp_path):
    from scripts.prepare_coverage import hand_contexts
    record = {"hero_pos": "btn", "hero_hole": "AsQd", "shown": {"opp": "AcAd"},
              "board": "Kh8d3c2s9h", "stacks": {"hero": 10, "opp": 20},
              "history": context()["history"]}
    record["history"][1].append(["btn", "bet", 1])
    targets = list(hand_contexts(record))
    flop = next(c for c in targets if len(c["board"]) == 6)
    assert flop["board"] == "Kh8d3c"
    assert "9h" not in json.dumps(flop) and "AcAd" not in json.dumps(flop)
