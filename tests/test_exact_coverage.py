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
    root = {**context(stack=3), "board": "", "history": [[]]}
    data.write_text(json.dumps({"context": root}) + "\n")
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


def test_update_count_without_average_mass_is_not_covered(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    cfg = Config(chip_bb=.5, stacks_bb=(3,))
    with CoverageTrainer(tmp_path / "zero.sqlite", cfg, targets=[context(stack=3)]) as trainer:
        trainer.train(20)
        key = trainer.targets[0][0]
        actions = json.loads(trainer.db.execute("SELECT actions FROM nodes WHERE key=?", (key,)).fetchone()[0])
        trainer.db.execute("UPDATE nodes SET updates=5000,total=? WHERE key=?", (json.dumps([0] * len(actions)), key))
        trainer.db.commit()
        assert trainer.coverage(1000)["trained_targets"] == 0


def test_coverage_resume_rejects_changed_target_table(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    cfg = Config(chip_bb=.5, stacks_bb=(3,))
    path = tmp_path / "tampered.sqlite"
    with CoverageTrainer(path, cfg, targets=[context(stack=3)]) as trainer:
        trainer.db.execute("UPDATE coverage_targets SET context=replace(context,'AsQd','AcQd')")
        trainer.db.commit()
    with pytest.raises(ValueError, match="manifest"):
        CoverageTrainer(path)


def test_exact_only_engine_does_not_need_any_legacy_policy_files(tmp_path, monkeypatch):
    from bot import engine as module
    from bot.coverage import ExactCoverageError
    monkeypatch.setattr(module, "POLICY", tmp_path / "no-legacy-policies")
    e = Engine(exact_policy_path=False)
    e.warm()
    with pytest.raises(ExactCoverageError):
        e.decide("btn", "AsQd", "", [[]], stack=10)


def test_default_strict_engine_really_persists_its_queue():
    from bot.coverage import CoverageStore, ExactCoverageError
    engine = Engine(exact_policy_path=False)
    with pytest.raises(ExactCoverageError):
        engine.decide("btn", "AsQd", "", [[]], stack=10)
    assert engine.coverage_path is not None
    with CoverageStore(engine.coverage_path) as store:
        assert store.summary()["queued_exact_cells"] == 1


def test_exact_turn_river_and_raise_policies_bypass_every_legacy_path(tmp_path):
    from preflop.exactcoverage import CoverageTrainer
    from bot.coverage import ExactCoverageError
    river = context(stack=100)
    river["history"][-1] += [["btn", "bet", 2], ["bb", "raise", 6], ["btn", "raise", 14], ["bb", "raise", 30]]
    turn = {**context(stack=100), "board": "Kh8d3c2s", "history": context()["history"][:2] + [[["bb", "check", 0]] , []]}
    cfg = Config(chip_bb=.5, stacks_bb=(100,), extra_sizes_bb=(2, 6, 14, 30))
    path = tmp_path / "candidate.sqlite"
    with CoverageTrainer(path, cfg, targets=[turn, river], seed=22) as trainer:
        trainer.train(100)
        # Routing fixture: positive exact averages are supplied explicitly.
        # This test makes no convergence claim for an unreachable historical line.
        for key, _ in trainer.targets:
            actions = trainer.target_actions[key]
            trainer.db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
                               (key, json.dumps(actions), json.dumps([0] * len(actions)),
                                json.dumps([1] * len(actions)), 10, 10, json.loads(key)["street"]))
        trainer.db.commit()
        trainer.checkpoint(tmp_path / "frozen.sqlite")
    e = Engine(exact_policy_path=tmp_path / "frozen.sqlite", exact_candidate=True, exact_min_updates=1)
    with patch.object(e, "_postflop", side_effect=AssertionError("legacy forbidden")):
        for c in (turn, river):
            d = e.decide(c["hero_position"], c["hero_hole"], c["board"], c["history"], stack=100)
            assert d.source == "exact" and d.diagnostics["fallbacks"] == []
            assert d.diagnostics["adjustments"] == []
        with pytest.raises(ExactCoverageError):
            e.decide("btn", "AcQd", river["board"], river["history"], stack=100)
    e.exact_policy.close()


def test_resume_cannot_ignore_manifest_precision_or_raise_menu(tmp_path):
    import hashlib
    from scripts.prepare_coverage import prepare
    from scripts.train_exact import main
    data, plan, path = tmp_path / "log.jsonl", tmp_path / "plan.json", tmp_path / "run.sqlite"
    data.write_text(json.dumps({"context": {**context(stack=3), "board": "", "history": [[]]}}) + "\n")
    manifest = prepare(tmp_path / "queue.sqlite", [data], [], chip_bb=.5)
    plan.write_text(json.dumps(manifest))
    command = ["train", "--path", str(path), "--coverage-targets", str(plan), "--iterations", "1"]
    main(command)
    manifest["config"]["fractions"] = [.5, 1]
    payload = {k: manifest[k] for k in ("config", "targets")}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    plan.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration"):
        main(command)


def test_legacy_nearest_size_and_group_average_are_counted_explicitly(tmp_path, monkeypatch):
    from bot import engine as module
    from bot.coverage import CoverageStore
    e = Engine(exact_required=False, exact_policy_path=False, coverage_path=tmp_path / "counts.sqlite")
    d = e.decide("bb", "AsQd", "", [[["btn", "raise", 2.37]]])
    assert "legacy_preflop_nearest_size" in d.diagnostics["fallbacks"]
    monkeypatch.setattr(module, "map_combo", lambda *args: None)
    e.use_solver_flops, e.solved_open = True, 3
    e.pre["open"] = 3
    nodes = {"btn_vs_check": {(frozenset(("As", "Qd")), None): {"check": .5, "bet 2bb": .5}}}
    monkeypatch.setattr(e, "_flop", lambda t: ("Kh8d3c", nodes))
    d = e.decide("btn", "AsQd", "Kh8d3c", [[["btn", "raise", 3], ["bb", "call", 0]], [["bb", "check", 0]]])
    assert "legacy_flop_hand_group_average" in d.diagnostics["fallbacks"]
    with CoverageStore(tmp_path / "counts.sqlite") as store:
        assert store.summary()["fallbacks"]["legacy_flop_hand_group_average"]["fraction_of_attempts"] == .5


def test_private_expansion_enumerates_every_literal_combo_for_requested_board(tmp_path):
    from math import comb
    from scripts.prepare_coverage import prepare
    data = tmp_path / "one.jsonl"
    data.write_text(json.dumps({"context": context(stack=3)}) + "\n")
    plan = prepare(tmp_path / "queue.sqlite", [data], [], chip_bb=.5, expand_private=True)
    assert plan["target_count"] == comb(47, 2)
    assert len({c["hero_hole"] for c in plan["targets"]}) == comb(47, 2)
    assert plan["full_game_complete"] is False


def test_prepare_retains_unsupported_requests_and_cli_reports_only_real_attempts(tmp_path, capsys, monkeypatch):
    import sys
    from scripts.prepare_coverage import main
    from bot.coverage import CoverageStore
    data, plan, queue = tmp_path / "log.jsonl", tmp_path / "plan.json", tmp_path / "queue.sqlite"
    good = {**context(stack=10), "board": "", "history": [[]]}
    bad = {**good, "hero_position": "bb", "history": [[["btn", "raise", 20]]]}
    data.write_text("not json\n{}\n" + json.dumps({"context": good}) + "\n" + json.dumps({"context": bad}) + "\n")
    monkeypatch.setattr(sys, "argv", ["prepare", "--queue", str(queue), "--decisions", str(data),
                                     "--hands", str(tmp_path / "empty"), "--output", str(plan)])
    main()
    result = json.loads(plan.read_text())
    assert result["target_count"] == 1
    assert result["rejected_or_missing"]["invalid_archive_record"] == 1
    with CoverageStore(queue) as store:
        assert store.summary()["attempts"] == 0
        assert len(store.targets()) == 2
        assert any(row["reason"].startswith("unsupported:") for row in store.targets())
    from bot.coverage import main as coverage_main
    monkeypatch.setattr(sys, "argv", ["coverage", "--path", str(queue), "--export", str(tmp_path / "export.json")])
    coverage_main()
    assert len(json.loads((tmp_path / "export.json").read_text())) == 2
