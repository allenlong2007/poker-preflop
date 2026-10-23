"""Exact policy routing and after-reveal evidence never leak into past decisions."""
import json
import sqlite3
from pathlib import Path

import pytest

from bot.engine import Engine
from preflop.exactgame import Config
from preflop.exacttrain import Trainer


def test_engine_routes_covered_candidate_without_bucket_overrides(tmp_path):
    path, frozen = tmp_path / "train.sqlite", tmp_path / "snapshot.sqlite"
    config = Config(chip_bb=0.5, stacks_bb=(3,))
    with Trainer(path, config, seed=21) as trainer:
        trainer.train(10)
        trainer.checkpoint(frozen)
    with sqlite3.connect(frozen) as db:
        key = db.execute("SELECT key FROM nodes WHERE street=0 AND updates>0 LIMIT 1").fetchone()[0]
    obs = json.loads(key)
    history = [[[("btn", "bb")[p], k, to * config.chip_bb] for p, k, to in h] for h in obs["history"]]
    engine = Engine(seed=1, exact_policy_path=frozen, exact_candidate=True, exact_min_updates=1)
    d = engine.decide(("btn", "bb")[obs["actor"]], "".join(obs["hole"]), "", history, stack=3)
    assert d.source == "exact"
    assert d.diagnostics["exact"]["status"] == "ready"
    assert d.diagnostics["adjustments"] == []
    engine.exact_policy.close()
    guarded = Engine(seed=1, exact_policy_path=frozen)
    d = guarded.decide("btn", "AsKh", "", [[]], stack=3)
    assert d.source != "exact" and d.diagnostics["exact"]["status"] == "unpromoted"
    guarded.exact_policy.close()


def test_revealed_analysis_keeps_exact_cards_and_only_street_visible_board():
    from bot.exact_opponent import analyze_revealed

    record = {"id": "h1", "hero_id": "hero", "opp_id": "opp", "opp_pos": "bb", "hero_pos": "btn",
              "stacks": {"hero": 100, "opp": 100}, "shown": {"opp": "AsQs"}, "board": "Ks8s3d2cJh",
              "history": [[["btn", "raise", 2.25], ["bb", "call", 0]],
                          [["bb", "check", 0], ["btn", "bet", 1.5], ["bb", "call", 0]],
                          [["bb", "check", 0], ["btn", "check", 0]],
                          [["bb", "bet", 4], ["btn", "fold", 0]]]}
    evidence = analyze_revealed(record)
    assert evidence["opponent_hole"] == ["As", "Qs"]
    assert evidence["selection"] == "revealed_hands_only"
    flop = next(a for a in evidence["actions"] if a["street"] == 1)
    assert flop["board"] == ["Ks", "8s", "3d"]
    assert "Jh" not in json.dumps(flop)
    assert flop["features"]["flush_draw_suits"] == ["s"]
    assert flop["features"]["evaluation"] is not None
    assert evidence["actions"][-1]["action"] == "bet"
    assert analyze_revealed({**record, "shown": {}}) is None


def test_reveal_index_is_idempotent_and_reports_exact_combo(tmp_path):
    from bot.exact_opponent import EvidenceIndex

    record = {"id": "h1", "hero_id": "hero", "opp_id": "opp", "opp_pos": "btn", "hero_pos": "bb",
              "stacks": {"hero": 100, "opp": 100}, "shown": {"opp": "AsQs"}, "board": "",
              "history": [[["btn", "fold", 0]]]}
    with EvidenceIndex(tmp_path / "evidence.sqlite") as index:
        index.add(record)
        index.add(record)
        assert index.summary("opp")["revealed_hands"] == 1
        assert index.summary("opp")["exact_combos"] == {"AsQs": 1}


def test_raw_parser_attaches_revealed_analysis():
    from bot.opponent import record, split_hands
    from bot.pokernow.parse import read_log

    demo = Path(__file__).resolve().parents[1] / "bot" / "pokernow" / "demo_log.csv"
    hand = split_hands(read_log(demo.read_text()))[-1]
    hand.insert(-1, '"Friend @ fR7" shows a A♠, Q♠.')
    result = record(hand, hero_name="Allen")
    assert result["revealed_analysis"]["opponent_hole"] == ["As", "Qs"]
