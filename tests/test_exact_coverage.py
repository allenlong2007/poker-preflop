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
