"""Experiment 50: stage switches, held-out opponents and simulated reads for ablation matches."""
import json

import numpy as np

from bot.engine import Act, Engine
from bot.heldout import HELDOUT, BaseMirror, make
from bot.selfplay import DECK, play_hand

STAGES = BaseMirror.STAGES


def test_switched_off_stages_never_change_a_decision():
    hist = [[Act("btn", "raise", 2.25), Act("bb", "call")], [Act("bb", "check"), Act("btn", "bet", 3.0)]]
    eng = Engine(exact_required=False, seed=0, pot_odds=True)
    eng.off = set(STAGES)
    d = eng.decide("bb", "AsAd", "Ah7c2d", hist, sample=False)
    assert not d.diagnostics.get("adjustments") or \
        {a["stage"] for a in d.diagnostics["adjustments"]} <= {"exploit_and_legality"}
    on = Engine(exact_required=False, seed=0, pot_odds=True).decide("bb", "AsAd", "Ah7c2d", hist, sample=False)
    assert {a["stage"] for a in on.diagnostics["adjustments"]} - {"exploit_and_legality"}   # some stage acts here


def test_held_out_bots_are_deterministic_and_legal():
    rng = np.random.default_rng(0)
    deals = [[DECK[i] for i in rng.choice(52, size=9, replace=False)] for _ in range(6)]
    for name in HELDOUT:
        runs = []
        for _ in range(2):
            a, b = make(name), make("geo")
            for stack in (100.0, 20.0):
                a.stack = b.stack = stack
                runs.append([play_hand({"btn": a, "bb": b}, {"btn": c[:2], "bb": c[2:4]}, c[4:], stack=stack)
                             for c in deals])
        assert runs[:2] == runs[2:]                                 # same seed, same deals: same results
        assert all(abs(x) <= 100 for r in runs for x in r)


def test_simulated_reads_pass_the_live_evidence_gates():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ablation import read
    _, state = read(("limper", 200))
    # The limper never bets when checked to: with identities attached the gate approves that node.
    assert state["evidence"]["nodes"]
    assert "flop_bet_when_checked_to" in state["profile"]["nodes"]
    json.dumps(state["evidence"])                                  # the evidence report stays serialisable
