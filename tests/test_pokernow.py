"""Checks for the PokerNow log parser and server (experiment 31)."""
import json
from pathlib import Path

import pytest

from bot.engine import POLICY
from bot.pokernow.parse import cards, describe, parse_hand, read_log

DEMO = Path(__file__).resolve().parents[1] / "bot" / "pokernow" / "demo_log.csv"
MSGS = read_log(DEMO.read_text())


def upto(text):
    """The demo log up to and including the first line containing `text`."""
    k = next(i for i, m in enumerate(MSGS) if text in m)
    return MSGS[:k + 1]


def test_cards_and_csv_order():
    assert cards("[A♠, 10♥, 2♦]") == ["As", "Th", "2d"]
    assert cards("K♥, 8♦, 3♣ [2♠]") == ["Kh", "8d", "3c", "2s"]
    assert MSGS[0].startswith("-- starting hand #1") and MSGS[-1] == "-- ending hand #2 --"


def test_latest_hand_seats_button_and_blinds():
    h = parse_hand(upto("Your hand is K♠"), hero_seat=3)
    t = describe(h)
    assert t["hand"] == 2 and t["heads_up"]
    assert t["button"] == "Friend" and not t["hero_has_button"]
    assert t["hero"]["seat"] == 3 and t["hero"]["position"] == "big blind"
    assert t["villain"]["seat"] == 7 and t["villain"]["position"] == "BTN / small blind"
    assert t["hero"]["stack_start"] == 19.90 and h.bb == 0.20 and h.sb == 0.10
    assert t["hero"]["bet_this_street"] == 0.20 and t["villain"]["bet_this_street"] == 0.10
    assert t["pot_main"] == 0 and t["pot_total"] == 0.30          # blinds are bets, not yet the pot
    assert t["to_act"] == "villain"                                 # the BTN acts first preflop


def test_bets_pot_and_to_call_are_separate():
    t = describe(parse_hand(upto("bets 1.80"), hero_seat=3))
    assert t["street"] == "turn" and t["board"] == ["Kh", "8d", "3c", "2s"]
    assert t["villain"]["bet_this_street"] == 1.80 and t["hero"]["bet_this_street"] == 0
    assert t["pot_main"] == 2.00                                    # 0.60 x 2 preflop + 0.40 x 2 flop
    assert t["pot_total"] == 3.80
    assert t["to_call"] == 1.80 and t["to_call_bb"] == 9.0
    assert t["hero"]["stack_now"] == 18.90 and t["to_act"] == "hero"


def test_history_in_big_blinds():
    h = parse_hand(upto("bets 4.10"), hero_seat=3)
    pre, flop, turn, river = h.history
    assert [(a.player, a.kind, a.to) for a in pre] == [("btn", "raise", 3.0), ("bb", "call", 0.0)]
    assert [a.kind for a in flop] == ["check", "bet", "call"] and flop[1].to == 2.0
    assert turn[1].to == 9.0 and abs(river[1].to - 20.5) < 1e-9


def test_seat_found_by_name_and_hand_over():
    h = parse_hand(MSGS, hero_name="Allen")
    assert h.hero_seat == 3 and h.over and describe(h)["to_act"] is None


def test_waits_for_the_next_card():
    h = parse_hand(upto("calls 1.80"), hero_seat=3)                # turn called, river not dealt yet
    assert h.to_act() is None


@pytest.mark.skipif(not (POLICY / "preflop.json").exists(), reason="match policies not built")
def test_server_decisions_through_the_demo_hand():
    from bot.pokernow.server import decide
    seen = []
    for k in range(1, len(MSGS) + 1):
        text = "entry,at,order\n" + "".join(f'"{m.replace(chr(34), chr(34) * 2)}",t,{i}\n' for i, m in enumerate(MSGS[:k]))
        out = decide({"log": text, "seat": 3})
        json.dumps(out)                                             # the response must be JSON-serialisable
        if out["decision"]:
            seen.append((out["table"]["street"], out["decision"]["node"]))
    streets = [s for s, _ in seen]
    assert streets[0] == "preflop" and {"flop", "turn", "river"} <= set(streets)
    assert any(n.startswith("bb_vs_raise") for _, n in seen)
