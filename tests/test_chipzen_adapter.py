"""Chipzen adapter: state translation, legal actions, and a full freezeout against a mini Chipzen-style dealer."""
import random

import pytest

chipzen = pytest.importorskip("chipzen")
from chipzen import Card, GameState  # noqa: E402

from bot.chipzen.adapter import Decider, big_blind, to_action, translate  # noqa: E402
from bot.engine import Decision  # noqa: E402


def gs(**kw):
    base = dict(hand_number=1, phase="preflop", hole_cards=[Card.from_str("Ah"), Card.from_str("Kd")], board=[],
                pot=30, your_stack=990, opponent_stacks=[980], your_seat=0, dealer_seat=0, to_call=10,
                min_raise=40, max_raise=1000, valid_actions=["fold", "call", "raise"], action_history=[])
    base.update(kw)
    return GameState(**base)


BLINDS = [{"seat": 0, "action": "post_small_blind", "amount": 10, "phase": "preflop"},
          {"seat": 1, "action": "post_big_blind", "amount": 20, "phase": "preflop"}]


def test_big_blind_reads_the_hand_posts_and_short_posts():
    assert big_blind(BLINDS) == 20
    short = [dict(BLINDS[0]), dict(BLINDS[1], amount=7)]
    assert big_blind(short) == 20                         # short all-in BB: the level is still 2 x SB
    assert big_blind([], fallback=50) == 50


def test_translate_button_open_and_effective_stack():
    s = gs(your_seat=1, dealer_seat=0, your_stack=980, opponent_stacks=[950], to_call=40,
           action_history=BLINDS + [{"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"}])
    hero, hist, eff = translate(s, 20)
    assert hero == "bb"
    assert [(a.player, a.kind, a.to) for a in hist[0]] == [("btn", "raise", 2.5)]
    assert eff == pytest.approx(min(980 + 20, 950 + 50) / 20)


def test_translate_postflop_bet_then_raise_labels():
    h = BLINDS + [{"seat": 0, "action": "call", "amount": 10, "phase": "preflop"},
                  {"seat": 1, "action": "check", "amount": 0, "phase": "preflop"},
                  {"seat": 1, "action": "raise", "amount": 20, "phase": "flop"},
                  {"seat": 0, "action": "raise", "amount": 60, "phase": "flop"}]
    s = gs(phase="flop", board=[Card.from_str(c) for c in ("2c", "7d", "Qs")], your_seat=1, dealer_seat=0,
           your_stack=960, opponent_stacks=[920], action_history=h)
    hero, hist, eff = translate(s, 20)
    assert [(a.player, a.kind, a.to) for a in hist[1]] == [("bb", "bet", 1.0), ("btn", "raise", 3.0)]
    assert eff == pytest.approx(50.0)


def test_to_action_clamps_and_never_folds_a_free_check():
    s = gs(min_raise=40, max_raise=300, valid_actions=["fold", "call", "raise"])
    assert to_action(Decision("raise", 100.0, {}, "", ""), s, 20) == ("raise", 300)
    assert to_action(Decision("raise", 1.0, {}, "", ""), s, 20) == ("raise", 40)
    free = gs(to_call=0, valid_actions=["check", "raise"])
    assert to_action(Decision("fold", 0, {}, "", ""), free, 20) == ("check", 0)
    short = gs(valid_actions=["fold", "call"])
    assert to_action(Decision("raise", 5.0, {}, "", ""), short, 20) == ("call", 0)


def deal_hand(rng, stacks, sb, button, deciders, hand_no):
    """One heads-up hand with Chipzen semantics (raise = street total, call = chips added). Returns new stacks."""
    deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
    rng.shuffle(deck)
    hole = {0: deck[:2], 1: deck[2:4]}
    board_all = deck[4:9]
    other = 1 - button
    stack = dict(stacks)
    hist, put = [], {0: 0, 1: 0}
    street_in = {0: 0, 1: 0}

    def post(seat, amt, kind):
        amt = min(amt, stack[seat])
        stack[seat] -= amt
        put[seat] += amt
        street_in[seat] += amt
        hist.append({"seat": seat, "action": kind, "amount": amt, "phase": "preflop"})

    post(button, sb, "post_small_blind")
    post(other, 2 * sb, "post_big_blind")
    folded = None
    for street, phase in enumerate(("preflop", "flop", "turn", "river")):
        street_in = street_in if street == 0 else {0: 0, 1: 0}
        high = 2 * sb if street == 0 else 0                 # the bet to match is the full level
        last_inc = 2 * sb
        actor = button if street == 0 else other
        pending = {s for s in (0, 1) if stack[s] > 0}
        while pending:
            if actor not in pending:
                actor = 1 - actor
                continue
            to_call = max(0, high - street_in[actor])
            if stack[1 - actor] == 0 and to_call == 0:      # nothing left to decide against an all-in
                pending.discard(actor)
                continue
            max_raise = street_in[actor] + stack[actor]
            valid = ["fold", "call"] if to_call else ["check"]
            if max_raise > high and stack[1 - actor] > 0:
                valid.append("raise")
            state = GameState(hand_number=hand_no, phase=phase, hole_cards=[Card.from_str(c) for c in hole[actor]],
                              board=[Card.from_str(c) for c in board_all[:(0, 3, 4, 5)[street]]],
                              pot=sum(put.values()), your_stack=stack[actor], opponent_stacks=[stack[1 - actor]],
                              your_seat=actor, dealer_seat=button, to_call=to_call, min_raise=high + last_inc,
                              max_raise=max_raise, valid_actions=valid, action_history=list(hist))
            kind, amount = deciders[actor](state)
            assert kind in valid, (kind, valid)
            pending.discard(actor)
            if kind == "fold":
                hist.append({"seat": actor, "action": "fold", "amount": 0, "phase": phase})
                folded = actor
                break
            if kind == "check":
                hist.append({"seat": actor, "action": "check", "amount": 0, "phase": phase})
            elif kind == "call":
                add = min(to_call, stack[actor])
                stack[actor] -= add
                put[actor] += add
                street_in[actor] += add
                hist.append({"seat": actor, "action": "call", "amount": add, "phase": phase})
            else:
                assert min(high + last_inc, max_raise) <= amount <= max_raise, (amount, high, last_inc, max_raise)
                stack[actor] -= amount - street_in[actor]
                put[actor] += amount - street_in[actor]
                street_in[actor] = amount
                last_inc = max(last_inc, amount - high)
                high = amount
                pending = {1 - actor} if stack[1 - actor] > 0 else set()
                hist.append({"seat": actor, "action": "raise", "amount": amount, "phase": phase})
            actor = 1 - actor
        if folded is not None:
            break
    # settle: refund uncalled, then pay the pot
    lo = min(put.values())
    for s in (0, 1):
        stack[s] += put[s] - lo
    pot = 2 * lo
    if folded is not None:
        stack[1 - folded] += pot
    else:
        import eval7
        b = [eval7.Card(c) for c in board_all]
        v = {s: eval7.evaluate([eval7.Card(c) for c in hole[s]] + b) for s in (0, 1)}
        if v[0] == v[1]:
            stack[0] += pot // 2
            stack[1] += pot - pot // 2
        else:
            stack[max(v, key=v.get)] += pot
    return stack


def test_full_freezeout_with_escalating_blinds_stays_legal(tmp_path):
    rng = random.Random(7)
    decider = Decider(log_dir=tmp_path)
    stacks, sb, button = {0: 1000, 1: 1000}, 5, 0
    for hand in range(1, 300):
        if hand % 20 == 0:
            sb = int(sb * 1.5)
        stacks = deal_hand(rng, stacks, sb, button, {0: decider, 1: decider}, hand)
        assert sum(stacks.values()) == 2000
        if 0 in stacks.values():
            break
        button = 1 - button
    rows = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert rows and not any('"error"' in r for r in rows)


def test_runner_refuses_a_shared_or_linked_token(tmp_path):
    import os
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from chipzen_remote import read_token
    tok = tmp_path / "token"
    tok.write_text("cz_extbot_x\n")
    os.chmod(tok, 0o600)
    assert read_token(tok) == "cz_extbot_x"
    os.chmod(tok, 0o644)
    with pytest.raises(SystemExit):
        read_token(tok)
    link = tmp_path / "link"
    link.symlink_to(tok)
    with pytest.raises(OSError):
        read_token(link)
    assert read_token(tmp_path / "missing") == ""


def test_hand_record_feeds_the_opponent_profile():
    from bot.chipzen.adapter import hand_record
    from bot.opponent import profile
    result = {"hand_number": 3, "payouts": [{"seat": 1, "amount": 1000}],
              "showdown": [{"seat": 0, "hole_cards": ["Ah", "Kd"]}, {"seat": 1, "hole_cards": ["Qs", "Qc"]}],
              "action_history": BLINDS + [
                  {"seat": 0, "action": "raise", "amount": 50, "phase": "preflop"},
                  {"seat": 1, "action": "raise", "amount": 500, "phase": "preflop"},
                  {"seat": 0, "action": "call", "amount": 450, "phase": "preflop"}],
              "stacks": [500, 1500]}
    r = hand_record(result, 0, "AhKd", "2c7d9sJhTd", "PluriBot", "m1")
    assert r["hero_pos"] == "btn" and r["opp_pos"] == "bb" and r["bb"] == 20
    assert r["stacks"] == {"hero": 50.0, "opp": 50.0}
    assert r["net"]["hero"] == -25.0 and r["shown"] == {"hero": "AhKd", "opp": "QsQc"}
    assert r["history"][0] == [["btn", "raise", 2.5], ["bb", "raise", 25.0], ["btn", "call", None]] or \
        r["history"][0][1] == ["bb", "raise", 25.0]
    prof = profile([r])
    assert prof["nodes"]["bb_vs_open"]["n"] == 1


def test_micro_pushfold_calls_wide_and_shoves_short():
    from bot.chipzen.adapter import v8_engine
    from bot.engine import Act
    e = v8_engine()
    d = e.decide("bb", "7h2h", "", [[Act("btn", "raise", 2.7)]], sample=False, stack=2.7)
    assert d.action == "call"
    d = e.decide("btn", "Kd7c", "", [[]], sample=False, stack=4.0)
    assert d.action == "raise" and d.to >= 3.99
