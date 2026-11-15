"""A mini Chipzen-style dealer for offline freezeouts (raise = street total, call = chips added, escalating blinds).

Deciders are callables GameState -> (action, amount), like bot.chipzen.adapter.Decider.
"""
import eval7
import numpy as np
from chipzen import Card, GameState

# Chipzen practice matches: 10000 chips, 50/100, the level rises every 20 hands (seen: 100, 150, 200, 300 bb).
START = 10000
LEVELS = (100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600)
HANDS_PER_LEVEL = 20


def deal_hand(rng, stacks, sb, button, deciders, hand_no, cards=None, out=None):
    """One heads-up hand. `cards` (9 card strings) fixes the deal for duplicate play. Returns new stacks. If `out`
    is a dict it receives a Chipzen-style round result (action_history, payouts, showdown, stacks, hand_number)
    plus the hole cards and the board dealt."""
    if cards is None:
        deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        rng.shuffle(deck)
        cards = deck[:9]
    hole = {button: cards[:2], 1 - button: cards[2:4]}
    board_all = cards[4:9]
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
    lo = min(put.values())                                  # refund the uncalled part, then pay the pot
    for s in (0, 1):
        stack[s] += put[s] - lo
    pot = 2 * lo
    before = dict(stack)
    shown = []
    if folded is not None:
        stack[1 - folded] += pot
    else:
        shown = [{"seat": t, "hole_cards": list(hole[t])} for t in (0, 1)]
        b = [eval7.Card(c) for c in board_all]
        v = {s: eval7.evaluate([eval7.Card(c) for c in hole[s]] + b) for s in (0, 1)}
        if v[0] == v[1]:
            stack[0] += pot // 2
            stack[1] += pot - pot // 2
        else:
            stack[max(v, key=v.get)] += pot
    if out is not None:
        out.update(hand_number=hand_no, action_history=hist, showdown=shown, stacks=[stack[0], stack[1]],
                   payouts=[{"seat": t, "amount": stack[t] - before[t] + put[t] - lo} for t in (0, 1)
                            if stack[t] - before[t] + put[t] - lo > 0],
                   hole={t: "".join(hole[t]) for t in (0, 1)}, board="".join(board_all),
                   street=max([("preflop", "flop", "turn", "river").index(a["phase"]) for a in hist]))
    return stack


def freezeout(deciders, seed, swap=False, max_hands=400, on_hand=None):
    """One Chipzen-format match. With swap=True seat 0 gets the cards seat 1 got with swap=False (duplicate play).
    Returns (winning seat or None, hands played)."""
    rng = np.random.default_rng(seed)
    deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
    stacks, button = {0: START, 1: START}, 0
    for hand in range(1, max_hands + 1):
        cards = [deck[i] for i in rng.choice(52, 9, replace=False)]
        bb = LEVELS[min((hand - 1) // HANDS_PER_LEVEL, len(LEVELS) - 1)]
        btn = 1 - button if swap else button
        out = {}
        stacks = deal_hand(None, stacks, bb // 2, btn, deciders, hand, cards, out)
        for seat, d in deciders.items():
            if hasattr(d, "observe"):
                d.observe(out, seat)
        if on_hand:
            on_hand(out, bb)
        if 0 in stacks.values():
            return max(stacks, key=stacks.get), hand
        button = 1 - button
    return None, max_hands
