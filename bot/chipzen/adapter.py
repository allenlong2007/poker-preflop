"""Chipzen.ai adapter: plays HUBot heads-up through the Chipzen SDK (remote / External API path).

Chipzen sends a `GameState` per decision (chips, a flat action history with synthetic blind posts, legal raise
range). The engine wants big blinds and per-street histories of Act(position, kind, street total). This module
converts one into the other, then turns the engine's Decision back into a legal Chipzen Action.

Blinds escalate during a Chipzen match, so the big blind is read from each hand's own blind posts, never assumed.
Only heads-up is supported; at a bigger table the bot checks or folds.
"""
import json
import logging
import os
import time
from pathlib import Path

from bot.engine import Act, Engine

log = logging.getLogger("hubot.chipzen")

PHASES = ("preflop", "flop", "turn", "river")
POSTS = ("post_small_blind", "post_big_blind", "post_ante")


def v8_engine():
    """The v8 configuration (same as bot/pokernow/server.py on this branch)."""
    return Engine(pot_odds=True, exact_required=False, postflop_depths=(25, 40, 60, 100), postflop_nearest=True)


def big_blind(history, fallback=None):
    """This hand's big blind in chips, from its blind posts (a short all-in blind posts less than the level)."""
    bb = sb = 0
    for a in history:
        if a.get("action") == "post_big_blind":
            bb = max(bb, int(a.get("amount") or 0))
        elif a.get("action") == "post_small_blind":
            sb = max(sb, int(a.get("amount") or 0))
    level = max(bb, 2 * sb)
    return level if level > 0 else fallback


def translate(state, bb):
    """GameState -> (hero position, engine history, effective stack in bb). Heads-up only."""
    if len(state.opponent_stacks) != 1:
        raise ValueError(f"not heads-up: {len(state.opponent_stacks) + 1} seats")
    me = state.your_seat
    pos = lambda seat: "btn" if seat == state.dealer_seat else "bb"      # heads-up: the button is the small blind
    history = [[] for _ in PHASES]
    put_in, street_in, raised = {}, {}, {}
    for a in state.action_history:
        seat, kind, amt = a["seat"], a["action"], int(a.get("amount") or 0)
        street = PHASES.index(a.get("phase", "preflop"))
        key = (seat, street)
        if kind in POSTS:
            put_in[seat] = put_in.get(seat, 0) + amt
            if kind != "post_ante":
                street_in[key] = street_in.get(key, 0) + amt
            continue
        if kind == "call":
            put_in[seat] = put_in.get(seat, 0) + amt
            street_in[key] = street_in.get(key, 0) + amt
            history[street].append(Act(pos(seat), "call"))
        elif kind in ("raise", "bet", "all_in"):
            put_in[seat] = put_in.get(seat, 0) + amt - street_in.get(key, 0)
            street_in[key] = amt
            name = "raise" if street == 0 or raised.get(street) else "bet"
            raised[street] = True
            history[street].append(Act(pos(seat), name, round(amt / bb, 4)))
        elif kind in ("check", "fold"):
            history[street].append(Act(pos(seat), kind))
    opp = next(s for s in put_in if s != me) if any(s != me for s in put_in) else None
    start_me = state.your_stack + put_in.get(me, 0)
    start_opp = state.opponent_stacks[0] + (put_in.get(opp, 0) if opp is not None else 0)
    street = PHASES.index(state.phase)
    return pos(me), history[:street + 1], min(start_me, start_opp) / bb


def to_action(d, state, bb):
    """Engine Decision -> a legal Chipzen Action (as an (action, amount) pair)."""
    valid = set(state.valid_actions)
    if d.action in ("bet", "raise") and "raise" in valid:
        lo, hi = min(state.min_raise, state.max_raise), state.max_raise
        amount = max(lo, min(hi, int(round(d.to * bb))))
        return "raise", amount
    if d.action in ("bet", "raise", "call"):
        if "call" in valid:
            return "call", 0
        if "check" in valid:
            return "check", 0
    if d.action == "check" and "check" in valid:
        return "check", 0
    if "check" in valid:            # never fold when checking is free
        return "check", 0
    return "fold", 0


def safe_default(state):
    return ("check", 0) if "check" in state.valid_actions else ("fold", 0)


class Decider:
    """The SDK-free core: GameState in, (action, amount) out, with a JSONL decision log."""

    def __init__(self, engine=None, log_dir=None):
        self.engine = engine or v8_engine()
        self.last_bb = None
        self.log_path = Path(log_dir) / "decisions.jsonl" if log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.log_path.parent, 0o700)

    def __call__(self, state):
        t0 = time.perf_counter()
        row = {"t": time.time(), "hand": state.hand_number, "phase": state.phase,
               "hole": "".join(map(str, state.hole_cards)), "board": "".join(map(str, state.board))}
        try:
            bb = big_blind(state.action_history, self.last_bb)
            if not bb:
                raise ValueError("no big blind found")
            self.last_bb = bb
            hero, history, eff = translate(state, bb)
            d = self.engine.decide(hero, row["hole"], row["board"], history, stack=eff)
            out = to_action(d, state, bb)
            row.update(bb=bb, hero=hero, eff_bb=round(eff, 2), engine=[d.action, d.to], source=d.source)
        except Exception as e:                                   # never miss the clock over a bug
            log.exception("decision failed, playing the safe default")
            out = safe_default(state)
            row["error"] = repr(e)
        row.update(action=list(out), ms=round((time.perf_counter() - t0) * 1000, 1))
        if self.log_path:
            fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            os.fchmod(fd, 0o600)                                # in case it existed with wider permissions
            with os.fdopen(fd, "a") as f:
                f.write(json.dumps(row) + "\n")
        return out


def make_bot(log_dir=None, engine=None):
    """A chipzen.Bot subclass instance wrapping Decider (imports the SDK lazily)."""
    from chipzen import Action, Bot

    class HUBotChipzen(Bot):
        def __init__(self):
            super().__init__()
            self.decider = Decider(engine=engine, log_dir=log_dir)

        def on_match_start(self, match_info):
            cfg = (match_info or {}).get("game_config") or {}
            if cfg.get("big_blind"):
                self.decider.last_bb = int(cfg["big_blind"])
            log.info("match start: %s", {k: cfg.get(k) for k in ("starting_stack", "small_blind", "big_blind", "ante")})

        def decide(self, state):
            kind, amount = self.decider(state)
            return {"fold": Action.fold, "check": Action.check, "call": Action.call}[kind]() if kind != "raise" \
                else Action.raise_to(amount)

    return HUBotChipzen()
