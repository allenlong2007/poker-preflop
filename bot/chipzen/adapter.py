"""Chipzen.ai adapter: plays HUBot heads-up through the Chipzen SDK (remote / External API path).

Chipzen sends a `GameState` per decision (chips, a flat action history with synthetic blind posts, legal raise
range). The engine wants big blinds and per-street histories of Act(position, kind, street total). This module
converts one into the other, then turns the engine's Decision back into a legal Chipzen Action.

Blinds escalate during a Chipzen match, so the big blind is read from each hand's own blind posts, never assumed.
Only heads-up is supported; at a bigger table the bot checks or folds.
"""
import itertools
import json
import logging
import os
import time
from pathlib import Path

from bot.engine import Act, Engine

log = logging.getLogger("hubot.chipzen")

PHASES = ("preflop", "flop", "turn", "river")
POSTS = ("post_small_blind", "post_big_blind", "post_ante")


def v8_engine(wide_3bet=False, reraise_defend=False, aggro_defend=False):
    """The v8 configuration (same as bot/pokernow/server.py on this branch), plus opt-in Chipzen stages."""
    return Engine(pot_odds=True, exact_required=False, postflop_depths=(25, 40, 60, 100), postflop_nearest=True,
                  micro_pushfold=True, wide_3bet=wide_3bet, reraise_defend=reraise_defend,
                  aggro_defend=aggro_defend)


def threebet_seen(action_history, seat):
    """From a finished hand: (opponent 3-bet?, opponent faced our button open?) as (0/1, 0/1)."""
    pre = [a for a in action_history if isinstance(a, dict) and a.get("phase", "preflop") == "preflop"
           and isinstance(a.get("action"), str)]
    button = next((a.get("seat") for a in pre if a["action"] == "post_small_blind"), None)
    acts = [a for a in pre if not a["action"].startswith("post")]
    if button is None or button != seat or len(acts) < 2 or acts[0].get("seat") != seat \
            or acts[0]["action"] != "raise":
        return 0, 0
    return int(acts[1]["action"] == "raise"), 1


def aggression_seen(action_history, seat):
    """From a finished hand: (opponent postflop bets/raises, opponent postflop actions)."""
    k = n = 0
    last = {}                                                    # phase -> the previous action on that street
    for a in action_history:
        ph = a.get("phase") if isinstance(a, dict) else None
        if not isinstance(ph, str) or ph not in PHASES[1:]:
            continue
        prev, last[ph] = last.get(ph), a
        if a.get("seat") == seat or a.get("action") not in ("bet", "raise", "all_in", "check", "call", "fold"):
            continue
        # an all-in straight after our bet may be a raise or only a call for less: leave it out of the read
        if a["action"] == "all_in" and isinstance(prev, dict) and prev.get("seat") == seat \
                and prev.get("action") in ("bet", "raise", "all_in"):
            continue
        n += 1
        k += a["action"] in ("bet", "raise", "all_in")
    return k, n


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


def replay(action_history, dealer_seat, bb):
    """Chipzen's flat history -> (engine history: 4 per-street lists of Act, chips each seat put in)."""
    pos = lambda seat: "btn" if seat == dealer_seat else "bb"           # heads-up: the button is the small blind
    history = [[] for _ in PHASES]
    put_in, street_in, raised = {}, {}, {}
    for a in action_history:
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
    return history, put_in


def translate(state, bb):
    """GameState -> (hero position, engine history, effective stack in bb). Heads-up only."""
    if len(state.opponent_stacks) != 1:
        raise ValueError(f"not heads-up: {len(state.opponent_stacks) + 1} seats")
    me = state.your_seat
    history, put_in = replay(state.action_history, state.dealer_seat, bb)
    opp = next(s for s in put_in if s != me) if any(s != me for s in put_in) else None
    start_me = state.your_stack + put_in.get(me, 0)
    start_opp = state.opponent_stacks[0] + (put_in.get(opp, 0) if opp is not None else 0)
    street = PHASES.index(state.phase)
    pos = "btn" if me == state.dealer_seat else "bb"
    return pos, history[:street + 1], min(start_me, start_opp) / bb


def hand_record(result, me, hole, board, opp_name, match_id, fallback_bb=None, opp_key=None):
    """A finished Chipzen hand (round_result) -> a bot/opponent.py style record, from our point of view."""
    hist = result.get("action_history") or []
    bb = big_blind(hist, fallback_bb)
    sb_seat = next((a["seat"] for a in hist if a.get("action") == "post_small_blind"), None)
    seats = sorted({a["seat"] for a in hist})
    if not bb or sb_seat is None or len(seats) != 2 or me not in seats:
        return None
    opp = next(s for s in seats if s != me)
    history, put_in = replay(hist, sb_seat, bb)
    paid = {p["seat"]: int(p["amount"]) for p in result.get("payouts") or []}
    after = result.get("stacks") or []
    start = {s: (after[s] if s < len(after) else 0) + put_in.get(s, 0) - paid.get(s, 0) for s in (me, opp)}
    shown = {("hero" if e["seat"] == me else "opp"): "".join(e["hole_cards"]) for e in result.get("showdown") or []}
    pos = lambda s: "btn" if s == sb_seat else "bb"
    return {"id": f"{match_id}:{result.get('hand_number')}", "number": result.get("hand_number"),
            "match_id": match_id, "hero_id": "hubot", "hero_name": "HUBot_v8", "bot_version": 8,
            "opp_name": str(opp_name)[:64], "opp_id": f"chipzen:{str(opp_name)[:64]}",
            "opp_key": str(opp_key)[:128] if opp_key else None, "hero_pos": pos(me), "opp_pos": pos(opp),
            "bb": bb, "stacks": {"hero": round(start[me] / bb, 2), "opp": round(start[opp] / bb, 2)},
            "history": [[[a.player, a.kind, a.to] for a in street] for street in history],
            "board": board, "hero_hole": hole, "shown": shown,
            "net": {"hero": round((paid.get(me, 0) - put_in.get(me, 0)) / bb, 2)}}


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


_ANON = itertools.count()
READS_MAX_BYTES = 20_000_000   # hands.jsonl beyond this is not read (a season is far smaller)
READS_MAX_LINE = 100_000
AGGR_SEED_CAP = 60             # the same for carried-over postflop actions (aggro_defend)
READS_SEED_CAP = 30            # carried-over opens count as at most this many, so the current match still moves it


def load_reads(path, aggr=None):
    """Per-opponent [3-bets, our button opens they faced] from the stored hand records (hands.jsonl), keyed by the
    platform's participant_id (records without one are ignored), so the read carries across matches and restarts.
    Bad lines are skipped and any failure gives an empty read."""
    reads = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            used = 0
            while used <= READS_MAX_BYTES:
                line = f.readline(READS_MAX_LINE + 1)            # never holds more than the cap in memory
                if not line:
                    break
                used += len(line)
                if len(line) > READS_MAX_LINE:
                    while line and not line.endswith("\n") and used <= READS_MAX_BYTES:   # skip the rest of it
                        line = f.readline(READS_MAX_LINE + 1)
                        used += len(line)
                    continue
                try:
                    r = json.loads(line)
                    key = r.get("opp_key")
                    if aggr is not None and isinstance(key, str) and key and r.get("hero_pos") in ("btn", "bb"):
                        opp = "bb" if r["hero_pos"] == "btn" else "btn"
                        acts = [a for street in r["history"][1:] for a in street if a[0] == opp]
                        c = aggr.setdefault(key, [0, 0])
                        c[0] += sum(a[1] in ("bet", "raise") for a in acts)
                        c[1] += len(acts)
                    pre = r["history"][0]
                    if not isinstance(key, str) or not key or r.get("hero_pos") != "btn" or len(pre) < 2 \
                            or list(pre[0][:2]) != ["btn", "raise"]:
                        continue
                    c = reads.setdefault(key, [0, 0])
                    c[0] += int(pre[1][1] == "raise")
                    c[1] += 1
                except Exception:
                    continue
    except Exception:
        if aggr is not None:
            aggr.clear()
        return {}
    for c in reads.values():                                     # scale long histories down, keeping the rate
        if c[1] > READS_SEED_CAP:
            c[0], c[1] = c[0] * READS_SEED_CAP / c[1], READS_SEED_CAP
    for key in [k for k, c in (aggr or {}).items() if not c[1]]:
        del aggr[key]
    for c in (aggr or {}).values():
        if c[1] > AGGR_SEED_CAP:
            c[0], c[1] = c[0] * AGGR_SEED_CAP / c[1], AGGR_SEED_CAP
    return reads


class Decider:
    """The SDK-free core: GameState in, (action, amount) out, with a JSONL decision log."""

    def __init__(self, engine=None, log_dir=None):
        self.engine = engine or v8_engine()
        self.last_bb = None
        self.seat, self.opp = None, "unknown"
        # opponent name -> [3-bets, our opens they faced], seeded from earlier matches' hand records
        # opponent -> [postflop bets/raises, postflop actions] (aggro_defend), from the same records
        self.aggr = {}
        try:
            self.reads = load_reads(Path(log_dir) / "hands.jsonl", self.aggr) if log_dir else {}
        except Exception:                                        # a bad file never stops the bot from starting
            self.reads, self.aggr = {}, {}
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
            self.seat = state.your_seat
            self.engine.opp_3bet = tuple(self.reads.get(self.opp, (0, 0)))
            self.engine.opp_aggr = tuple(self.aggr.get(self.opp, (0, 0)))
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
            _append_private(self.log_path, row)
        return out


    def set_opponent(self, key):
        """The opponent's stable key (pid:<participant_id>). None: a fresh key that nothing else shares."""
        self.opp = key if key else f"anon:{next(_ANON)}"

    def observe(self, result, seat=None):
        """A finished hand (Chipzen round result): update this opponent's 3-bet count."""
        seat = self.seat if seat is None else seat
        if seat is None:
            return
        k, n = threebet_seen(result.get("action_history") or [], seat)
        if n:
            r = self.reads.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
        k, n = aggression_seen(result.get("action_history") or [], seat)
        if n:
            r = self.aggr.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
            if r[1] > AGGR_SEED_CAP:                             # a sliding read: recent play counts, old play fades
                r[0], r[1] = r[0] * AGGR_SEED_CAP / r[1], AGGR_SEED_CAP


def _append_private(path, row):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(row) + "\n")


def make_bot(log_dir=None, engine=None):
    """A chipzen.Bot subclass instance wrapping Decider (imports the SDK lazily). Every finished hand is stored as
    an opponent record in <log_dir>/hands.jsonl."""
    from chipzen import Action, Bot

    class HUBotChipzen(Bot):
        def __init__(self):
            super().__init__()
            self.decider = Decider(engine=engine, log_dir=log_dir)
            self.match_id, self.me, self.opp_name, self.opp_key = "", None, "unknown", None
            self.hole, self.board = "", ""

        def on_match_start(self, match_info):
            match_info = match_info or {}
            cfg = match_info.get("game_config") or {}
            if cfg.get("big_blind"):
                self.decider.last_bb = int(cfg["big_blind"])
            self.match_id = match_info.get("match_id", "")
            self.me = match_info.get("your_seat")
            others = [s for s in match_info.get("seats") or [] if not s.get("is_self")]
            self.opp_name = str(others[0].get("display_name", "unknown"))[:64] if len(others) == 1 else "multi"
            pid = others[0].get("participant_id") if len(others) == 1 else None
            self.opp_key = f"pid:{pid}" if isinstance(pid, str) and 0 < len(pid) <= 120 else None
            self.decider.set_opponent(self.opp_key)            # no stable id: no carried-over read
            log.info("match start vs %s: %s", self.opp_name,
                     {k: cfg.get(k) for k in ("starting_stack", "small_blind", "big_blind", "ante")})

        def on_round_start(self, message):
            st = (message or {}).get("state") or {}
            self.hole, self.board = "".join(st.get("your_hole_cards") or []), ""
            super().on_round_start(message)

        def on_phase_change(self, message):
            self.board = "".join(((message or {}).get("state") or {}).get("board") or []) or self.board

        def on_round_result(self, message):
            try:
                result = (message or {}).get("result") or {}
                if self.me is not None and log_dir:
                    rec = hand_record(result, self.me, self.hole, self.board, self.opp_name, self.match_id,
                                      self.decider.last_bb, self.opp_key)
                    if rec:
                        _append_private(Path(log_dir) / "hands.jsonl", rec)
            except Exception:
                log.exception("could not record the hand")
            try:                                                 # the 3-bet read never blocks the hand record
                self.decider.observe((message or {}).get("result") or {}, self.me)
            except Exception:
                log.exception("could not update the 3-bet read")
            super().on_round_result(message)

        def decide(self, state):
            self.board = "".join(map(str, state.board)) or self.board
            if self.me is None:
                self.me = state.your_seat
            kind, amount = self.decider(state)
            return {"fold": Action.fold, "check": Action.check, "call": Action.call}[kind]() if kind != "raise" \
                else Action.raise_to(amount)

    return HUBotChipzen()
