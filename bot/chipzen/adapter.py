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

from bot.engine import LIGHT3_MIN_BB, SIZE_CAP, SIZE_SMALL, Act, Engine, State, parse_cards, weak_showdown

log = logging.getLogger("hubot.chipzen")

PHASES = ("preflop", "flop", "turn", "river")
POSTS = ("post_small_blind", "post_big_blind", "post_ante")


def v8_engine(wide_3bet=False, reraise_defend=False, aggro_defend=False, deep_jam=False, light_3bet=False,
              river_air=False, size_tell=False, bet_more=False):
    """The v8 configuration (same as bot/pokernow/server.py on this branch), plus opt-in Chipzen stages."""
    return Engine(pot_odds=True, exact_required=False, postflop_depths=(25, 40, 60, 100), postflop_nearest=True,
                  micro_pushfold=True, wide_3bet=wide_3bet, reraise_defend=reraise_defend,
                  aggro_defend=aggro_defend, deep_jam=deep_jam, light_3bet=light_3bet,
                  river_air=river_air, size_tell=size_tell, bet_more=bet_more)


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


def fold3_seen(action_history, seat):
    """From a finished hand: (opponent folded to our 3-bet?, opponent faced our 3-bet?) as (0/1, 0/1)."""
    pre = [a for a in action_history if isinstance(a, dict) and a.get("phase", "preflop") == "preflop"
           and isinstance(a.get("action"), str)]
    button = next((a.get("seat") for a in pre if a["action"] == "post_small_blind"), None)
    acts = [a for a in pre if not a["action"].startswith("post")]
    if button is None or button == seat or len(acts) < 3 or acts[0].get("seat") != button \
            or acts[0]["action"] != "raise" or acts[1].get("seat") != seat or acts[1]["action"] != "raise" \
            or acts[2].get("seat") != button:
        return 0, 0
    return int(acts[2]["action"] == "fold"), 1


def size_tell_from_record(rec):
    """From a hand record: (1 if the opponent's river bet was big else 0, 1 if the hand it showed was weak else 0)
    when it bet the river (not a raise) and showed down; else None."""
    try:
        opp, hole, board = rec["opp_pos"], (rec.get("shown") or {}).get("opp"), rec.get("board") or ""
        if not (isinstance(hole, str) and len(hole) == 4 and isinstance(board, str) and len(board) == 10):
            return None
        hist = [[Act(*a) for a in street] for street in rec["history"]]
        if len(hist) < 4:
            return None
        for i, a in enumerate(hist[3]):
            if a.player == opp and a.kind == "bet":
                pot = State(hist[:3] + [hist[3][:i]]).street_start[3][0]
                if pot <= 0:
                    return None
                return int(a.to / pot >= SIZE_SMALL), int(weak_showdown(parse_cards(hole), parse_cards(board)))
        return None
    except Exception:
        return None


def add_size(sizes, key, tell):
    """Count one river-bet showdown (from size_tell_from_record) in sizes[key] = [[weak, n] small, [weak, n] big]."""
    if tell is None:
        return
    big, weak = tell
    c = sizes.setdefault(key, [[0, 0], [0, 0]])[big]
    c[0] += weak
    c[1] += 1
    if c[1] > SIZE_CAP:                                          # a sliding read: recent showdowns count
        c[0], c[1] = c[0] * SIZE_CAP / c[1], SIZE_CAP


def fold3_from_record(rec):
    """From a hand record: (opponent folded to our 3-bet?, faced it?), only for a non-jam 3-bet from the big blind
    vs their open with both stacks at least LIGHT3_MIN_BB, so the read matches the spots light_3bet acts in."""
    try:
        pre, st = rec["history"][0], rec["stacks"]
        hero, opp = float(st["hero"]), float(st["opp"])
        if rec.get("hero_pos") != "bb" or len(pre) < 3 or list(pre[0][:2]) != ["btn", "raise"] \
                or list(pre[1][:2]) != ["bb", "raise"] or pre[2][0] != "btn" or min(hero, opp) < LIGHT3_MIN_BB \
                or pre[1][2] is None or float(pre[1][2]) >= 0.95 * min(hero, opp):
            return 0, 0
        return int(pre[2][1] == "fold"), 1
    except Exception:
        return 0, 0


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


def folds_faced(acts, me):
    """One street's actions as (player, kind) pairs: (opponent folds to our bets/raises, times they faced one).
    Kinds: bet / raise / all_in are aggressive, fold folds; anything else answers without folding."""
    k = n = 0
    pending = False
    for who, kind in acts:
        if who == me:
            pending = kind in ("bet", "raise", "all_in")
            continue
        if pending:
            n += 1
            k += kind == "fold"
        pending = False
    return k, n


def fold_seen(action_history, seat):
    """From a finished hand: (opponent folds to our postflop bets/raises, times they faced one)."""
    k = n = 0
    for ph in PHASES[1:]:
        acts = [(a.get("seat") == seat, a.get("action")) for a in action_history
                if isinstance(a, dict) and a.get("phase") == ph and a.get("action") not in POSTS]
        fk, fn = folds_faced(acts, True)
        k, n = k + fk, n + fn
    return k, n


JAM_MIN_BB = 25.0


def jam_seen(rec):
    """From a hand record: (opponent preflop all-ins, opponent preflop actions), both only while 25bb+ deep."""
    k = n = 0
    try:
        opp = rec.get("opp_pos") or ("bb" if rec["hero_pos"] == "btn" else "btn")
        st = rec["stacks"]
        if min(float(st["hero"]), float(st["opp"])) < JAM_MIN_BB:
            return 0, 0
        for a in rec["history"][0]:
            if a[0] != opp:
                continue
            n += 1
            k += a[1] in ("raise", "bet") and a[2] is not None and float(a[2]) >= 0.95 * float(st["opp"])
    except Exception:
        return 0, 0
    return int(k), n


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


def load_reads(path, aggr=None, jams=None, f3=None, sizes=None, folds=None):
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
                    key, name = r.get("opp_key"), r.get("opp_name")
                    if (not isinstance(key, str) or key.startswith("pid:")) and isinstance(name, str) \
                            and 0 < len(name) < 64 and name not in ("unknown", "multi"):
                        key = f"name:{name}"                     # older records: per-match pid, so key on the name
                    if aggr is not None and isinstance(key, str) and key and r.get("hero_pos") in ("btn", "bb"):
                        opp = "bb" if r["hero_pos"] == "btn" else "btn"
                        acts = [a for street in r["history"][1:] for a in street if a[0] == opp]
                        c = aggr.setdefault(key, [0, 0])
                        c[0] += sum(a[1] in ("bet", "raise") for a in acts)
                        c[1] += len(acts)
                    if folds is not None and isinstance(key, str) and key and r.get("hero_pos") in ("btn", "bb"):
                        fk = fn = 0
                        for street in r["history"][1:]:
                            a, b = folds_faced([(a[0], a[1]) for a in street], r["hero_pos"])
                            fk, fn = fk + a, fn + b
                        if fn:
                            c = folds.setdefault(key, [0, 0])
                            c[0] += fk
                            c[1] += fn
                    if jams is not None and isinstance(key, str) and key and not key.startswith("pid:"):
                        jk, jn = jam_seen(r)
                        if jn:
                            c = jams.setdefault(key, [0, 0])
                            c[0] += jk
                            c[1] += jn
                    if sizes is not None and isinstance(key, str) and key and not key.startswith("pid:"):
                        add_size(sizes, key, size_tell_from_record(r))
                    pre = r["history"][0]
                    if f3 is not None and isinstance(key, str) and key and not key.startswith("pid:"):
                        fk, fn = fold3_from_record(r)
                        if fn:
                            c = f3.setdefault(key, [0, 0])
                            c[0] += fk
                            c[1] += fn
                    if not isinstance(key, str) or not key or r.get("hero_pos") != "btn" or len(pre) < 2 \
                            or list(pre[0][:2]) != ["btn", "raise"]:
                        continue
                    c = reads.setdefault(key, [0, 0])
                    c[0] += int(pre[1][1] == "raise")
                    c[1] += 1
                except Exception:
                    continue
    except Exception:
        for x in (aggr, jams, f3, sizes, folds):
            if x is not None:
                x.clear()
        return {}
    for c in reads.values():                                     # scale long histories down, keeping the rate
        if c[1] > READS_SEED_CAP:
            c[0], c[1] = c[0] * READS_SEED_CAP / c[1], READS_SEED_CAP
    for c in (f3 or {}).values():
        if c[1] > READS_SEED_CAP:
            c[0], c[1] = c[0] * READS_SEED_CAP / c[1], READS_SEED_CAP
    for c in list((jams or {}).values()) + list((folds or {}).values()):
        if c[1] > AGGR_SEED_CAP:
            c[0], c[1] = c[0] * AGGR_SEED_CAP / c[1], AGGR_SEED_CAP
    for key in [k for k, c in (aggr or {}).items() if not c[1]]:
        del aggr[key]
    for c in (aggr or {}).values():
        if c[1] > AGGR_SEED_CAP:
            c[0], c[1] = c[0] * AGGR_SEED_CAP / c[1], AGGR_SEED_CAP
    return reads


SCOUT_FILE = "scout_seeds.json"


def load_scout_seeds(path, f3, reads=None, aggr=None, sizes=None, folds=None, betmore=None):
    """Add scouted reads from other bots' replays: {"name:<bot>": {"f3": [folds to 3-bets, 3-bets faced],
    "3b": [3-bets, opens faced], "aggr": [postflop bets/raises, postflop actions],
    "fold": [folds to our postflop bets/raises, times faced], "bet_more": true (turn bet_more on vs this bot only)}}. Counts must be whole numbers
    with 0 <= k <= n <= the read's cap (READS_SEED_CAP, or AGGR_SEED_CAP for aggr); anything else is skipped, and
    any failure leaves the reads as they were."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for key, seed in list(data.items())[:100]:
        try:
            if not (isinstance(key, str) and key.startswith("name:") and 5 < len(key) < 69 and isinstance(seed, dict)) \
                    or key in ("name:unknown", "name:multi"):
                continue
            for name, store, cap in (("f3", f3, READS_SEED_CAP), ("3b", reads, READS_SEED_CAP),
                                     ("aggr", aggr, AGGR_SEED_CAP), ("fold", folds, AGGR_SEED_CAP)):
                if store is None or name not in seed:
                    continue
                pair = seed[name]
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                k, n = pair
                if not (type(k) is int and type(n) is int and 0 <= k <= n <= cap and n):
                    continue
                c = store.setdefault(key, [0, 0])
                c[0], c[1] = c[0] + k, c[1] + n
                if c[1] > cap:
                    c[0], c[1] = c[0] * cap / c[1], cap
            if betmore is not None and seed.get("bet_more") is True:
                betmore.add(key)
            sz = seed.get("sizes") if sizes is not None else None
            if isinstance(sz, (list, tuple)) and len(sz) == 2 and all(
                    isinstance(c, (list, tuple)) and len(c) == 2 and type(c[0]) is int and type(c[1]) is int
                    and 0 <= c[0] <= c[1] <= SIZE_CAP for c in sz):
                cur = sizes.setdefault(key, [[0, 0], [0, 0]])
                for i in (0, 1):
                    cur[i][0] += sz[i][0]
                    cur[i][1] += sz[i][1]
                    if cur[i][1] > SIZE_CAP:
                        cur[i][0], cur[i][1] = cur[i][0] * SIZE_CAP / cur[i][1], SIZE_CAP
        except Exception:
            continue


class Decider:
    """The SDK-free core: GameState in, (action, amount) out, with a JSONL decision log."""

    def __init__(self, engine=None, log_dir=None):
        self.engine = engine or v8_engine()
        self.last_bb = None
        self.seat, self.opp = None, "unknown"
        # opponent name -> [3-bets, our opens they faced], seeded from earlier matches' hand records
        # opponent -> [postflop bets/raises, postflop actions] (aggro_defend), from the same records
        self.aggr, self.jams = {}, {}                             # + [deep preflop jams, deep preflop actions]
        self.f3 = {}                                             # + [folds to our 3-bets, our 3-bets faced]
        self.sizes = {}                                          # + [[weak, n] small, [weak, n] big] river bets
        self.folds = {}                                          # + [folds to our postflop bets, times faced]
        self.betmore_for = set()                                 # opponents a scout seed turns bet_more on for
        self.bet_more_base = bool(getattr(self.engine, "bet_more", False))
        try:
            self.reads = load_reads(Path(log_dir) / "hands.jsonl", self.aggr, self.jams, self.f3,
                                    self.sizes, self.folds) if log_dir else {}
        except Exception:                                        # a bad file never stops the bot from starting
            self.reads, self.aggr, self.jams, self.f3, self.sizes, self.folds = {}, {}, {}, {}, {}, {}
        self.log_path = Path(log_dir) / "decisions.jsonl" if log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.log_path.parent, 0o700)
            try:                                                 # scouted reads, after the dir is private
                load_scout_seeds(Path(log_dir) / SCOUT_FILE, self.f3, self.reads, self.aggr, self.sizes, self.folds,
                                 self.betmore_for)
            except Exception:
                pass

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
            self.engine.opp_jams = tuple(self.jams.get(self.opp, (0, 0)))
            self.engine.opp_f3 = tuple(self.f3.get(self.opp, (0, 0)))
            self.engine.opp_fold = tuple(self.folds.get(self.opp, (0, 0)))
            self.engine.bet_more = self.bet_more_base or self.opp in self.betmore_for
            self.engine.opp_sizes = tuple(tuple(c) for c in self.sizes.get(self.opp, ((0, 0), (0, 0))))
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

    def observe(self, result, seat=None, board=None):
        """A finished hand (Chipzen round result): update this opponent's 3-bet count."""
        seat = self.seat if seat is None else seat
        if seat is None:
            return
        k, n = threebet_seen(result.get("action_history") or [], seat)
        if n:
            r = self.reads.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
        rec = hand_record(result, seat, "", "", "", "", self.last_bb)
        k, n = fold3_from_record(rec) if rec else (0, 0)
        if n:
            r = self.f3.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
            if r[1] > READS_SEED_CAP:
                r[0], r[1] = r[0] * READS_SEED_CAP / r[1], READS_SEED_CAP
        k, n = aggression_seen(result.get("action_history") or [], seat)
        if n:
            r = self.aggr.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
            if r[1] > AGGR_SEED_CAP:                             # a sliding read: recent play counts, old play fades
                r[0], r[1] = r[0] * AGGR_SEED_CAP / r[1], AGGR_SEED_CAP
        k, n = fold_seen(result.get("action_history") or [], seat)
        if n:
            r = self.folds.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
            if r[1] > AGGR_SEED_CAP:
                r[0], r[1] = r[0] * AGGR_SEED_CAP / r[1], AGGR_SEED_CAP
        rec = hand_record(result, seat, "", "", "", "", self.last_bb)
        k, n = jam_seen(rec) if rec else (0, 0)
        if n:
            r = self.jams.setdefault(self.opp, [0, 0])
            r[0] += k
            r[1] += n
            if r[1] > AGGR_SEED_CAP:
                r[0], r[1] = r[0] * AGGR_SEED_CAP / r[1], AGGR_SEED_CAP
        b = board if isinstance(board, str) and board else result.get("board")
        if rec and isinstance(b, str) and len(b) == 10:
            add_size(self.sizes, self.opp, size_tell_from_record(dict(rec, board=b)))


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
            # participant_id turned out to be per match (PluriBot had a new one in each of 4 matches, 2026-11-15),
            # so reads key on the bot's display name; the 30-open / 60-action caps bound what a name clash can do.
            name = others[0].get("display_name") if len(others) == 1 else None
            self.opp_key = f"name:{name}" if isinstance(name, str) and 0 < len(name) < 64 else None  # < 64: the
            # records store opp_name cut to 64, so a 64-char stored name may be a longer one; neither path keys it
            self.decider.set_opponent(self.opp_key)            # no usable name: no carried-over read
            if len(others) == 1:
                log.info("opponent seat fields: %s", sorted(str(k)[:32] for k in others[0])[:20])
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
                self.decider.observe((message or {}).get("result") or {}, self.me, self.board)
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
