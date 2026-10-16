"""PokerNow game log -> the current hand, seen from one fixed seat (experiment 31).

PokerNow's log (the "Download log" CSV, or the JSON the log panel loads) has one entry per event:

    -- starting hand #12 (id: abc123)  (No Limit Texas Hold'em) (dealer: "Alice @ AbC1") --
    Player stacks: #1 "Alice @ AbC1" (20.00) | #4 "Bob @ XyZ9" (19.40)
    "Alice @ AbC1" posts a small blind of 0.10
    "Bob @ XyZ9" posts a big blind of 0.20
    Your hand is A♠, 10♥
    "Alice @ AbC1" raises to 0.60
    "Bob @ XyZ9" calls 0.60
    Flop:  [J♥, 5♣, 2♦]           Turn: J♥, 5♣, 2♦ [7♠]           River: J♥, 5♣, 2♦, 7♠ [K♣]
    "Bob @ XyZ9" checks / bets 0.40 / folds / raises to 2.00 and go all in
    -- ending hand #12 --

`parse_hand` reads the latest hand and returns a `Hand`: who sits where, who has the button, stacks, each
player's bet on the current street, the pot, the board, the hero's cards and the action history in the
engine's format (big blinds). The hero is whoever sits in the locked seat.
"""
import csv
import io
import json
import re
from dataclasses import dataclass, field

from bot.engine import Act, State

SUITS = {"♠": "s", "♥": "h", "♦": "d", "♣": "c", "s": "s", "h": "h", "d": "d", "c": "c"}
PLAYER = r'"(?P<name>.+?) @ (?P<pid>[^"]+)"'
RE_START = re.compile(r"-- starting hand #(?P<no>\d+).*?(?:\(dealer: " + PLAYER + r"\))?\s*--")
RE_END = re.compile(r"-- ending hand #(?P<no>\d+)")
RE_STACKS = re.compile(r'#(?P<seat>\d+) "(?P<name>.+?) @ (?P<pid>[^"]+)" \((?P<stack>[\d.]+)\)')
RE_ACTION = re.compile(PLAYER + r" (?P<verb>posts a small blind of|posts a big blind of|posts a straddle of|"
                                 r"raises to|bets|calls|checks|folds)(?: (?P<amt>[\d.]+))?(?P<allin> and go all in)?")
RE_HAND = re.compile(r"Your hand is (?P<cards>.+)")
RE_BOARD = re.compile(r"^(?P<street>Flop|Turn|River)[^:]*:\s*(?P<cards>.+)$")
RE_CARD = re.compile(r"(10|[2-9TJQKA])\s*([♠♥♦♣shdc])")


def card(tok):
    m = RE_CARD.search(tok)
    if not m:
        return None
    r = "T" if m.group(1) == "10" else m.group(1)
    return r + SUITS[m.group(2)]


def cards(text):
    return [c for c in (card(t) for t in re.split(r"[,\[\]]", text)) if c]


def read_log(text, kind="auto"):
    """Raw log (CSV or JSON text) -> list of messages, oldest first."""
    text = text.strip()
    if kind == "json" or (kind == "auto" and text[:1] in "[{"):
        data = json.loads(text)
        rows = data.get("logs", data) if isinstance(data, dict) else data
        rows = sorted(rows, key=lambda r: (str(r.get("created_at", r.get("at", ""))), r.get("order", 0)))
        return [r["msg"] for r in rows]
    rows = list(csv.DictReader(io.StringIO(text)))
    if rows and "order" in rows[0]:
        rows.sort(key=lambda r: int(r["order"]))
    elif rows and "at" in rows[0]:
        rows.sort(key=lambda r: r["at"])
    return [r["entry"] for r in rows]


@dataclass
class Seat:
    seat: int
    name: str
    pid: str
    start: float                    # chips at the start of the hand
    put_in: float = 0.0             # chips put in this hand (all streets)
    street_bet: float = 0.0         # chips in on the current street
    folded: bool = False
    all_in: bool = False


@dataclass
class Hand:
    number: int = 0
    seats: dict = field(default_factory=dict)       # seat number -> Seat
    dealer_name: str = ""
    bb: float = 0.20                # big blind in chips (the unit for the engine)
    sb: float = 0.10
    hole: list = field(default_factory=list)
    board: list = field(default_factory=list)
    history: list = field(default_factory=lambda: [[]])   # engine Acts per street (big blinds)
    street: int = 0                 # 0 preflop ... 3 river
    over: bool = False
    hero_seat: int | None = None
    notes: list = field(default_factory=list)

    # -- who is who (heads-up: the dealer is the small blind / BTN) ------------------------------
    def hero(self):
        return self.seats.get(self.hero_seat)

    def villain(self):
        others = [s for k, s in self.seats.items() if k != self.hero_seat]
        return others[0] if len(others) == 1 else None

    def heads_up(self):
        return len(self.seats) == 2 and self.hero() is not None

    def position(self, s):
        return "btn" if s.name == self.dealer_name else "bb"

    def by_position(self, pos):
        return next((s for s in self.seats.values() if self.position(s) == pos), None)

    # -- pot and stacks (chips) ------------------------------------------------------------------
    def pot_main(self):
        """Chips in the middle from earlier streets (not counting this street's bets)."""
        return sum(s.put_in - s.street_bet for s in self.seats.values())

    def pot_total(self):
        return sum(s.put_in for s in self.seats.values())

    def stack_now(self, s):
        return s.start - s.put_in

    def effective_bb(self):
        return min(s.start for s in self.seats.values()) / self.bb

    def to_act(self):
        """'btn' / 'bb' whose turn it is in the replayed betting, or None if the hand is over."""
        if self.over or not self.heads_up():
            return None
        st = State(self.history, stack=self.effective_bb())
        if any(st.left(p) <= 1e-9 for p in ("btn", "bb")) and st.facing("btn") <= 1e-9 and st.facing("bb") <= 1e-9:
            return None                                     # all-in and called: no more decisions
        acts = st.history[st.street]
        if st.street >= 1 and len(self.board) < 2 + st.street:
            return None                                     # waiting for the next card
        if not acts:
            return "btn" if st.street == 0 else "bb"
        return "bb" if acts[-1].player == "btn" else "btn"


def parse_hand(messages, hero_seat=None, hero_name=None):
    """The latest hand in the log. hero_seat (locked seat number) wins over hero_name."""
    start = max((k for k, m in enumerate(messages) if RE_START.search(m)), default=None)
    h = Hand(hero_seat=hero_seat)
    if start is None:
        h.notes.append("no hand started yet")
        return h
    m = RE_START.search(messages[start])
    h.number = int(m.group("no"))
    h.dealer_name = m.group("name") or ""
    by_name = {}
    for msg in messages[start + 1:]:
        if msg.startswith("Player stacks"):
            for sm in RE_STACKS.finditer(msg):
                s = Seat(int(sm.group("seat")), sm.group("name"), sm.group("pid"), float(sm.group("stack")))
                h.seats[s.seat] = s
                by_name[s.name] = s
            if h.hero_seat is None and hero_name:
                h.hero_seat = next((s.seat for s in h.seats.values() if s.name == hero_name), None)
            continue
        if RE_END.search(msg):
            h.over = True
            break
        if (hm := RE_HAND.match(msg)):
            h.hole = cards(hm.group("cards"))
            continue
        if (bm := RE_BOARD.match(msg)):
            h.board = cards(bm.group("cards"))
            new_street = {"Flop": 1, "Turn": 2, "River": 3}[bm.group("street")]
            while h.street < new_street:
                h.street += 1
                h.history.append([])
                for s in h.seats.values():
                    s.street_bet = 0.0
            continue
        am = RE_ACTION.search(msg)
        if not am or am.group("name") not in by_name:
            continue
        s = by_name[am.group("name")]
        verb, amt = am.group("verb"), float(am.group("amt") or 0)
        pos = h.position(s)
        if verb.startswith("posts"):
            if "small" in verb:
                h.sb = amt
            elif "big" in verb:
                h.bb = amt
            else:
                h.notes.append("straddle: not modelled")
            s.put_in += amt - s.street_bet
            s.street_bet = amt
            continue
        if verb == "folds":
            s.folded = True
            h.history[h.street].append(Act(pos, "fold"))
            h.over = True
        elif verb == "checks":
            h.history[h.street].append(Act(pos, "check"))
        elif verb == "calls":
            # PokerNow writes the street total for a call; use the opponent's bet, capped by the stack.
            other = max((o.street_bet for o in h.seats.values() if o is not s), default=0.0)
            to = min(other, s.start - (s.put_in - s.street_bet))
            s.put_in += to - s.street_bet
            s.street_bet = to
            h.history[h.street].append(Act(pos, "call"))
        else:                                               # bets / raises to (street total)
            s.put_in += amt - s.street_bet
            s.street_bet = amt
            kind = "bet" if verb == "bets" else "raise"
            h.history[h.street].append(Act(pos, kind, round(amt / h.bb, 4)))
        if am.group("allin"):
            s.all_in = True
    if h.hero_seat is None and hero_name is None:
        h.notes.append("hero seat not set")
    return h


def describe(h):
    """A labeled snapshot for the overlay (chips and big blinds kept separate)."""
    hero, vil = h.hero(), h.villain()
    bb = lambda x: round(x / h.bb, 2)
    out = {"hand": h.number, "street": ["preflop", "flop", "turn", "river"][min(h.street, 3)],
           "heads_up": h.heads_up(), "over": h.over, "big_blind_chips": h.bb, "notes": list(h.notes),
           "board": h.board, "hero_cards": h.hole, "hero_seat": h.hero_seat,
           "button": h.dealer_name, "hero_has_button": bool(hero and hero.name == h.dealer_name)}
    for key, s in (("hero", hero), ("villain", vil)):
        if s is None:
            out[key] = None
            continue
        out[key] = {"seat": s.seat, "name": s.name, "position": "BTN / small blind" if h.position(s) == "btn"
                    else "big blind", "stack_start": s.start, "stack_now": round(h.stack_now(s), 2),
                    "stack_now_bb": bb(h.stack_now(s)), "bet_this_street": round(s.street_bet, 2),
                    "bet_this_street_bb": bb(s.street_bet), "folded": s.folded, "all_in": s.all_in}
    out["pot_main"] = round(h.pot_main(), 2)
    out["pot_main_bb"] = bb(h.pot_main())
    out["pot_total"] = round(h.pot_total(), 2)
    out["pot_total_bb"] = bb(h.pot_total())
    if hero and vil:
        out["to_call"] = round(max(0.0, vil.street_bet - hero.street_bet), 2)
        out["to_call_bb"] = bb(max(0.0, vil.street_bet - hero.street_bet))
        out["effective_stack_bb"] = round(h.effective_bb(), 1)
    nxt = h.to_act()
    out["to_act"] = None if nxt is None else ("hero" if hero and h.position(hero) == nxt else "villain")
    return out
