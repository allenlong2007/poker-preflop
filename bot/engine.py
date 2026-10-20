"""The match bot's decision engine (experiment 29): what to do in any 100bb heads-up spot.

Given the hero's seat, hole cards, the board and the betting so far, it returns an action, its size and
the full action distribution. Nothing here talks to a poker site; an interface (PokerNow or otherwise)
only has to translate the table into `Act` lists and the decision back into clicks.

Sources, in order of preference:
  1. Preflop: experiment 26's solved tree (`output/match/policy/preflop.json`, per hand class). The
     opponent's open / 3-bet sizes are matched to the nearest trained size.
  2. Flop of a 3bb single-raised pot: experiment 28's TexasSolver solve of the representative flop for
     the board's texture class. The hero's combo is mapped onto that flop (suits and ranks relative to
     the board); if the mapped combo isn't in the solved range or lands in a different hand group, the
     solver's average over that hand group is used.
  3. Everything else postflop: the whole-hand model (`output/match/policy/<pot>.npz`) by texture, line
     and hand group. Opponent bet sizes are matched to the nearest trained size (in log terms).
  4. Spots no model covers (a re-raise of a raise postflop): a simple rule (jam the nuts, call strong made
     hands, fold the rest).

All amounts are in big blinds; `to` is the player's total chips put in on the current street.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import eval7
import numpy as np

from preflop import fullgame as fg
from preflop.cards import HANDS, INDEX
from preflop.flopgame import texture as flop_texture
from preflop.rivergame import river_texture
from preflop.turngame import turn_texture

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "output" / "match" / "policy"
POLICY_FINE = ROOT / "output" / "match" / "policy_fine"     # experiment 30: finer turn / river sizes
POTS = ("srp", "3bet", "limp", "iso", "srp_alt")
FLOPS = ROOT / "output" / "match" / "flops"
STACK = 100.0
EPS = 1e-6
RANK_CHARS = "23456789TJQKA"              # eval7 rank index order
SUIT_CHARS = "cdhs"                        # eval7 suit index order
JAM_FRACTION = 0.6                         # a preflop raise to >= 60% of the stack is treated as all-in
SOLVED_OPEN = 3.0                          # default open of the solved flops (index.json records the real one)


@dataclass
class Act:
    player: str                            # "btn" or "bb"
    kind: str                              # "fold", "check", "call", "bet", "raise"
    to: float = 0.0                        # bet / raise: the player's total on this street afterwards


@dataclass
class Decision:
    action: str                            # fold / check / call / bet / raise
    to: float                              # total on this street (0 for fold / check)
    probs: dict                            # label -> probability of every option at this node
    node: str
    source: str                            # preflop / solver / model / rule
    label: str = ""                        # the chosen option's label in `probs`
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- cards

def parse_cards(s):
    """'AhKd' or 'Ah Kd' or ['Ah', 'Kd'] -> list of eval7.Card."""
    if isinstance(s, str):
        s = s.replace(" ", "")
        s = [s[k:k + 2] for k in range(0, len(s), 2)]
    return [c if isinstance(c, eval7.Card) else eval7.Card(c) for c in s]


def card_str(c):
    return RANK_CHARS[c.rank] + SUIT_CHARS[c.suit]


def hand_class(hole):
    a, b = sorted(hole, key=lambda c: -c.rank)
    if a.rank == b.rank:
        return RANK_CHARS[a.rank] * 2
    return RANK_CHARS[a.rank] + RANK_CHARS[b.rank] + ("s" if a.suit == b.suit else "o")


# ----------------------------------------------------------------------------- betting state

class State:
    """Replays the history: pot, stacks, who is in for what on the current street."""

    def __init__(self, history, stack=STACK):
        self.stack = stack
        self.history = [list(h) for h in history] + [[] for _ in range(4 - len(history))]
        self.street = max(k for k in range(4) if k == 0 or self.history[k] or self._closed(k - 1))
        self.total = {"btn": 0.0, "bb": 0.0}       # chips in before the current street
        self.street_start = []                     # (pot, behind) at the start of each street
        for s in range(self.street + 1):
            behind = stack - max(self.total.values())
            self.street_start.append((sum(self.total.values()), behind))
            inp = self._street_in(s)
            if s < self.street:
                for p in inp:
                    self.total[p] += inp[p]
        self.inp = self._street_in(self.street)

    def _street_in(self, s):
        inp = {"btn": 0.5, "bb": 1.0} if s == 0 else {"btn": 0.0, "bb": 0.0}
        for a in self.history[s]:
            if a.kind in ("bet", "raise"):
                inp[a.player] = a.to
            elif a.kind == "call":
                other = "bb" if a.player == "btn" else "btn"
                inp[a.player] = min(inp[other], self.stack - self.total[a.player])
        return inp

    def _closed(self, s):
        acts = self.history[s]
        if not acts:
            return False
        last = acts[-1]
        if last.kind == "call":
            return not (s == 0 and len(acts) == 1)    # a preflop limp leaves the BB its option
        if last.kind == "check":
            return s == 0 or (len(acts) >= 2)          # BB checks the limp / both check
        return False

    def pot(self):
        return sum(self.total.values()) + sum(self.inp.values())

    def facing(self, hero):
        other = "bb" if hero == "btn" else "btn"
        return max(0.0, self.inp[other] - self.inp[hero])

    def left(self, player):
        """Chips the player still has behind."""
        return self.stack - self.total[player] - self.inp[player]


def _aggressor(acts):
    last = None
    for a in acts:
        if a.kind in ("bet", "raise"):
            last = a.player
    return {None: 0, "btn": 1, "bb": 2}[last]


def _nearest(x, options, log=True):
    f = (lambda v: math.log(max(v, 1e-9))) if log else (lambda v: v)
    return min(range(len(options)), key=lambda k: abs(f(x) - f(options[k])))


def size_weights(x, sizes, translate=True):
    """Map a bet of x (fraction of the pot) onto the trained sizes: [(size index, weight)].

    translate=True uses the pseudo-harmonic mapping (Ganzfried & Sandholm 2013): a bet between trained
    sizes A < x < B is treated as A with probability (B - x)(1 + A) / ((B - A)(1 + x)), else B, and the
    two nodes' strategies are blended with those weights. Bets outside the trained range use the end size.
    translate=False snaps to the nearest size (on a log scale).
    """
    if not translate:
        return [(_nearest(x, sizes), 1.0)]
    order = sorted(range(len(sizes)), key=lambda k: sizes[k])
    if x <= sizes[order[0]] + 1e-9:
        return [(order[0], 1.0)]
    if x >= sizes[order[-1]] - 1e-9:
        return [(order[-1], 1.0)]
    hi = next(n for n, k in enumerate(order) if sizes[k] >= x - 1e-9)
    a, b = order[hi - 1], order[hi]
    A, B = sizes[a], sizes[b]
    if abs(B - x) < 1e-9:
        return [(b, 1.0)]
    w = (B - x) * (1 + A) / ((B - A) * (1 + x))
    return [(a, w), (b, 1 - w)]


# ----------------------------------------------------------------------------- the engine

class Engine:
    def __init__(self, policy_dir=None, flops_dir=FLOPS, use_solver_flops=True, translate=True, seed=None):
        """policy_dir: postflop policies (default: experiment 30's finer sizes when all are trained, else
        experiment 26's). translate: blend neighbouring sizes when facing an untrained bet size."""
        if policy_dir is None:
            policy_dir = POLICY_FINE if all((POLICY_FINE / f"{p}.npz").exists() for p in POTS) else POLICY
        self.pre = json.load(open(POLICY / "preflop.json"))
        self.policy_dir, self.flops_dir = Path(policy_dir), Path(flops_dir)
        self.use_solver_flops = use_solver_flops and (Path(flops_dir) / "index.json").exists()
        meta = json.load(open(Path(flops_dir) / "index.json")) if self.use_solver_flops else {}
        self.index = meta.get("classes", {})
        self.solved_open = float(meta.get("open", SOLVED_OPEN))
        self.translate = translate
        self._pots, self._flops, self._opts = {}, {}, {}
        self.rng = np.random.default_rng(seed)

    # -- public ------------------------------------------------------------------------------

    def warm(self):
        """Load every policy and solved flop up front, so no decision during a game waits on disk."""
        for pot in POTS:
            if (self.policy_dir / f"{pot}.npz").exists():
                self._pot(pot)
                self.opts(pot)
        for t in self.index:
            self._flop(int(t))
        return self

    def decide(self, hero, hole, board, history, sample=True, stack=STACK):
        """stack: effective stack in big blinds (the policies were trained at 100bb; all-ins and bet caps
        use the real stack)."""
        hole, board = parse_cards(hole), parse_cards(board) if board else []
        st = State([[a if isinstance(a, Act) else Act(*a) for a in h] for h in history], stack=stack)
        if st.street == 0:
            d = self._preflop(hero, hole, st)
        else:
            d = self._postflop(hero, hole, board[:2 + st.street], st)
        if sample:
            labels = list(d.probs)
            p = np.array([d.probs[k] for k in labels], dtype=float)
            d.label = labels[int(self.rng.choice(len(labels), p=p / p.sum()))]
        else:
            d.label = max(d.probs, key=d.probs.get)
        d.action, d.to = self._options[d.label]
        return d

    # -- preflop -----------------------------------------------------------------------------

    def _preflop(self, hero, hole, st):
        i = INDEX[hand_class(hole)]
        acts = st.history[0]
        kinds = [(a.player, a.kind) for a in acts]
        jam = lambda a: a.to >= JAM_FRACTION * st.stack - EPS
        n = self.pre["nodes"]
        opts = {}
        if hero == "btn" and not acts:
            node, pol = "btn_root", n["btn_root"]
            opts = {"fold": ("fold", 0), "limp": ("call", 1.0), "raise": ("raise", self.pre["open"]),
                    "all-in": ("raise", st.stack)}
        elif hero == "bb" and kinds == [("btn", "call")]:
            node, pol = "bb_vs_limp", n["bb_vs_limp"]
            opts = {"check": ("check", 0), "raise": ("raise", self.pre["iso_to"]), "all-in": ("raise", st.stack)}
        elif hero == "bb" and kinds == [("btn", "raise")]:
            s = acts[0].to
            if jam(acts[0]):
                node, pol = "bb_vs_open_jam", n["bb_vs_open_jam"]
            else:
                key = min(self.pre["vs_open"], key=lambda k: abs(math.log(float(k) / s)))
                spot = self.pre["vs_open"][key]
                node, pol = f"bb_vs_raise (open {key})", spot["bb_vs_raise"]
                opts = {"3-bet": ("raise", round(spot["threebet_to"] * s / float(key), 2)),
                        "all-in": ("raise", st.stack)}
            opts.update({"fold": ("fold", 0), "call": ("call", min(s, st.stack))})
        elif hero == "btn" and kinds == [("btn", "call"), ("bb", "raise")]:
            r = acts[1].to
            if jam(acts[1]):
                node, pol = "btn_vs_limp_jam", n["btn_vs_limp_jam"]
            else:
                node, pol = "btn_vs_iso", n["btn_vs_iso"]
                opts = {"all-in": ("raise", st.stack)}
            opts.update({"fold": ("fold", 0), "call": ("call", min(r, st.stack))})
        elif hero == "btn" and kinds == [("btn", "raise"), ("bb", "raise")]:
            t = acts[1].to
            if jam(acts[1]):
                node, pol = "btn_vs_3bet (all-in)", n["btn_vs_3bet"]
            else:
                key = min(self.pre["vs_3bet"], key=lambda k: abs(math.log(float(k) / t)))
                node, pol = f"btn_vs_3bet_small (3-bet {key})", self.pre["vs_3bet"][key]
                opts = {"all-in": ("raise", st.stack)}
            opts.update({"fold": ("fold", 0), "call": ("call", min(t, st.stack))})
        else:
            # Facing an all-in (or a raise we treat as one) deeper in the tree: call / fold from the
            # matching node.
            last = acts[-1]
            name = {"bb": "bb_vs_4bet" if kinds[0] == ("btn", "raise") else "bb_vs_limp_reraise",
                    "btn": "btn_vs_3bet" if kinds[0] == ("btn", "raise") else "btn_vs_limp_jam"}[hero]
            node, pol = name, n[name]
            opts = {"fold": ("fold", 0), "call": ("call", min(last.to, st.stack))}
        probs = {a: float(v[i]) for a, v in pol.items() if a in opts}
        self._options = opts
        return Decision("", 0, _normalise(probs), node, "preflop")

    # -- postflop ----------------------------------------------------------------------------

    def pot_type(self, st):
        """Which trained postflop pot this hand is (and the preflop raise size it was trained with)."""
        pre = st.history[0]
        kinds = [(a.player, a.kind) for a in pre]
        if kinds[:1] == [("btn", "call")]:
            return "iso" if any(k == "raise" for _, k in kinds) else "limp"
        raises = [a for a in pre if a.kind == "raise"]
        if len(raises) == 1:
            s = raises[0].to
            alt = float(self._pot("srp_alt")["pot_to"])
            return "srp" if abs(math.log(s / self.pre["open"])) <= abs(math.log(s / alt)) else "srp_alt"
        return "3bet"

    def _pot(self, name):
        if name not in self._pots:
            self._pots[name] = dict(np.load(self.policy_dir / f"{name}.npz"))
        return self._pots[name]

    def opts(self, pot):
        """The sizes a pot's policy was trained with (stored in the file since experiment 30)."""
        if pot not in self._opts:
            d = self._pot(pot)
            self._opts[pot] = _opts_from_json(str(d["opts_json"])) if "opts_json" in d else _match_opts()
        return self._opts[pot]

    def _postflop(self, hero, hole, board, st):
        s = st.street - 1                                   # 0 flop, 1 turn, 2 river
        acts = st.history[st.street]
        P, behind = st.street_start[st.street]
        b_facing = st.facing(hero)
        hero_left = st.left(hero)
        other = "bb" if hero == "btn" else "btn"
        opp_all_in = st.left(other) <= EPS
        pot = self.pot_type(st)
        o = self.opts(pot)
        leads, bets = fg.street_sizes(s, o)
        raises = fg.raise_sizes(o, s)

        # Hand group (the river keeps "missed draw" from the turn) and board texture.
        prev = None
        for n in range(3, len(board) + 1):
            prev = fg.fine_bucket(hole, board[:n], prev)
        bkt = prev
        tex = (flop_texture(board[:3]) if s == 0 else turn_texture(board[:3], board[3]) if s == 1
               else river_texture(board[:4], board[4]))
        line = 0 if s == 0 else fg.line_index(_aggressor(st.history[st.street - 1]), P, behind)

        aggressive = [a for a in acts if a.kind in ("bet", "raise")]
        seq = "".join("x" if a.kind == "check" else "b" if a.kind in ("bet", "raise") else "c" for a in acts)
        bet_to = lambda frac: min(round(frac * P, 2), behind)
        raise_to = lambda b: min(round(fg.raise_to(o, raises[0], b, P), 2), st.inp[hero] + hero_left) \
            if raises else None
        # Nodes of the whole-hand model; cells(k) is the table cell when facing trained size k.
        weights = [(None, 1.0)]
        if seq == "":
            kind, cell = "bb_first", (tex, line, bkt)
            labels = ["check"] + [f"lead {f:.2g}" for f in leads]
            opts = {"check": ("check", 0)} | {f"lead {f:.2g}": ("bet", bet_to(f)) for f in leads}
        elif seq == "x":
            kind, cell = "btn_vs_check", (tex, line, bkt)
            labels = ["check"] + [f"bet {f:.2g}" for f in bets]
            opts = {"check": ("check", 0)} | {f"bet {f:.2g}": ("bet", bet_to(f)) for f in bets}
        elif seq in ("xb", "b"):
            b = aggressive[0].to
            sizes = bets if seq == "xb" else leads
            weights = size_weights(b / P, sizes, self.translate)
            kind = "bb_vs_bet" if seq == "xb" else "btn_vs_lead"
            cell = lambda k: (tex, line, bkt, k)
            labels = ["fold", "call"] + [f"raise {r:.2g}" for r in raises]
            opts = {"fold": ("fold", 0), "call": ("call", min(b, st.inp[hero] + hero_left))}
            opts |= {f"raise {r:.2g}": ("raise", raise_to(b)) for r in raises}
        elif seq in ("xbb", "bb"):
            b0, R = aggressive[0].to, aggressive[1].to
            sizes = bets if seq == "xbb" else leads
            weights = size_weights(b0 / P, sizes, self.translate)
            kind = "btn_vs_xr" if seq == "xbb" else "bb_vs_raise"
            cell = lambda k: (tex, line, bkt, k, 0)
            labels = ["fold", "call"]
            opts = {"fold": ("fold", 0), "call": ("call", min(R, st.inp[hero] + hero_left))}
        else:
            return self._rule(hero, bkt, st, acts)

        cells = [(cell(k) if callable(cell) else cell, w) for k, w in weights]
        source, node = "model", f"{fg.STREETS[s]} {kind}"
        probs = None
        notes = []
        if weights[0][0] is not None:
            sizes = bets if seq.startswith("x") else leads
            faced = aggressive[0].to / P
            notes.append(f"facing {faced:.2f} pot -> " + " / ".join(f"{sizes[k]:.2f} ({w:.0%})" for k, w in weights))
        if s == 0 and pot == "srp" and self.use_solver_flops and abs(st.history[0][0].to - self.solved_open) < 0.26 \
                and kind in ("bb_first", "btn_vs_check", "bb_vs_bet"):
            parts = [(self._solver_flop(hole, board, kind, labels, c, bets), w) for c, w in cells]
            if all(p is not None for (p, _), _ in parts):
                probs = _blend([(p, w) for (p, _), w in parts])
                source = "solver"
                notes.append(parts[0][0][1])
        if probs is None:
            table = self._pot(pot)[f"sigma_{s}_{kind}"]
            probs = _blend([(dict(zip(labels, map(float, table[c]))), w) for c, w in cells])
            notes.append(f"pot {pot}, tex {tex}, line {line}, group {fg.BUCKETS[bkt]}")

        # Can't raise an all-in player; a raise that needs the whole stack is an all-in.
        if opp_all_in or hero_left <= b_facing + EPS:
            for lab in [l for l in probs if l.startswith("raise")]:
                probs["call"] = probs.get("call", 0) + probs.pop(lab)
                opts.pop(lab, None)
        self._options = opts
        return Decision("", 0, _normalise(probs), node, source, notes=notes)

    def _rule(self, hero, bkt, st, acts):
        """Facing a re-raise: jam the nuts, call strong made hands, fold the rest."""
        b = st.inp["bb" if hero == "btn" else "btn"]
        all_in = st.inp[hero] + st.left(hero)
        opts = {"fold": ("fold", 0), "call": ("call", min(b, all_in))}
        if bkt == 0 and all_in > b + EPS:
            opts["raise all-in"] = ("raise", all_in)
            probs = {"raise all-in": 1.0}
        elif bkt in (0, 1, 2):
            probs = {"call": 1.0}
        else:
            probs = {"fold": 1.0}
        self._options = opts
        return Decision("", 0, probs, f"{fg.STREETS[st.street - 1]} re-raise", "rule",
                        notes=[f"group {fg.BUCKETS[bkt]}"])

    # -- experiment 28's solved flops ------------------------------------------------------------

    def _flop(self, t):
        if t not in self._flops:
            meta = self.index.get(str(t))
            if meta is None:
                self._flops[t] = None
            else:
                d = json.load(open(self.flops_dir / meta["file"]))
                nodes = {}
                for name, rows in d["nodes"].items():
                    by = {}
                    for r in rows:
                        key = (frozenset((r["combo"][:2], r["combo"][2:])), r.get("facing"))
                        by[key] = {a: v for a, v in r.items() if a not in ("combo", "facing")}
                    nodes[name] = by
                self._flops[t] = (meta["flop"], nodes)
        return self._flops[t]

    def _solver_flop(self, hole, flop, kind, labels, cell, bets):
        t = flop_texture(flop)
        got = self._flop(t)
        if got is None:
            return None, "no solved flop for this class"
        rep_str, nodes = got
        rep = parse_cards(rep_str)
        if kind not in nodes:
            return None, "node not in the solved flop"
        facing = None
        if kind == "bb_vs_bet":
            # The solve's own label for this size: the nearest bet amount among those it faced.
            want = bets[cell[3]] * 2 * self.solved_open
            labels_faced = {k[1] for k in nodes[kind] if k[1]}
            facing = min(labels_faced, key=lambda f: abs(float(f.split()[1].rstrip("b")) - want)) if labels_faced else None
        table = {k: v for k, v in nodes[kind].items() if k[1] == facing}
        bkt = cell[2]
        mapped = map_combo(hole, flop, rep)
        note = f"solved flop {rep_str}"
        row = None
        if mapped is not None and fg.fine_bucket(mapped, rep) == bkt:
            row = table.get((frozenset(card_str(c) for c in mapped), facing))
            if row is not None:
                note += f", as {''.join(card_str(c) for c in mapped)}"
        if row is None:                                     # average over the solver's combos in this group
            rows = [v for (combo, _), v in table.items()
                    if fg.fine_bucket(parse_cards("".join(sorted(combo))), rep) == bkt]
            if not rows:
                return None, note + ": no combos in this hand group"
            row = {a: float(np.mean([r.get(a, 0.0) for r in rows])) for a in rows[0]}
            note += f", average of {len(rows)} combos ({fg.BUCKETS[bkt]})"
        return _solver_labels(kind, row, labels), note


def _solver_labels(kind, row, labels):
    """Solver action names ('bet 2bb', 'raise 7bb') -> the engine's labels, in the same order."""
    out = dict.fromkeys(labels, 0.0)
    if kind == "bb_vs_bet":
        for a, v in row.items():
            lab = "fold" if a == "fold" else "call" if a == "call" else labels[2]
            out[lab] += v
    else:                                                   # check / bet sizes in increasing order
        sized = sorted((a for a in row if a != "check"), key=lambda a: float(a.split()[1].rstrip("b")))
        out["check"] += row.get("check", 0.0)
        for a, lab in zip(sized, labels[1:]):
            out[lab] += row[a]
    return out


def _blend(parts):
    """Weighted average of several {label: prob} dicts with the same labels."""
    out = dict.fromkeys(parts[0][0], 0.0)
    for p, w in parts:
        for k, v in p.items():
            out[k] += w * v
    return out


def _opts_from_json(text):
    o = json.loads(text)
    o["sizes"] = tuple({k: tuple(v) for k, v in st.items()} for st in o["sizes"])
    for k in ("raise_pot", "raises", "raise_streets"):
        if k in o:
            o[k] = tuple(o[k])
    return o


def _normalise(p):
    tot = sum(p.values())
    if tot <= 0:
        return {k: 1 / len(p) for k in p}
    return {k: v / tot for k, v in p.items()}


def _match_opts():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from open_size_study import OPTS_MATCH
    return OPTS_MATCH


# ----------------------------------------------------------------------------- combo mapping

def map_combo(hole, flop, rep):
    """Map a hole hand on `flop` to the equivalent hand on the representative flop `rep` (same class).

    Suits: the board's suit groups (largest first, then by highest card) are matched to rep's; the other
    suits keep a fixed order. Ranks: board ranks map to rep's board ranks, and other ranks keep their
    distance below the next board rank up (or below the ace), clipped into the gap. Returns None if the
    two hole cards collide or land on a board card.
    """
    suit_map = _suit_map(flop, rep)
    fr = sorted({c.rank for c in flop}, reverse=True)
    rr = sorted({c.rank for c in rep}, reverse=True)
    if len(fr) != len(rr):
        return None
    upper_f, upper_r = [13] + fr, [13] + rr                # the gap above board rank k is (fr[k], upper[k])
    lower_f, lower_r = fr + [-1], rr + [-1]

    def rank(r):
        if r in fr:
            return rr[fr.index(r)]
        k = next(k for k in range(len(fr) + 1) if lower_f[k] < r < upper_f[k])
        if upper_r[k] - lower_r[k] < 2:                    # no room between rep's board ranks
            return None
        return max(upper_r[k] - (upper_f[k] - r), lower_r[k] + 1)

    ranks = [rank(c.rank) for c in hole]
    if None in ranks:
        return None
    if hole[0].rank != hole[1].rank and ranks[0] == ranks[1]:
        return None
    out = [eval7.Card(RANK_CHARS[r] + SUIT_CHARS[suit_map[c.suit]]) for r, c in zip(ranks, hole)]
    if len({card_str(c) for c in out} | {card_str(c) for c in rep}) != 5:
        return None
    return out


def _suit_map(flop, rep):
    def groups(b):
        g = {}
        for c in b:
            g.setdefault(c.suit, []).append(c.rank)
        return sorted(g, key=lambda su: (-len(g[su]), -max(g[su])))
    gf, gr = groups(flop), groups(rep)
    m = dict(zip(gf, gr))
    rest_f = [su for su in range(4) if su not in m]
    rest_r = [su for su in range(4) if su not in m.values()]
    m.update(zip(rest_f, rest_r))
    return m
