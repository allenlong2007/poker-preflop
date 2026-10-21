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

from bot import potodds
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
# Experiment 35: the whole-hand model's 12 hand groups make the BB fold too much on the turn and river
# (bluff-catchers share a group with air). Against TexasSolver on the same spots (scripts/bb_turn_compare.py)
# the BB folded ~10-15 points too often to turn bets and ~5-7 points to river bets, and led the river too
# rarely after the turn checked through. These corrections move part of the fold (or check) to call (or lead),
# only for hands with a pair or better (a live draw on the turn, overcards against small bets); air folds.
DEFEND_KEEP = {1: 0.7, 2: 0.85}            # street -> share of a model fold kept (turn, river), for those hands
FALLBACK_UNTRAINED = True
STRICT_PRIOR = 0.0                        # experiment 42: strictness before any data on the opponent (trained)
RIVER_SHIFT = (0.6, 0.3)                  # experiment 40: (1/3 -> 3/4, 1/2 -> pot) share of the model's river bets
SHORT_BB, SHORT_SPR = 12.0, 1.2           # experiment 40: postflop short-stack mode thresholds (behind, at street start)
SHORT_JAM = 0.55                          # equity vs the opponent's range needed to put the stack in unprompted
RIVER_LEAD = 0.5                           # extra share of a model river check turned into a lead (value hands)
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
    def __init__(self, policy_dir=None, flops_dir=FLOPS, use_solver_flops=True, translate=True, seed=None,
                 defend_keep=None, river_lead=None, pot_odds=False):
        """policy_dir: postflop policies (default: experiment 30's finer sizes when all are trained, else
        experiment 26's). translate: blend neighbouring sizes when facing an untrained bet size.
        defend_keep / river_lead: experiment 35's BB corrections (default DEFEND_KEEP / RIVER_LEAD)."""
        self.defend_keep = dict(DEFEND_KEEP if defend_keep is None else defend_keep)
        self.river_shift = RIVER_SHIFT
        self.adapted, self.lam, self.exploit = {}, 0.0, {"bluff": 1.0, "defend": 1.0}
        self.pot_odds = pot_odds              # experiment 39: equity vs the opponent's range against the price
        self.opp_profile = None               # the opponent's profile (bot/opponent.py), set by the adapter
        self.size_value = None                # experiment 39: bluff-share scale by the opponent's bet size
        self.bluff_model = None               # experiment 41: learned P(bluff) for the opponent's bets
        self.strict = STRICT_PRIOR            # experiment 42: trust in value-heavy priors, set from opponent data
        self.river_lead = RIVER_LEAD if river_lead is None else river_lead
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
            if self._short(st, hero):
                d = self._short_stack(d, hero, hole, board[:2 + st.street], st)
            elif self.pot_odds and st.facing(hero) > EPS:
                if "fold" in self._options and "call" in self._options:
                    probs = {k: d.probs.get(k, 0.0) for k in set(d.probs) | {"fold", "call"}}
                    d.probs = _normalise(potodds.check(probs, hole, board[:2 + st.street], st, hero, self.pre,
                                                       self.opp_profile, self.exploit["defend"], d.notes,
                                                       self.size_value, self._bluffs(), self.strict))
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
        # Experiment 36: shorter stacks use the tree solved at the nearest depth (the all-in 4-bet only
        # exists at 40bb or shorter); the per-size answers (vs_open, vs_3bet) are 100bb solves.
        depths = {float(k): v for k, v in self.pre.get("by_depth", {}).items()}
        depth = min([STACK] + list(depths), key=lambda d: abs(math.log(d / max(st.stack, 1.0))))
        deep = depth == STACK
        # A node the depth tree doesn't have (the 40 / 60bb trees have no shoves, so no answers to one either)
        # falls back to the 100bb tree's answer.
        n = _Fallback(self.pre["nodes"] if deep else depths[depth], self.pre["nodes"],
                      *[depths[d] for d in sorted(depths, reverse=True)])
        f4 = self.pre.get("fourbet_x")
        opts = {}
        raise_ = lambda to: ("raise", min(round(to, 2), st.stack))
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
                node, pol = f"bb_vs_raise (open {key})", spot["bb_vs_raise"] if deep else n["bb_vs_raise"]
                opts = {"3-bet": raise_(spot["threebet_to"] * s / float(key)), "all-in": ("raise", st.stack)}
            opts.update({"fold": ("fold", 0), "call": ("call", min(s, st.stack))})
        elif hero == "btn" and kinds == [("btn", "call"), ("bb", "raise")]:
            r = acts[1].to
            if jam(acts[1]):
                node, pol = "btn_vs_limp_jam", n["btn_vs_limp_jam"]
            else:
                node, pol = "btn_vs_iso", n["btn_vs_iso"]
                opts = {"all-in": ("raise", st.stack)}
                if f4:
                    opts["re-raise"] = raise_(f4 * r)
            opts.update({"fold": ("fold", 0), "call": ("call", min(r, st.stack))})
        elif hero == "btn" and kinds == [("btn", "raise"), ("bb", "raise")]:
            t = acts[1].to
            if jam(acts[1]):
                node, pol = "btn_vs_3bet (all-in)", n["btn_vs_3bet"]
            else:
                key = min(self.pre["vs_3bet"], key=lambda k: abs(math.log(float(k) / t)))
                node, pol = f"btn_vs_3bet_small (3-bet {key})", self.pre["vs_3bet"][key] if deep \
                    else n["btn_vs_3bet_small"]
                opts = {"all-in": ("raise", st.stack)}
                if f4:
                    opts["4-bet"] = raise_(f4 * t)
            opts.update({"fold": ("fold", 0), "call": ("call", min(t, st.stack))})
        elif hero == "bb" and len(acts) == 3 and kinds[1:] == [("bb", "raise"), ("btn", "raise")] \
                and not jam(acts[2]) and kinds[0][0] == "btn":
            # A non-all-in 4-bet (or a re-raise of our raise over a limp): fold / call / jam.
            name = "bb_vs_4bet_small" if kinds[0] == ("btn", "raise") else "bb_vs_limp_3bet"
            node = name
            pol = n.get(name) or self.pre["nodes"].get(name)
            opts = {"fold": ("fold", 0), "call": ("call", min(acts[2].to, st.stack)), "all-in": ("raise", st.stack)}
        else:
            # Facing an all-in (or a raise we treat as one) deeper in the tree: call / fold from the
            # matching node.
            last = acts[-1]
            first = kinds[0] == ("btn", "raise")
            if hero == "btn" and len(acts) >= 4:
                name = "btn_vs_5bet" if first else "btn_vs_limp_4bet"
            elif hero == "bb":
                name = "bb_vs_4bet" if first else "bb_vs_limp_reraise"
            else:
                name = "btn_vs_3bet" if first else "btn_vs_limp_jam"
            node = name
            pol = n.get(name) or self.pre["nodes"].get(name) or n["bb_vs_4bet" if hero == "bb" else "btn_vs_3bet"]
            opts = {"fold": ("fold", 0), "call": ("call", min(last.to, st.stack))}
        probs = {a: float(v[i]) for a, v in pol.items() if a in opts}
        base = node.split(" ")[0]
        if deep and self.lam > 0 and base in self.adapted:
            ad = self.adapted[base]
            probs = {a: (1 - self.lam) * p + self.lam * float(ad[a][i]) if a in ad else (1 - self.lam) * p
                     for a, p in probs.items()}
            node += f" (adapted {self.lam:.0%})"
        if not deep:
            node += f" ({depth:g}bb tree)"
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
            row = lambda c: dict(zip(labels, map(float, table[c])))
            # Experiment 40: a cell the training never reached is uniform (every action alike); then use the same
            # spot's deep-stack cell (line with stack-to-pot >= 4) instead of playing at random.
            flat = lambda d: len(d) > 2 and max(d.values()) - min(d.values()) < 1e-6
            if FALLBACK_UNTRAINED and line % 3 and any(flat(row(c)) for c, _ in cells):
                deep = line - line % 3
                cells = [((c[0], deep) + tuple(c[2:]), w) for c, w in cells]
                notes.append(f"untrained cell: line {line} -> {deep}")
                line = deep
            probs = _blend([(row(c), w) for c, w in cells])
            notes.append(f"pot {pot}, tex {tex}, line {line}, group {fg.BUCKETS[bkt]}")
            probs = self._oop_correction(probs, s, kind, bkt, st, notes, aggressive[0].to / P if aggressive else 0)
            if kind == "bb_first" and s == 1 and pot in ("srp", "srp_alt"):
                probs = _small_leads(probs, notes)
            if s == 2 and kind in ("btn_vs_check", "bb_first"):
                probs = _polar_river(probs, notes, self.river_shift)

        probs = self._exploit(probs, kind, bkt, notes, aggressive[0].to / P if aggressive else 0)
        # Can't raise an all-in player; a raise that needs the whole stack is an all-in.
        if opp_all_in or hero_left <= b_facing + EPS:
            for lab in [l for l in probs if l.startswith("raise")]:
                probs["call"] = probs.get("call", 0) + probs.pop(lab)
                opts.pop(lab, None)
        self._options = opts
        return Decision("", 0, _normalise(probs), node, source, notes=notes)

    # -- experiment 40: short stacks postflop -------------------------------------------------------

    def _bluffs(self):
        """The opponent's learned bluff model, or the self-play prior before there is one."""
        if self.bluff_model is None:
            from bot import bluffmodel
            self.bluff_model = bluffmodel.prior() or False
        return self.bluff_model or None

    def _short(self, st, hero):
        """Stacks too short for small-ball postflop play: at most SHORT_SPR pots behind, or SHORT_BB big blinds."""
        P, behind = st.street_start[st.street]
        return behind <= SHORT_BB or behind <= SHORT_SPR * P

    def _short_stack(self, d, hero, hole, board, st):
        """Get it in with equity, otherwise check / fold; call a bet by its price. Replaces small c-bets and thin
        value at short stacks, where any bet commits a large share of the stack anyway."""
        rng = potodds.opponent_range(self.pre, st, hero, board, hole, self.opp_profile, self.exploit["defend"],
                                     self.size_value, self._bluffs(), self.strict)
        eq = potodds.equity(list(hole), list(board), rng)
        b = st.facing(hero)
        all_in = st.inp[hero] + st.left(hero)
        other = "bb" if hero == "btn" else "btn"
        can_raise = st.left(other) > EPS and st.left(hero) > b + EPS
        opts = {"fold": ("fold", 0), "check": ("check", 0), "call": ("call", min(st.inp[other], all_in)),
                "all-in": ("raise" if b > EPS else "bet", all_in)}
        if b > EPS:
            need = b / (st.pot() + b)
            probs = ({"all-in": 1.0} if can_raise and eq >= SHORT_JAM else {"call": 1.0}) if eq >= need else {"fold": 1.0}
            why = f"equity {eq:.0%} vs price {need:.0%}"
        else:
            probs = {"all-in": 1.0} if eq >= SHORT_JAM else {"check": 1.0}
            why = f"equity {eq:.0%} vs jam threshold {SHORT_JAM:.0%}"
        self._options = opts
        return Decision("", 0, probs, d.node + " (short stack)", "short",
                        notes=d.notes + [f"short stack ({st.left(hero):.1f}bb behind): {why}"])

    def set_adaptation(self, nodes, lam, exploit):
        """Experiment 37 (bot/adapt.py): best-response preflop nodes blended in with weight lam, and the
        postflop bluff / defend scales against the current opponent."""
        self.adapted, self.lam = dict(nodes), float(lam)
        self.exploit = {"bluff": float(exploit.get("bluff", 1.0)), "defend": float(exploit.get("defend", 1.0))}

    def _exploit(self, probs, kind, bkt, notes, faced=0.0):
        """Scale bluffs by how often the opponent folds, and calling down by how often they bet."""
        b, d = self.exploit["bluff"], self.exploit["defend"]
        if kind in ("btn_vs_check", "bb_first") and bkt >= 9 and abs(b - 1) > 0.02:
            bets = [k for k in probs if k.startswith(("bet", "lead"))]
            tot = sum(probs[k] for k in bets)
            if 0 < tot < 1:
                new = min(0.95, tot * b)
                probs = {k: (v * new / tot if k in bets else v * (1 - new) / (1 - tot)) for k, v in probs.items()}
                notes.append(f"opponent folds x{b:.2f}: bluff {tot:.0%} -> {new:.0%}")
        elif kind in ("bb_vs_bet", "btn_vs_lead", "btn_vs_xr", "bb_vs_raise") and abs(d - 1) > 0.02:
            made = bkt <= 7 or bkt in (8, 9) or (bkt == 10 and faced <= 0.6)
            f, c = probs.get("fold", 0), probs.get("call", 0)
            if d > 1 and made and f > 0:                           # aggressive: fold less
                moved = f * (1 - 1 / d)
                probs = dict(probs, fold=f - moved, call=c + moved)
                notes.append(f"opponent bets x{d:.2f}: {moved:.0%} fold -> call")
            elif d < 1 and bkt >= 5 and c > 0:                     # passive: fold marginal hands more
                moved = c * (1 - d)
                probs = dict(probs, fold=f + moved, call=c - moved)
                notes.append(f"opponent bets x{d:.2f}: {moved:.0%} call -> fold")
        return probs

    def _oop_correction(self, probs, s, kind, bkt, st, notes, faced=0.0):
        """Experiment 35's BB corrections to the model (see DEFEND_KEEP / RIVER_LEAD)."""
        # A pair or better; on the turn also a draw; against small bets (<= 0.6 pot) also overcards.
        made = bkt <= 7 or (s == 1 and bkt <= 9) or (bkt == 10 and faced <= 0.6)
        if kind == "bb_vs_bet" and s in DEFEND_KEEP and made and probs.get("fold", 0) > 0:
            moved = probs["fold"] * (1 - self.defend_keep[s])
            probs = dict(probs, fold=probs["fold"] - moved, call=probs.get("call", 0) + moved)
            notes.append(f"defense correction: {moved:.0%} fold -> call")
        elif kind == "bb_first" and s == 2 and bkt <= 6 and self.river_lead > 0 \
                and [a.kind for a in st.history[2]] == ["check", "check"]:
            leads = [k for k in probs if k.startswith("lead")]
            moved = probs.get("check", 0) * self.river_lead
            if leads and moved > 0:
                tot = sum(probs[k] for k in leads)
                probs = dict(probs, check=probs["check"] - moved)
                for k in leads:
                    probs[k] += moved * (probs[k] / tot if tot > 0 else 1 / len(leads))
                notes.append(f"river lead correction: {moved:.0%} check -> lead")
        return probs

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


class _Fallback(dict):
    """A depth tree's nodes; a node it lacks comes from the 100bb tree, else from the deepest shorter tree that has
    it (at 100bb there is no 4-bet shove, so no node for facing one; the 25bb tree has it)."""
    def __init__(self, nodes, *others):
        super().__init__(nodes)
        self.others = others

    def __missing__(self, key):
        for o in self.others:
            if key in o:
                return o[key]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _polar_river(probs, notes, shift):
    """Experiment 40: the model bets the river smaller than TexasSolver (1/3 and 1/2 pot 34% of the range vs 23%;
    3/4 to pot 15% vs 34%). shift = (share of 1/3-pot bets moved to 3/4, share of 1/2-pot moved to pot)."""
    a, b = shift
    keys = {round(float(k.split()[1]), 2): k for k in probs if k.startswith(("bet", "lead"))}
    moves = [(0.33, 0.75, a), (0.5, 1.0, b)]
    out = dict(probs)
    moved = 0.0
    for src, dst, share in moves:
        if src in keys and dst in keys and share > 0:
            x = out[keys[src]] * share
            out[keys[src]] -= x
            out[keys[dst]] += x
            moved += x
    if moved > 0.005:
        notes.append(f"river sizing toward the solver: {moved:.0%} moved to bigger bets")
    return out


def _small_leads(probs, notes):
    """Experiment 38: when the preflop caller leads the turn, TexasSolver only ever leads 1/3 pot (12-17% of
    its range on blank turns, never bigger); the model's bigger turn leads (~4% of the range) move to the
    smallest lead."""
    leads = sorted((k for k in probs if k.startswith("lead")), key=lambda k: float(k.split()[1]))
    big = sum(probs[k] for k in leads[1:])
    if len(leads) > 1 and big > 0:
        probs = dict(probs, **{leads[0]: probs[leads[0]] + big}, **{k: 0.0 for k in leads[1:]})
        notes.append(f"turn lead sized down: {big:.0%} to {leads[0]}")
    return probs


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
