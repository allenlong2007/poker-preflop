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
import sqlite3
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
COVERAGE_PATH = ROOT / "output/pokernow/coverage.sqlite"
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
MONSTER_TILT = 0.75                       # experiment 43: share of a nut-hand decision moved to max value / trap
STRICT_PRIOR = 0.0                        # experiment 42: strictness before any data on the opponent (trained)
RIVER_SHIFT = (0.6, 0.3)                  # experiment 40: (1/3 -> 3/4, 1/2 -> pot) share of the model's river bets
SHORT_BB, SHORT_SPR = 12.0, 1.2           # experiment 40: postflop short-stack mode thresholds (behind, at street start)
SHORT_JAM = 0.55                          # equity vs the opponent's range needed to put the stack in unprompted
# Experiment 45 (scripts/trap_study.py, TexasSolver on 32 flops): how often strong hands slow-play, by board
# texture and class: (flop: just call a bet instead of raising, turn: BB checks instead of leading, turn: BTN checks
# back). None: no solver data there (TRAP_CLASS, the class average, fills in). The flop BTN never checks them back.
TRAP = {
    ("dry", "set"): (0.08, 0.13, 0.0), ("dry", "two pair"): (0.5, 0.44, 0.2),
    ("monotone", "flush"): (0.03, 0.1, 0.1), ("monotone", "full house+"): (None, 0.12, 0.0),
    ("monotone", "set"): (0.0, 0.04, 0.02), ("monotone", "straight"): (0.74, 0.11, 0.16),
    ("monotone", "two pair"): (0.63, 0.39, 0.18), ("paired", "flush"): (None, 0.56, 0.0),
    ("paired", "full house+"): (0.11, 0.45, 0.22), ("paired", "straight"): (None, 0.56, 0.0),
    ("paired", "trips"): (0.7, 0.2, 0.0), ("wet", "set"): (0.02, 0.19, 0.0),
    ("wet", "straight"): (0.23, 0.13, 0.12), ("wet", "two pair"): (0.51, 0.49, 0.22),
}
TRAP_CLASS = {"flush": (0.03, 0.15, 0.1), "full house+": (0.11, 0.44, 0.22), "set": (0.04, 0.12, 0.01),
              "straight": (0.4, 0.14, 0.13), "trips": (0.7, 0.2, 0.0), "two pair": (0.55, 0.45, 0.2)}
TRAP_BET_PRIOR = 0.45      # a typical bet rate when checked to; more than this traps more, less traps less
TRAP_NUTS_AGGRESSION = 1.3  # the nuts only check (to raise) against an aggression read at least this high
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
    shove_read: bool = False               # experiment 46: decided by this opponent's observed shoves
    diagnostics: dict = field(default_factory=dict)


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
                 defend_keep=None, river_lead=None, pot_odds=False, exact_policy_path=None,
                 exact_candidate=False, exact_min_updates=1000, exact_required=True, coverage_path=True):
        """policy_dir: postflop policies (default: experiment 30's finer sizes when all are trained, else
        experiment 26's). translate: blend neighbouring sizes when facing an untrained bet size.
        defend_keep / river_lead: experiment 35's BB corrections (default DEFEND_KEEP / RIVER_LEAD)."""
        self.exact_required = exact_required
        self.coverage_path = COVERAGE_PATH if coverage_path is True and exact_required else (
            Path(coverage_path) if coverage_path not in (None, False, True) else None)
        self.coverage_counts = {}
        self.defend_keep = dict(DEFEND_KEEP if defend_keep is None else defend_keep)
        self.river_shift = RIVER_SHIFT
        self.adapted, self.lam, self.exploit = {}, 0.0, {"bluff": 1.0, "defend": 1.0}
        self.pot_odds = pot_odds              # experiment 39: equity vs the opponent's range against the price
        self.read_evidence = {}
        self.opp_profile = None               # the opponent's profile (bot/opponent.py), set by the adapter
        self.size_value = None                # experiment 39: bluff-share scale by the opponent's bet size
        self.bluff_model = None               # experiment 41: learned P(bluff) for the opponent's bets
        self.strict = STRICT_PRIOR            # experiment 42: trust in value-heavy priors, set from opponent data
        self.river_lead = RIVER_LEAD if river_lead is None else river_lead
        self.river_jam = True                 # experiment 47 (switchable for comparisons)
        if policy_dir is None:
            policy_dir = POLICY_FINE if all((POLICY_FINE / f"{p}.npz").exists() for p in POTS) else POLICY
        self.pre = {} if exact_required else json.load(open(POLICY / "preflop.json"))
        self.policy_dir, self.flops_dir = Path(policy_dir), Path(flops_dir)
        self.use_solver_flops = not exact_required and use_solver_flops and (Path(flops_dir) / "index.json").exists()
        meta = json.load(open(Path(flops_dir) / "index.json")) if self.use_solver_flops else {}
        self.index = meta.get("classes", {})
        self.solved_open = float(meta.get("open", SOLVED_OPEN))
        self.translate = translate
        self._pots, self._flops, self._opts = {}, {}, {}
        self.rng = np.random.default_rng(seed)
        self.exact_policy = None
        self.exact_status = {"status": "disabled"}
        self.exact_path = None
        if exact_policy_path is not False:
            self.exact_path = Path(exact_policy_path) if exact_policy_path is not None else ROOT / "output/exact/active.sqlite"
            self.exact_status = {"status": "missing_artifact"}
            if self.exact_path.exists():
                from preflop.exacttrain import ExactPolicy
                try:
                    self.exact_policy = ExactPolicy(self.exact_path, min_updates=exact_min_updates,
                                                   candidate=exact_candidate)
                except (OSError, ValueError, sqlite3.DatabaseError) as error:
                    self.exact_status = {"status": "load_error", "reason": str(error)}

    # -- public ------------------------------------------------------------------------------

    def warm(self):
        """Load every policy and solved flop up front, so no decision during a game waits on disk."""
        if self.exact_required:
            return self
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
        exact = dict(self.exact_status)
        if self.exact_policy is not None:
            exact = self.exact_policy.lookup({
                "hero_position": hero, "hero_hole": "".join(map(card_str, hole)),
                "board": "".join(map(card_str, board[:2 + st.street])) if st.street else "",
                "effective_stack_bb": stack,
                "history": [[[a.player, a.kind, a.to] for a in street] for street in st.history]})
        context = {"hero_position": hero, "hero_hole": "".join(map(card_str, hole)),
                   "board": "".join(map(card_str, board[:2 + st.street])) if st.street else "",
                   "effective_stack_bb": stack,
                   "history": [[[a.player, a.kind, a.to] for a in street] for street in st.history]}
        if exact["status"] != "ready" and self.exact_required:
            from bot.coverage import ExactCoverageError
            diagnostics = {"exact": exact, "context": context, "fallbacks": [f"exact:{exact['status']}"],
                           "strategy_mode": "exact_required", "coverage": self._record_coverage(
                               context, exact["status"], "withheld", [f"exact:{exact['status']}"])}
            raise ExactCoverageError(diagnostics)
        if exact["status"] == "ready":
            self._options = exact.pop("options")
            probs = exact.pop("probs")
            d = Decision("", 0, probs, f"exact street {st.street}", "exact",
                         notes=[f"exact-card average policy: {exact['updates']} updates"],
                         diagnostics={"base": {"source": "exact", "probs": dict(probs)}})
        elif st.street == 0:
            d = self._preflop(hero, hole, st)
            d.diagnostics.setdefault("base", {"source": d.source, "node": d.node, "probs": dict(d.probs)})
            d = _adjust(d, "vs_shove", self._vs_shove, hero, hole, st)
            d = _adjust(d, "small_raise", self._small_raise, hero, hole, st)
            d = _adjust(d, "reraise_war", self._vs_reraise_war, hero, hole, st)
        else:
            d = self._postflop(hero, hole, board[:2 + st.street], st)
            d.diagnostics.setdefault("base", {"source": d.source, "node": d.node, "probs": dict(d.probs)})
            d = _adjust(d, "monster", self._monster, hero, hole, board[:2 + st.street], st)
            d = _adjust(d, "trap", self._trap, hero, hole, board[:2 + st.street], st)
            if self.river_jam:
                d = _adjust(d, "river_jam", self._river_jam, hero, hole, board[:2 + st.street], st)
            if self._short(st, hero):
                d = _adjust(d, "short_stack", self._short_stack, hero, hole, board[:2 + st.street], st)
            elif self.pot_odds and st.facing(hero) > EPS:
                if "fold" in self._options and "call" in self._options:
                    before = dict(d.probs)
                    probs = {k: d.probs.get(k, 0.0) for k in set(d.probs) | {"fold", "call"}}
                    evidence = {}
                    d.probs = _normalise(potodds.check(probs, hole, board[:2 + st.street], st, hero, self.pre,
                                                       self.opp_profile, self.exploit["defend"], d.notes,
                                                       self.size_value, self._bluffs(), self.strict,
                                                       diagnostics=evidence))
                    d.diagnostics["pot_odds"] = evidence
                    _trace(d.diagnostics, "pot_odds", before, d.probs)
        d.diagnostics.update({
            "exact": exact,
            "context": {"hero_position": hero, "hero_hole": "".join(map(card_str, hole)),
                        "board": "".join(map(card_str, board[:2 + st.street])) if st.street else "",
                        "history": [[[a.player, a.kind, a.to] for a in street] for street in st.history],
                        "effective_stack_bb": st.stack, "pot_bb": st.pot(), "to_call_bb": st.facing(hero)},
            "adaptation": {"strictness": self.strict, "preflop_weight": self.lam,
                           "bluff_scale": self.exploit["bluff"], "defend_scale": self.exploit["defend"],
                           "opponent_hands": (self.opp_profile or {}).get("hands", 0),
                           "opponent_profile": self.opp_profile,
                           "read_evidence": self.read_evidence,
                           "bluff_model": self.bluff_model.w.tolist() if self.bluff_model else None},
            "legal_options": {k: {"action": a, "to_bb": to} for k, (a, to) in self._options.items()},
        })
        fallbacks = []
        if d.source != "exact":
            fallbacks.append(f"exact:{exact['status']}")
            fallbacks.append("legacy_preflop_hand_groups" if st.street == 0 else "legacy_postflop_buckets")
            lookup = d.diagnostics.get("lookup", {})
            if d.source == "solver":
                fallbacks.append("legacy_representative_flop")
            if d.source == "rule":
                fallbacks.append("legacy_raise_sequence_rule")
            if lookup.get("untrained_fallback"):
                fallbacks.append("legacy_untrained_cell")
            if lookup.get("depth_bb") and lookup["depth_bb"] != stack:
                fallbacks.append("legacy_nearest_stack")
            if lookup.get("depth_node_borrow"):
                fallbacks.append("legacy_depth_node_borrow")
            if lookup.get("size_translation"):
                fallbacks.append("legacy_size_translation")
        d.diagnostics.update(strategy_mode="exact_required" if self.exact_required else "legacy_permitted",
                             fallbacks=fallbacks, coverage=self._record_coverage(
                                 context, exact["status"], "exact" if d.source == "exact" else "legacy", fallbacks))
        d.diagnostics.setdefault("adjustments", [])
        if sample:
            labels = list(d.probs)
            p = np.array([d.probs[k] for k in labels], dtype=float)
            d.label = labels[int(self.rng.choice(len(labels), p=p / p.sum()))]
        else:
            d.label = max(d.probs, key=d.probs.get)
        d.action, d.to = self._options[d.label]
        return d

    def _record_coverage(self, context, status, route, reasons):
        self.coverage_counts[route] = self.coverage_counts.get(route, 0) + 1
        if self.coverage_path is not None:
            from bot.coverage import CoverageStore
            with CoverageStore(self.coverage_path) as store:
                store.record(context, status, route, reasons)
                return store.summary()
        return {"attempts": sum(self.coverage_counts.values()), "routes": dict(self.coverage_counts),
                "persistent_queue": False}

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
        diagnostics = {"base": {"source": "preflop", "node": node, "probs": dict(probs)},
                       "lookup": {"depth_bb": depth, "depth_node_borrow": n.borrowed}}
        base = node.split(" ")[0]
        if deep and self.lam > 0 and base in self.adapted:
            ad = self.adapted[base]
            probs = {a: (1 - self.lam) * p + self.lam * float(ad[a][i]) if a in ad else (1 - self.lam) * p
                     for a, p in probs.items()}
            _trace(diagnostics, "preflop_adaptation", diagnostics["base"]["probs"], probs)
            node += f" (adapted {self.lam:.0%})"
        if not deep:
            node += f" ({depth:g}bb tree)"
        self._options = opts
        return Decision("", 0, _normalise(probs), node, "preflop", diagnostics=diagnostics)

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
        diagnostics = {"lookup": {"pot_type": pot, "texture": tex, "line": line,
                                   "bucket": int(bkt), "untrained_fallback": False}}
        diagnostics["lookup"]["size_translation"] = weights[0][0] is not None and (len(weights) > 1 or abs(aggressive[0].to / P - (bets if seq.startswith("x") else leads)[weights[0][0]]) > EPS)
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
            if any(flat(row(c)) for c, _ in cells):
                notes.append("untrained legacy cell; deep-stack substitution disabled")
                diagnostics["lookup"]["untrained_fallback"] = True
            probs = _blend([(row(c), w) for c, w in cells])
            diagnostics["base"] = {"source": source, "node": node, "probs": dict(probs)}
            notes.append(f"pot {pot}, tex {tex}, line {line}, group {fg.BUCKETS[bkt]}")
            before = dict(probs)
            probs = self._oop_correction(probs, s, kind, bkt, st, notes, aggressive[0].to / P if aggressive else 0)
            _trace(diagnostics, "oop_correction", before, probs)
            if kind == "bb_first" and s == 1 and pot in ("srp", "srp_alt"):
                before = dict(probs)
                probs = _small_leads(probs, notes)
                _trace(diagnostics, "small_leads", before, probs)
            if s == 2 and kind in ("btn_vs_check", "bb_first"):
                before = dict(probs)
                probs = _polar_river(probs, notes, self.river_shift)
                _trace(diagnostics, "polar_river", before, probs)

        before_exploit = dict(probs)
        probs = self._exploit(probs, kind, bkt, notes, aggressive[0].to / P if aggressive else 0)
        # Can't raise an all-in player; a raise that needs the whole stack is an all-in.
        if opp_all_in or hero_left <= b_facing + EPS:
            for lab in [l for l in probs if l.startswith("raise")]:
                probs["call"] = probs.get("call", 0) + probs.pop(lab)
                opts.pop(lab, None)
        self._options = opts
        d = Decision("", 0, _normalise(probs), node, source, notes=notes)
        d.diagnostics = diagnostics
        d.diagnostics.setdefault("base", {"source": source, "node": node, "probs": before_exploit})
        _trace(d.diagnostics, "exploit_and_legality", before_exploit, d.probs)
        return d

    # -- experiment 40: short stacks postflop -------------------------------------------------------

    def _monster(self, d, hero, hole, board, st):
        """Experiments 43-44: with the nuts the bot never just calls or checks: facing a bet it raises (all-in on the
        river or when the stack is short, else its biggest raise), first to act it bets its biggest size (all-in on
        the river). The nuts: no two cards the opponent can hold beat the hand (straight flush, quads beyond the
        board, the nut flush on boards with no full house possible), or the best possible flush on any board."""
        if not _is_nuts(list(hole), list(board)):
            return d
        b = st.facing(hero)
        all_in = st.inp[hero] + st.left(hero)
        other = "bb" if hero == "btn" else "btn"
        P = st.pot()
        jam = st.street == 3 or st.left(hero) <= 2.5 * (P + b)
        if b > EPS:
            if st.left(other) <= EPS or st.left(hero) <= b + EPS:
                target = "call"                              # they are all-in (or we can't raise): calling is max
            else:
                raises = [k for k in self._options if k.startswith("raise")]
                if jam or not raises:
                    self._options["raise all-in"] = ("raise", all_in)
                    target = "raise all-in"
                else:
                    target = max(raises, key=lambda k: self._options[k][1])
        else:
            bets = [k for k in self._options if self._options[k][0] in ("bet", "raise")]
            if jam or not bets:
                self._options["bet all-in"] = ("bet", all_in)
                target = "bet all-in"
            else:
                target = max(bets, key=lambda k: self._options[k][1])
        d.probs = {target: 1.0}
        d.notes.append(f"the nuts: {target} (never just call or check)")
        return d

    def _river_jam(self, d, hero, hole, board, st):
        """Experiment 47 (Polk vs Negreanu: 'too conservative with putting in all the money'; Polk's river shoves of
        3x pot): first to bet the river with at most RIVER_JAM_SPR pots behind, a strong hand (two pair+, see
        strong_class) moves RIVER_JAM_SHARE of its pot-or-bigger bets to all-in, and air (no pair beyond the board)
        the same share of its overbets, so the jam stays polar. Medium hands never jam."""
        if st.street != 3 or st.facing(hero) > EPS or self._short(st, hero):
            return d
        P, left = st.pot(), st.left(hero)
        if left <= EPS or left > RIVER_JAM_SPR * P:
            return d
        strong = strong_class(list(hole), list(board)) is not None
        air = (eval7.evaluate(list(hole) + list(board)) >> 24) <= (eval7.evaluate(list(board)) >> 24)
        if not (strong or air):
            return d
        floor = 1.0 if strong else 1.5
        big = [k for k in d.probs if self._options.get(k, ("",))[0] in ("bet", "raise")
               and (self._options[k][1] - max(st.inp.values())) >= floor * P - EPS]
        moved = sum(d.probs[k] for k in big) * RIVER_JAM_SHARE
        if moved <= 1e-6:
            return d
        probs = {k: (p * (1 - RIVER_JAM_SHARE) if k in big else p) for k, p in d.probs.items()}
        self._options["bet all-in"] = ("bet", st.inp[hero] + left)
        probs["bet all-in"] = probs.get("bet all-in", 0.0) + moved
        d.probs = _normalise(probs)
        d.notes.append(f"river jam ({'value' if strong else 'bluff'}): {moved:.0%} of big bets -> all-in")
        return d

    def _vs_shove(self, d, hero, hole, st):
        """Use only a supported direct exact-jam frequency at this depth/history.

        The top-strength range remains a heuristic; frequency confidence does
        not establish range calibration or a poker EV improvement.
        """
        if "call" not in self._options or "fold" not in self._options:
            return d
        opp = "bb" if hero == "btn" else "btn"
        b = st.facing(hero)
        if st.left(opp) > EPS and st.left(hero) > b + EPS:
            return d                                         # not an all-in (theirs or ours by calling)
        node, reach = _shove_node(st.history[0], opp)
        from bot.reads import MAX_WIDTH, MIN_HANDS, MARGIN, VERSION, interval, jam_key
        # This channel alone must support a shove-rate change; raises/limps
        # cannot qualify a shove read. Conditional reach needs a joint model.
        i = max(k for k, a in enumerate(st.history[0]) if a.player == opp)
        prefix = [(a.player, a.kind, a.to) for a in st.history[0][:i]]
        key = jam_key(node, st.stack, prefix)
        obs = self.read_evidence.get("jams", {}).get(key, {})
        n = obs.get("n", 0)
        seen = obs.get("hits", 0)
        # Countable context allocation preserves the family error budget as
        # new exact depth/size contexts arrive, in first-observation order.
        rank = obs.get("index", 1)
        lo, hi = interval(seen, n, rank * (rank + 1))
        prior = SHOVE_PRIOR[0]
        applied = (self.read_evidence.get("version") == VERSION and n >= MIN_HANDS
                   and hi - lo <= MAX_WIDTH and (lo > prior + MARGIN or hi < prior - MARGIN)
                   and reach is None)
        d.diagnostics["shove_evidence"] = {
            "applied": applied, "n_hands": n, "interval": [lo, hi], "reference": prior, "context": key,
            "reason": "supported_shove_frequency" if applied else "uncertain_shove_frequency_or_conditional_reach"}
        if not applied:
            return d
        f = lo if lo > prior else hi
        share = 1.0
        eq = _equity_vs_top(hole, max(0.01, min(1.0, f * share)))
        need = b / (st.pot() + b)
        d.diagnostics["vs_shove"] = {"equity": eq, "price": need, "range_top_share": f * share,
                                      "observed_shoves": seen, "opportunities": n}
        call = eq > need + SHOVE_MARGIN
        d.probs = {"call": 1.0} if call else {"fold": 1.0}
        d.shove_read = True
        d.notes.append(f"vs shove: they shove {f:.0%} here ({seen:.0f}/{n}), range top {f * share:.0%}: "
                       f"equity {eq:.0%} vs price {need:.0%}: {'call' if call else 'fold'}")
        return d

    def _vs_reraise_war(self, d, hero, hole, st):
        """Experiment 48: facing a 4-bet or 5-bet, the tree assumes equilibrium ranges full of bluffs; live players
        showed value (3+ raise pots: -123bb in 10 hands; 33 and T9o called 4-bets, TT a 5-bet). As far as the read
        is value-heavy (strictness, from the population and their showdowns), continue only when the hand's
        equity against a value range (top REWAR_RANGE of hands for that many raises), discounted for realisation
        out of a bloated, low-SPR pot, beats the price; jams need REWAR_JAM_EQ raw equity. A shove read
        (_vs_shove) decides all-ins it has evidence for."""
        raises = sum(a.kind == "raise" for a in st.history[0])
        if raises < 3 or st.facing(hero) <= EPS or "fold" not in self._options or self.strict <= 0:
            return d
        if d.shove_read:
            return d                                         # _vs_shove decided it from this opponent's shoves
        share = REWAR_RANGE.get(raises, min(REWAR_RANGE.values()))
        eq = _equity_vs_top(hole, share)
        b = st.facing(hero)
        need = b / (st.pot() + b)
        opp = "bb" if hero == "btn" else "btn"
        closes = st.left(hero) <= b + EPS or st.left(opp) <= EPS   # an all-in call: no later streets to realise
        real = 1.0 if closes else REWAR_REAL
        # Full weight only with a read on this opponent; the population prior alone keeps the tree in the mix.
        own = bool(((self.opp_profile or {}).get("nodes") or {}))
        w = min(1.0 if own else REWAR_PRIOR_CAP, self.strict / REWAR_FULL_STRICT)
        d.diagnostics["reraise_war"] = {"equity": eq, "price": need, "range_top_share": share,
                                         "realization": real, "realized_equity": eq * real, "weight": w}
        go = [k for k in d.probs if k != "fold" and d.probs[k] > 0]
        cont = sum(d.probs[k] for k in go)
        if cont <= 0:
            return d
        probs = dict(d.probs)
        if eq * real <= need:
            for k in go:
                probs[k] *= 1 - w
            probs["fold"] = probs.get("fold", 0.0) + cont * w
            verdict = "fold"
        elif eq < REWAR_JAM_EQ and probs.get("all-in", 0) > 0 and "call" in self._options:
            x = probs.pop("all-in") * w
            probs["all-in"] = d.probs["all-in"] - x
            probs["call"] = probs.get("call", 0.0) + x
            verdict = "call, not jam"
        else:
            return d
        d.probs = _normalise(probs)
        d.notes.append(f"re-raise war ({raises} raises): equity {eq:.0%} vs top {share:.0%} (x{real}) vs price "
                       f"{need:.0%}: {verdict} ({w:.0%} weight, strictness {self.strict:.2f})")
        return d

    def _small_raise(self, d, hero, hole, st):
        """Experiment 46: the tree reads any preflop raise as its own (big) size; a min-raise priced at
        SMALL_RAISE_PRICE or less is only folded when the hand's equity against a strong range (top
        SMALL_RAISE_RANGE), discounted for realisation, can't beat the price. Min-raising bots exploited the folds."""
        b = st.facing(hero)
        if b <= EPS or d.probs.get("fold", 0.0) <= 0 or "call" not in self._options:
            return d
        # Experiment 48: only against an open or a 3-bet. A 4-bet or 5-bet range is far stronger than any top-25%
        # read, and a cheap-looking call there commits the stack at a tiny SPR: live, it called 1.7x 4-bets with 33 and
        # T9o and a 5-bet with TT (-151bb in 3 hands); the tree's own 4-bet / 5-bet answers decide those.
        if sum(a.kind == "raise" for a in st.history[0]) > SMALL_RAISE_MAX_RAISES:
            return d
        need = b / (st.pot() + b)
        if need > SMALL_RAISE_PRICE:
            return d
        raw_eq = _equity_vs_top(hole, SMALL_RAISE_RANGE)
        eq = raw_eq * SMALL_RAISE_REAL
        d.diagnostics["small_raise"] = {"equity": raw_eq, "price": need,
                                        "range_top_share": SMALL_RAISE_RANGE,
                                        "realization": SMALL_RAISE_REAL, "realized_equity": eq}
        if eq <= need:
            return d
        probs = dict(d.probs)
        probs["call"] = probs.get("call", 0.0) + probs.pop("fold")
        d.probs = probs
        d.notes.append(f"small raise: price {need:.0%} vs equity {eq:.0%} (top {SMALL_RAISE_RANGE:.0%}, realised): call")
        return d

    def _trap(self, d, hero, hole, board, st):
        """Experiment 45: slow-play strong hands as often as the solver does on this texture (TRAP), more against
        aggressive opponents and less against passive ones (their bet rate when checked to, and the aggression read).
        The flop BTN never checks back strong hands, the river is never trapped, and with the nuts the bot only
        traps out of position on the flop or turn against a very aggressive opponent (and then raises their bet:
        _monster)."""
        if st.street == 3 or self._short(st, hero):
            return d
        k = strong_class(list(hole), list(board))
        if k is None:
            return d
        flat, bb_check, btn_check = TRAP.get((trap_texture(board), k), TRAP_CLASS.get(k, (None, None, None)))
        facing = st.facing(hero) > EPS
        if facing:
            rate = flat if st.street == 1 else None          # the solver tree had no turn raises
        else:
            rate = None if st.street == 1 else bb_check if hero == "bb" else btn_check
        if rate is None:
            return d
        mult = float(np.clip(self.exploit["defend"] * self._checked_to_bet(st.street) / TRAP_BET_PRIOR, 0.0, 2.0))
        nuts = any(n.startswith("the nuts") for n in d.notes)
        if nuts:
            if facing or hero != "bb" or self.exploit["defend"] < TRAP_NUTS_AGGRESSION:
                return d
        passive = "call" if facing else "check"
        aggressive = [x for x in self._options if self._options[x][0] in ("bet", "raise")]
        if passive not in self._options or not aggressive:
            return d
        target = min(0.9, rate * mult)
        agg = sum(d.probs.get(x, 0.0) for x in aggressive)
        if agg + d.probs.get(passive, 0.0) <= EPS:
            return d
        share = agg + d.probs.get(passive, 0.0)                # the fold share (facing a bet) stays as it is
        new_agg = share * (1 - target)
        if nuts or agg <= EPS:
            probs = {x: p for x, p in d.probs.items() if x not in aggressive}
            best = max(aggressive, key=lambda x: self._options[x][1]) if agg <= EPS else None
            for x in aggressive:
                if d.probs.get(x, 0.0) > 0 or x == best:
                    probs[x] = new_agg * (d.probs.get(x, 0.0) / agg if agg > EPS else 1.0)
        else:
            probs = {x: (p * new_agg / agg if x in aggressive else p) for x, p in d.probs.items()}
        probs[passive] = share * target
        d.probs = _normalise(probs)
        d.notes.append(f"trap: {k} on a {trap_texture(board)} board: {passive} {target:.0%} (solver {rate:.0%} x{mult:.2f})")
        return d

    def _checked_to_bet(self, street):
        """The opponent's bet rate when checked to on this street, shrunk to the prior (bot/opponent.py profile)."""
        from bot.reads import approved_nodes
        x = approved_nodes(self.opp_profile).get(f"{['', 'flop', 'turn', 'river'][street]}_bet_when_checked_to", {})
        n = x.get("n", 0)
        return (n * x.get("bet", 0.0) + 10 * TRAP_BET_PRIOR) / (n + 10)

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
        out = Decision("", 0, probs, d.node + " (short stack)", "short",
                       notes=d.notes + [f"short stack ({st.left(hero):.1f}bb behind): {why}"])
        out.diagnostics["short_stack"] = {"equity": eq, "price": b / (st.pot() + b) if b > EPS else 0.0,
                                           "jam_threshold": SHORT_JAM, "range_combos": len(rng)}
        return out

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


def _trace(diagnostics, stage, before, after):
    """Record only probability changes; do not alter normalization or random sampling."""
    if before != after:
        diagnostics.setdefault("adjustments", []).append(
            {"stage": stage, "before": dict(before), "after": dict(after)})


def _adjust(d, stage, fn, *args):
    before, diagnostics = dict(d.probs), d.diagnostics
    result = fn(d, *args)
    result.diagnostics = {**diagnostics, **result.diagnostics}
    _trace(result.diagnostics, stage, before, result.probs)
    return result


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


def trap_texture(board):
    """Board texture for trapping (experiment 45): paired / monotone / wet (two-tone or straight-possible) / dry."""
    ranks = [c.rank for c in board]
    suits = [c.suit for c in board]
    if len(set(ranks)) < len(ranks):
        return "paired"
    if max(suits.count(x) for x in set(suits)) >= 3:
        return "monotone"
    r = sorted(set(ranks) | ({-1} if 12 in ranks else set()))        # the ace also plays low
    straighty = any(sum(a <= x <= a + 4 for x in r) >= 3 for a in r)
    two_tone = max(suits.count(x) for x in set(suits)) == 2 and len(board) == 3
    return "wet" if straighty or two_tone else "dry"


# Experiment 47 (Polk vs Negreanu): jam the river with polar hands when the stack is at most RIVER_JAM_SPR pots.
RIVER_JAM_SPR, RIVER_JAM_SHARE = 3.0, 0.5


SMALL_RAISE_PRICE = 0.2     # experiment 46: preflop raises priced at most this get the equity check
SMALL_RAISE_RANGE, SMALL_RAISE_REAL = 0.25, 0.75
# Experiment 48: value ranges against a 4-bet (3 raises) and a 5-bet+ (4+), realisation in those pots, the raw equity a
# jam needs, and the strictness at which the check has full weight (the population start is 0.37).
REWAR_RANGE = {3: 0.06, 4: 0.035}
REWAR_REAL, REWAR_JAM_EQ, REWAR_FULL_STRICT = 0.55, 0.48, 0.3
REWAR_PRIOR_CAP = 0.75      # weight of the check on the population read alone (no hands of this opponent yet)
SMALL_RAISE_MAX_RAISES = 2  # experiment 48: the check applies facing an open / iso or a 3-bet, never a 4-bet+
SHOVE_MIN_SEEN = 3          # experiment 46: preflop shoves seen at a spot before the read replaces the tree
SHOVE_PRIOR = (0.03, 5)     # prior shove share and its weight (deals)
SHOVE_MARGIN = 0.01         # equity needed above the price to call


def _shove_node(pre, opp):
    """The opponent's profile node for their latest preflop action, and the node + kinds that got their hand
    there (None: every hand)."""
    acts = [(a.player, a.kind) for a in pre]
    i = max(k for k, (p, _) in enumerate(acts) if p == opp)
    prev = [(p, "call" if k == "call" else "raise") for p, k in acts[:i]]
    if opp == "btn" and not prev:
        return "btn_first_in", None
    if opp == "bb" and prev == [("btn", "call")]:
        return "bb_vs_limp", None
    if opp == "bb" and prev == [("btn", "raise")]:
        return "bb_vs_open", None
    if opp == "btn" and prev == [("btn", "call"), ("bb", "raise")]:
        return "btn_vs_iso", ("btn_first_in", ("limp",))
    if opp == "btn" and prev == [("btn", "raise"), ("bb", "raise")]:
        return "btn_vs_3bet", ("btn_first_in", ("raise",))
    if opp == "bb" and prev == [("btn", "raise"), ("bb", "raise"), ("btn", "raise")]:
        return "bb_vs_4bet", ("bb_vs_open", ("raise",))
    return None, None


_EQ = None


def _equity_vs_top(hole, share):
    """Preflop all-in equity of this hand against the top `share` of hands (by equity vs random, by combos)."""
    global _EQ
    if _EQ is None:
        d = np.load(Path(__file__).resolve().parents[1] / "data" / "equity_matrix.npz")
        idx = [INDEX[h] for h in d["hands"]]
        E, W = np.zeros((169, 169)), np.zeros((169, 169))
        E[np.ix_(idx, idx)], W[np.ix_(idx, idx)] = d["E"], d["W"]
        vr = (E * W).sum(1) / W.sum(1)
        order = np.argsort(-vr)
        combos = np.array([6 if h[0] == h[1] else 4 if h.endswith("s") else 12 for h in HANDS], float)
        _EQ = (E, W, order, combos)
    E, W, order, combos = _EQ
    cum = np.cumsum(combos[order]) / combos.sum()
    w = np.zeros(169)
    k = int(np.searchsorted(cum, share))
    w[order[:k + 1]] = 1.0
    i = INDEX[hand_class(hole)]
    ww = W[i] * w
    return float((E[i] * ww).sum() / max(ww.sum(), 1e-9))


def strong_class(hole, board):
    """Strong made-hand class for trapping (experiment 45): two pair (both hole cards), set, trips, straight, flush,
    full house+ -- None for anything weaker or when the board plays."""
    t = eval7.evaluate(hole + board) >> 24
    if t <= 1 or eval7.evaluate(board) >> 24 >= t:
        return None
    ranks = [c.rank for c in board]
    if t == 2:
        return "two pair" if len(set(ranks)) == len(ranks) and all(c.rank in ranks for c in hole) else None
    if t == 3:
        return "set" if hole[0].rank == hole[1].rank else "trips"
    return {4: "straight", 5: "flush"}.get(t, "full house+")


def _is_nuts(hole, board):
    """No two-card hand the opponent can hold beats ours, or ours is the best possible flush."""
    if len(board) < 3:
        return False
    mine = eval7.evaluate(hole + board)
    if mine >> 24 < 4:                                       # below a straight: never the nuts here
        return False
    dead = set(hole) | set(board)
    rest = [c for c in _DECK if c not in dead]
    nuts = all(eval7.evaluate([a, c] + board) <= mine for i, a in enumerate(rest) for c in rest[i + 1:])
    if nuts or mine >> 24 != 5:
        return nuts
    best_flush = max((eval7.evaluate([a, c] + board) for i, a in enumerate(rest) for c in rest[i + 1:]
                      if eval7.evaluate([a, c] + board) >> 24 == 5), default=0)
    return mine >= best_flush


_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs"]


class _Fallback(dict):
    """A depth tree's nodes; a node it lacks comes from the 100bb tree, else from the deepest shorter tree that has
    it (at 100bb there is no 4-bet shove, so no node for facing one; the 25bb tree has it)."""
    def __init__(self, nodes, *others):
        super().__init__(nodes)
        self.others = others
        self.borrowed = []

    def __missing__(self, key):
        for o in self.others:
            if key in o:
                self.borrowed.append(key)
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
