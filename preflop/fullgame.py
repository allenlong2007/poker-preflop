"""The whole postflop hand -- flop, turn and river -- learned together (experiment 15).

Earlier games learned one street at a time and played the later streets with
simple rules, so no player could plan a line like "check-call the flop and the
turn" or "slowplay the flop, raise the turn". Here all three streets are one
game, learned with external-sampling Monte Carlo CFR: for each sampled hand,
the learning player tries every option at its decisions while the opponent's
moves (and the cards, which are dealt up front) are sampled. A flop decision is
therefore valued by how the rest of the hand actually plays out under the
current turn and river strategies.

Each street (BTN opened 2.25bb preflop, BB called):

    BB:   check | lead (size j)                (flop lead = "donk", only if enabled)
      after a check:  BTN  check | bet (size k)
                      BB   fold | call | raise (multiple r)
                      BTN  fold | call
      after a lead:   BTN  fold | call | raise (multiple r)
                      BB   fold | call

Sizes default to SIZES / RAISE_X; opts["sizes"] and opts["raises"] override them.

Decisions are learned per (board texture, line, hand bucket[, size]):
  * texture: the flop / turn / river textures of experiments 8, 11 and 12
  * line: who bet last on the previous street (nobody / BTN / BB) x how deep
    the stacks are vs the pot (stack-to-pot ratio >= 4, 1.5-4, < 1.5)
  * 12 finer hand buckets (below); on the river the draws become missed draws
"""
import os
from concurrent.futures import ProcessPoolExecutor

import eval7
import numpy as np

from .cards import HANDS, n_combos
from .flopgame import TEXTURES as FLOP_TEXTURES, texture as flop_texture
from .postflop import COMBOS, DECK, _has_draw
from .rivergame import TEXTURES as RIVER_TEXTURES, river_texture
from .turngame import TEXTURES as TURN_TEXTURES, turn_texture

STREETS = ("flop", "turn", "river")
N_TEX = (len(FLOP_TEXTURES), len(TURN_TEXTURES), len(RIVER_TEXTURES))
N_LINES = (1, 9, 9)
SIZES = ({"lead": (1 / 3,), "bet": (1 / 3, 3 / 4)},
         {"lead": (1 / 2,), "bet": (1 / 2, 1.0)},
         {"lead": (1 / 2,), "bet": (1 / 2, 1.0, 3 / 2)})
RAISE_X = 3.0

BUCKETS = ["straight or better", "set / trips", "two pair", "overpair", "top pair, T+ kicker",
           "top pair, weak kicker", "middle pair", "weak pair", "strong draw", "weak draw", "overcards", "air"]
RIVER_BUCKETS = BUCKETS[:8] + ["missed strong draw", "missed weak draw", "overcards", "air"]
NB = len(BUCKETS)
STRONG = (0, 1, 2)                 # hands that can slowplay
MEDIUM = (4, 5, 6, 7)              # one-pair hands below an overpair: check-call candidates
WEAK = (9, 10, 11)                 # bluffing hands (weak draws, overcards, air); 8 = strong draw (semi-bluff)

# Node kinds: (who decides, has a size index, number of actions given (n_lead, n_bet))
NODES = {"bb_first": ("bb", False), "btn_vs_check": ("btn", False), "bb_vs_bet": ("bb", True),
         "btn_vs_xr": ("btn", True), "btn_vs_lead": ("btn", True), "bb_vs_raise": ("bb", True)}


def street_sizes(s, opts):
    """(lead sizes, bet sizes) on street s; opts["sizes"] overrides SIZES, and the flop lead needs opts["donk"]."""
    sz = (opts.get("sizes") or SIZES)[s]
    return (tuple(sz["lead"]) if (s > 0 or opts.get("donk")) else ()), tuple(sz["bet"])


def raise_sizes(opts):
    """Raise multiples available to whoever faces a bet or lead (opts["raises"], default 3x only)."""
    return tuple(opts.get("raises", (RAISE_X,)))


def n_actions(kind, s, opts):
    """(size dimensions, number of actions) for a node kind on street s."""
    n_lead, n_bet = (len(x) for x in street_sizes(s, opts))
    n_raise = len(raise_sizes(opts))
    return {"bb_first": ((), 1 + n_lead), "btn_vs_check": ((), 1 + n_bet), "bb_vs_bet": ((n_bet,), 2 + n_raise),
            "btn_vs_xr": ((n_bet, n_raise), 2), "btn_vs_lead": ((n_lead,), 2 + n_raise),
            "bb_vs_raise": ((n_lead, n_raise), 2)}[kind]


def table_shapes(opts):
    shapes = {}
    for s in range(3):
        for kind in NODES:
            dims, a = n_actions(kind, s, opts)
            if 0 in dims:
                continue
            shapes[(s, kind)] = (N_TEX[s], N_LINES[s], NB) + dims + (a,)
    return shapes


# ----------------------------------------------------------------------------- hand features

def _straight_draw(hole, board):
    """Any four-to-a-straight (open-ended or gutshot) that uses a hole card."""
    ranks = {c.rank for c in list(hole) + board}
    if 12 in ranks:
        ranks.add(-1)
    hole_ranks = {c.rank for c in hole} | ({-1} if any(c.rank == 12 for c in hole) else set())
    return any(len(ranks & set(range(lo, lo + 5))) >= 4 and hole_ranks & set(range(lo, lo + 5))
               for lo in range(-1, 9))


def fine_bucket(hole, board, prev=None):
    """One of BUCKETS for this hole hand on this board (3, 4 or 5 cards)."""
    typ = eval7.evaluate(list(hole) + board) >> 24
    btyp = eval7.evaluate(board) >> 24
    river = len(board) == 5
    ranks = sorted({c.rank for c in board}, reverse=True)
    h1, h2 = hole[0].rank, hole[1].rank
    if typ >= 4 and typ > btyp:
        return 0
    if typ == 3 and btyp < 3:
        return 1
    if typ == 2 and btyp == 0 and h1 != h2 and h1 in ranks and h2 in ranks:
        return 2
    top = ranks[0]
    second = ranks[1] if len(ranks) > 1 else -1
    if h1 == h2:
        return 3 if h1 > top else 6 if h1 > second else 7
    if top in (h1, h2):
        kicker = h2 if h1 == top else h1
        return 4 if kicker >= 8 else 5
    if second in (h1, h2):
        return 6
    if h1 in ranks or h2 in ranks:
        return 7
    if river:
        if prev in (8, 9):
            return prev                     # missed strong / weak draw
    elif _has_draw(hole, board):
        return 8
    elif _straight_draw(hole, board) or (len(board) == 3 and hole[0].suit == hole[1].suit
                                         and sum(c.suit == hole[0].suit for c in board) == 1):
        return 9                            # gutshot or backdoor flush draw
    return 10 if min(h1, h2) > top else 11


def line_index(aggressor, pot, left):
    spr = left / pot
    return aggressor * 3 + (0 if spr >= 4 else 1 if spr >= 1.5 else 2)


class Hand:
    __slots__ = ("i", "j", "tex", "bb", "btn", "r")


def deal(rng, i, j):
    btn = COMBOS[i][rng.integers(len(COMBOS[i]))]
    bb = COMBOS[j][rng.integers(len(COMBOS[j]))]
    if set(btn) & set(bb):
        return None
    dead = set(btn) | set(bb)
    live = [c for c in DECK if c not in dead]
    board = [live[k] for k in rng.choice(len(live), size=5, replace=False)]
    h = Hand()
    h.i, h.j = i, j
    h.tex = (flop_texture(board[:3]), turn_texture(board[:3], board[3]), river_texture(board[:4], board[4]))
    for name, hole in (("btn", btn), ("bb", bb)):
        bs, prev = [], None
        for n in (3, 4, 5):
            prev = fine_bucket(hole, board[:n], prev)
            bs.append(prev)
        setattr(h, name, tuple(bs))
    v_b, v_o = eval7.evaluate(list(btn) + board), eval7.evaluate(list(bb) + board)
    h.r = 1.0 if v_b > v_o else 0.5 if v_b == v_o else 0.0
    return h


# ----------------------------------------------------------------------------- the game

AGGRESSIVE = {"bb_first": lambda a: a >= 1, "btn_vs_check": lambda a: a >= 1, "bb_vs_bet": lambda a: a >= 2,
              "btn_vs_xr": lambda a: False, "btn_vs_lead": lambda a: a >= 2, "bb_vs_raise": lambda a: False}
FLOP_CLASS = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3]     # BTN flop bucket -> strong / pair / draw / air (line stats)


def is_check_call(code):
    """BB checked, the BTN bet (any size), the BB called."""
    return code.startswith("xb") and code.endswith("c") and "r" not in code


class Game:
    """One sampled hand, played in 'train' mode (external-sampling MCCFR) or 'eval' mode (exact expectation)."""

    def __init__(self, h, sigma, opts, mode, trav=None, rng=None, acc=None, wt=1.0, override=None):
        self.h, self.sigma, self.opts, self.mode = h, sigma, opts, mode
        self.trav, self.rng, self.acc, self.wt = trav, rng, acc, wt
        self.override = override          # sample mode: f(street, kind, cell, player, sigma) -> sigma or None
        self.final = None                 # sample mode: the hand's complete betting line

    # Policy at a decision, after any forced choices or banned actions for this experiment.
    def policy(self, s, kind, cell, player):
        sig = self.sigma[(s, kind)][cell]
        o, b = self.opts, cell[2]
        if o.get("donk_profile") is not None and s == 0 and kind == "bb_first":
            p = float(o["donk_profile"][b])                 # opponent profile: P(donk 1/3 pot | flop bucket)
            out = np.zeros(len(sig))
            out[0], out[1] = 1 - p, p
            return out, False, None
        if o.get("force_donk") is not None and s == 0 and kind == "bb_first":
            ft = cell[0]
            if o.get("force_donk_textures") is None or ft in o["force_donk_textures"]:
                p = o["force_donk"]
                return np.array([1 - p, p]), False, None
        if o.get("slowplay") is not None and s == 0 and kind == "bb_vs_bet" and b in STRONG:
            sp, nr = o["slowplay"], len(sig) - 2
            return np.array([0.0, sp] + [(1 - sp) / nr] * nr), False, None
        if o.get("no_early_bluffs") and s < 2 and player == "bb" and b in WEAK:
            allowed = np.array([not AGGRESSIVE[kind](a) for a in range(len(sig))])
            if not allowed.all():
                sig = np.where(allowed, sig, 0.0)
                tot = sig.sum()
                sig = sig / tot if tot > 0 else allowed / allowed.sum()
                return sig, True, allowed
        return sig, True, None

    def decide(self, s, kind, cell, player, acts, rc):
        sig, learn, allowed = self.policy(s, kind, cell, player)
        key = (s, kind)
        if self.mode == "eval":
            self.acc.node(key, cell, sig, rc, player, s, kind, self.h)
            if key in self.opts.get("eval_all", ()):
                # "What if" values: play out every action (even ones the strategy never takes), so the
                # value of each option vs the opponent's actual responses can be compared.
                vals = [acts[a](rc * p) for a, p in enumerate(sig)]
                self.acc.action_value(key, cell, vals, rc)
                return float(sum(p * v for p, v in zip(sig, vals)))
            v = 0.0
            for a, p in enumerate(sig):
                if p > 1e-6:
                    v += p * acts[a](rc * p)
            return v
        if self.mode == "sample":                           # play one concrete hand: sample every decision
            if self.override is not None:
                forced = self.override(s, kind, cell, player, sig)
                if forced is not None:
                    sig = forced
            x, c = self.rng.random(), 0.0
            for a, p in enumerate(sig):
                c += p
                if x < c:
                    return acts[a](1.0)
            return acts[len(sig) - 1](1.0)
        if player == self.trav:
            sgn = 1.0 if player == "btn" else -1.0
            if not learn:
                return sum(p * acts[a](1.0) for a, p in enumerate(sig) if p > 1e-9)
            u = np.zeros(len(sig))
            for a in range(len(sig)):
                if allowed is None or allowed[a]:
                    u[a] = sgn * acts[a](1.0)
            v = float(sig @ u)
            delta = u - v
            if allowed is not None:
                delta = np.where(allowed, delta, 0.0)
            self.acc.dR[key][cell] += delta
            return sgn * v
        if learn:
            self.acc.dS[key][cell] += self.wt * sig
        x, c = self.rng.random(), 0.0
        for a, p in enumerate(sig):
            c += p
            if x < c:
                return acts[a](1.0)
        return acts[len(sig) - 1](1.0)

    def street(self, s, P, left, pin, line, hist, rc):
        h = self.h
        if s == 3 or left <= 1e-9:
            if self.mode == "eval":
                self.acc.line(hist, rc, h.btn[0])
            self.final = hist + ("showdown",)
            return h.r * P - pin
        t, bk, tb = h.tex[s], h.bb[s], h.btn[s]
        leads, bets = street_sizes(s, self.opts)
        raises = raise_sizes(self.opts)
        rc_now = [rc]

        def nxt(P2, left2, pin2, aggressor, code):
            if self.mode == "eval" and is_check_call(code) and all(is_check_call(c) for c in hist):
                self.acc.callchain[s, h.bb[0]] += rc_now[0]      # BB has check-called every street so far
            return self.street(s + 1, P2, left2, pin2, line_index(aggressor, P2, left2), hist + (code,), rc_now[0])

        def end(value, code):                                    # a fold ends the hand
            if self.mode == "eval":
                self.acc.line(hist + (code,), rc_now[0], h.btn[0])
            self.final = hist + (code,)
            return value

        def wrap(f):
            def g(r):
                rc_now[0] = r
                return f()
            return g

        def btn_vs_check():
            def bet(k):
                b = min(bets[k] * P, left)

                def xr(r):
                    R = min(raises[r] * b, left)
                    return self.decide(s, "btn_vs_xr", (t, line, tb, k, r), "btn",
                                       [wrap(lambda: end(-pin - b, f"xb{k}r{r}f")),
                                        wrap(lambda: nxt(P + 2 * R, left - R, pin + R, 2, f"xb{k}r{r}c"))],
                                       rc_now[0])
                return self.decide(s, "bb_vs_bet", (t, line, bk, k), "bb",
                                   [wrap(lambda: end(P - pin, f"xb{k}f")),
                                    wrap(lambda: nxt(P + 2 * b, left - b, pin + b, 1, f"xb{k}c"))]
                                   + [wrap(lambda r=r: xr(r)) for r in range(len(raises))], rc_now[0])
            acts = [wrap(lambda: nxt(P, left, pin, 0, "xx"))] + [wrap(lambda k=k: bet(k)) for k in range(len(bets))]
            return self.decide(s, "btn_vs_check", (t, line, tb), "btn", acts, rc_now[0])

        def lead(j):
            b = min(leads[j] * P, left)

            def raised(r):
                R = min(raises[r] * b, left)
                return self.decide(s, "bb_vs_raise", (t, line, bk, j, r), "bb",
                                   [wrap(lambda: end(P + b - pin, f"l{j}r{r}f")),
                                    wrap(lambda: nxt(P + 2 * R, left - R, pin + R, 1, f"l{j}r{r}c"))], rc_now[0])
            return self.decide(s, "btn_vs_lead", (t, line, tb, j), "btn",
                               [wrap(lambda: end(-pin, f"l{j}f")),
                                wrap(lambda: nxt(P + 2 * b, left - b, pin + b, 2, f"l{j}c"))]
                               + [wrap(lambda r=r: raised(r)) for r in range(len(raises))], rc_now[0])

        if not leads:
            return btn_vs_check()
        acts = [wrap(btn_vs_check)] + [wrap(lambda j=j: lead(j)) for j in range(len(leads))]
        return self.decide(s, "bb_first", (t, line, bk), "bb", acts, rc)

    def play(self, s, P, left):
        return self.street(0, P, left, 0.0, 0, (), 1.0)


# ----------------------------------------------------------------------------- accumulators

class TrainAcc:
    def __init__(self, shapes):
        self.dR = {k: np.zeros(v) for k, v in shapes.items()}
        self.dS = {k: np.zeros(v) for k, v in shapes.items()}


class EvalAcc:
    """Reach-weighted statistics collected while evaluating the average strategy."""

    def __init__(self, shapes):
        self.strat = {k: np.zeros(v) for k, v in shapes.items()}        # reach x policy
        self.reach = {k: np.zeros(v[:-1]) for k, v in shapes.items()}   # reach
        self.aggr = {p: np.zeros((3, NB)) for p in ("btn", "bb")}       # reach x P(bet / raise / lead)
        self.decisions = {p: np.zeros((3, NB)) for p in ("btn", "bb")}  # reach of all decisions
        self.callchain = np.zeros((3, NB))   # BB check-called flop .. street s, by BB's flop bucket
        self.n_flop_bucket = np.zeros(NB)
        self.value_by_flop = np.zeros(N_TEX[0])
        self.n_by_flop = np.zeros(N_TEX[0])
        self.value_by_flop_bucket = np.zeros(NB)
        self.lines = {}                      # full betting line -> reach by BTN flop class (strong/pair/draw/air)
        self.avals = {k: np.zeros(v) for k, v in shapes.items()}       # reach x value of each action ("what if")

    def action_value(self, key, cell, vals, rc):
        self.avals[key][cell] += rc * np.asarray(vals)

    def line(self, seq, rc, btn_flop_bucket):
        key = "/".join(seq)
        arr = self.lines.get(key)
        if arr is None:
            arr = self.lines[key] = np.zeros(4)
        arr[FLOP_CLASS[btn_flop_bucket]] += rc

    def node(self, key, cell, sig, rc, player, s, kind, h):
        self.strat[key][cell] += rc * sig
        self.reach[key][cell] += rc
        b = cell[2]
        self.decisions[player][s, b] += rc
        self.aggr[player][s, b] += rc * sum(p for a, p in enumerate(sig) if AGGRESSIVE[kind](a))


# ----------------------------------------------------------------------------- workers

def _regret_match(R):
    pos = np.maximum(R, 0)
    tot = pos.sum(-1, keepdims=True)
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1 / R.shape[-1])


def _deal_many(btn_w, bb_w, n, seed):
    rng = np.random.default_rng(seed)
    bi = rng.choice(169, size=n, p=btn_w / btn_w.sum())
    bj = rng.choice(169, size=n, p=bb_w / bb_w.sum())
    return rng, [h for h in (deal(rng, i, j) for i, j in zip(bi, bj)) if h is not None]


def _train_worker(job):
    btn_w, bb_w, n, seed, regrets, opts, wt, s_pot, stack = job
    sigma = {k: _regret_match(R) for k, R in regrets.items()}
    acc = TrainAcc({k: R.shape for k, R in regrets.items()})
    rng, hands = _deal_many(btn_w, bb_w, n, seed)
    for h in hands:
        for trav in ("btn", "bb"):
            Game(h, sigma, opts, "train", trav, rng, acc, wt).play(0, s_pot, stack)
    return acc.dR, acc.dS


def _eval_worker(job):
    btn_w, bb_w, n, seed, sigma, opts, s_pot, stack = job
    acc = EvalAcc({k: v.shape for k, v in sigma.items()})
    _, hands = _deal_many(btn_w, bb_w, n, seed)
    total = 0.0
    for h in hands:
        v = Game(h, sigma, opts, "eval", acc=acc).play(0, s_pot, stack)
        total += v
        acc.value_by_flop[h.tex[0]] += v
        acc.n_by_flop[h.tex[0]] += 1
        acc.n_flop_bucket[h.bb[0]] += 1
        acc.value_by_flop_bucket[h.bb[0]] += v
    return acc, total, len(hands)


def merge_policies(btn_sigma, bb_sigma):
    """One policy dict where the BTN's decisions come from btn_sigma and the BB's from bb_sigma."""
    return {k: (btn_sigma[k] if NODES[k[1]][0] == "btn" else bb_sigma[k]) for k in btn_sigma}


def _pool_map(fn, jobs):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(fn, jobs))


def learn(btn_range, bb_range, opts, s=2.25, stack=100, batches=150, hands_per_batch=30_000, seed=0,
          verbose=True, log_every=25):
    """Learn the whole postflop game. Returns the average strategy (dict of arrays) and a history."""
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    btn_w, bb_w = cw * btn_range, cw * bb_range
    shapes = table_shapes(opts)
    R = {k: np.zeros(v) for k, v in shapes.items()}
    S = {k: np.zeros(v) for k, v in shapes.items()}
    workers = os.cpu_count()
    history = []
    for t in range(1, batches + 1):
        jobs = [(btn_w, bb_w, hands_per_batch // workers, seed * 1_000_003 + t * 1000 + k, R, opts, float(t),
                 2 * s, stack - s) for k in range(workers)]
        for dR, dS in _pool_map(_train_worker, jobs):
            for key in R:
                R[key] += dR[key]
                S[key] += dS[key]
        for key in R:
            R[key] = np.maximum(R[key], 0)              # CFR+
        if verbose and (t % log_every == 0 or t == 1):
            avg = average(S)
            lead = avg[(0, "bb_first")][..., 1:].sum(-1).mean() if (0, "bb_first") in avg else 0.0
            cbet = avg[(0, "btn_vs_check")][..., 1:].sum(-1).mean()
            history.append({"batch": t, "flop_donk_avg_cell": lead, "flop_cbet_avg_cell": cbet})
            print(f"  batch {t}/{batches}: avg-cell flop c-bet {cbet:.0%}" +
                  (f", donk {lead:.0%}" if (0, "bb_first") in avg else ""), flush=True)
    return average(S), history


def average(S):
    out = {}
    for k, v in S.items():
        tot = v.sum(-1, keepdims=True)
        out[k] = np.where(tot > 0, v / np.where(tot > 0, tot, 1), 1 / v.shape[-1])
    return out


def evaluate(btn_range, bb_range, sigma, opts, s=2.25, stack=100, n=200_000, seed=12345):
    """Play n hands with the average strategy (exact expectation over both players' choices).

    Returns (BTN value per called pot in bb, merged EvalAcc).
    """
    cw = np.array([n_combos(h) for h in HANDS], dtype=float)
    workers = os.cpu_count()
    jobs = [(cw * btn_range, cw * bb_range, n // workers, seed + k, sigma, opts, 2 * s, stack - s)
            for k in range(workers)]
    parts = _pool_map(_eval_worker, jobs)
    acc = parts[0][0]
    for other, _, _ in parts[1:]:
        for name in ("strat", "reach", "avals"):
            for k in getattr(acc, name):
                getattr(acc, name)[k] += getattr(other, name)[k]
        for p in ("btn", "bb"):
            acc.aggr[p] += other.aggr[p]
            acc.decisions[p] += other.decisions[p]
        for name in ("callchain", "n_flop_bucket", "value_by_flop", "n_by_flop", "value_by_flop_bucket"):
            setattr(acc, name, getattr(acc, name) + getattr(other, name))
        for key, arr in other.lines.items():
            acc.lines[key] = acc.lines.get(key, 0) + arr
    total = sum(p[1] for p in parts)
    count = sum(p[2] for p in parts)
    return total / count, acc
