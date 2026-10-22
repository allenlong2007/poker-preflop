"""Experiment 37: adapt to the opponent from the hands stored by bot/opponent.py.

Preflop (a restricted Nash response, Johanson et al. 2007): the opponent's observed frequencies at each preflop
node become a fixed strategy (their most aggressive action goes to their strongest hands, then calls, then
folds, among the hands that reach the node). The preflop tree is solved again with the opponent playing that
fixed strategy with probability p = n / (n + N_CONF) (at most P_MAX) at each node and best-responding otherwise.
The bot's answer exploits what was seen in proportion to the evidence and stays robust where the read is
thin or wrong; a pure best response to a noisy read (tried first) bluffs into lines that only work if the read
is exact. The bot has no all-in options in these solves: at 100bb it adapts with normal sizes, never by shoving.

Postflop (scaled, not solved): compared with the bot's own self-play profile (the baseline),
    bluff   = (their fold rate to bets / baseline)  ->  more or fewer bluffs with weak hands
    defend  = (their betting rate / baseline)       ->  call down wider against an aggressive opponent, fold
                                                        more marginal hands against a passive one
both shrunk toward 1 by sample size and clipped to [0.5, 2].

Usage:
    python -m bot.adapt baseline [deals]       (self-play profile for comparison; ~1 min)
    python -m bot.adapt show [opponent id]     (what the bot would change against them)
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np

from bot.opponent import load, profile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "output" / "match" / "baseline_profile.json"
SHRINK = 20            # postflop: pseudo-hands of baseline behaviour added to each observed rate
N_CONF, P_MAX = 40, 0.85   # preflop: confidence in the read at a node after n observations, and its cap
PRIOR = 2              # preflop: pseudo-observations of equilibrium play smoothing each observed node
MIN_NEW = 3            # re-solve preflop after this many new hands

# Opponent preflop node (opponent.profile name) -> (tree node, {their action: tree action}).
PRE_NODES = {
    "btn_first_in": ("btn_root", {"fold": "fold", "limp": "limp", "raise": "raise", "all-in": "all-in"}),
    "bb_vs_limp": ("bb_vs_limp", {"check": "check", "raise": "raise", "all-in": "all-in"}),
    "bb_vs_open": ("bb_vs_raise", {"fold": "fold", "call": "call", "raise": "3-bet", "all-in": "all-in"}),
    "btn_vs_iso": ("btn_vs_iso", {"fold": "fold", "call": "call", "raise": "re-raise", "all-in": "all-in"}),
    "btn_vs_3bet": ("btn_vs_3bet_small", {"fold": "fold", "call": "call", "raise": "4-bet", "all-in": "all-in"}),
    "bb_vs_4bet": ("bb_vs_4bet_small", {"fold": "fold", "call": "call", "raise": "all-in", "all-in": "all-in"}),
}
AGGRESSION = ("all-in", "4-bet", "3-bet", "re-raise", "raise", "call", "limp", "check", "fold")
POST_FOLD = ("flop_vs_bet", "turn_vs_bet", "river_vs_bet")
POST_BET = ("flop_bet_when_checked_to", "turn_bet_when_checked_to", "river_bet_when_checked_to",
            "flop_lead", "turn_lead", "river_lead", "flop_cbet")


def _pooled(nodes, names, kind):
    n = sum(nodes.get(k, {}).get("n", 0) for k in names)
    hits = sum(nodes.get(k, {}).get("n", 0) * nodes.get(k, {}).get(kind, 0) for k in names)
    return hits, n


def _ratio(obs, base, names, kind):
    h, n = _pooled(obs, names, kind)
    hb, nb = _pooled(base, names, kind)
    if not n or not nb or hb <= 0:
        return 1.0, n
    f_base = hb / nb
    f = (h + SHRINK * f_base) / (n + SHRINK)                  # shrunk toward the baseline
    return float(np.clip(f / f_base, 0.5, 2.0)), n


def postflop_exploit(prof, base):
    bluff, n_f = _ratio(prof["nodes"], base["nodes"], POST_FOLD, "fold")
    defend, n_b = _ratio(prof["nodes"], base["nodes"], POST_BET, "bet")
    return {"bluff": round(bluff, 3), "defend": round(defend, 3), "n_fold": n_f, "n_bet": n_b}


LOOSE_SHARE = 0.5      # of an action's excess over equilibrium, the share played by any hand alike
BAND_BLUFF = {"small": 0.20, "medium": 0.28, "big": 0.36}   # bluff share at showdown we'd expect by default
SIZE_SHRINK = 20


def size_value(prof):
    """Experiment 39: scale of the opponent's bluff share by bet size, from the bets they showed down; feeds the
    pot-odds range (bot/potodds.py). 1 = as expected; 0.5 = half the bluffs (fold more); 2 = twice (call more)."""
    nodes = prof["nodes"]
    scale = {}
    for band, base in BAND_BLUFF.items():
        sh = nodes.get(f"shown_{band}", {})
        n = sh.get("n", 0)
        bluffs = n * sh.get("bluff", 0.0)
        scale[band] = float(np.clip((bluffs + SIZE_SHRINK * base) / (n + SIZE_SHRINK) / base, 0.5, 2.0))
    from bot.opponent import size_band
    return lambda frac: scale[size_band(frac)], scale


# Experiment 42: strictness = clip(S0 - K * ln(aggression), 0, 1) where aggression is their postflop raise and
# bet rates relative to the bot's own self-play (each shrunk by SHRINK pseudo-hands toward it). Trained by
# scripts/strictness_train.py over a population of opponents.
S0, K = 1.0, 1.0


def strictness(prof, base, s0=None, k=None):
    s0, k = S0 if s0 is None else s0, K if k is None else k
    def ratio(names, kind):
        h, n = _pooled(prof["nodes"], names, kind)
        hb, nb = _pooled(base["nodes"], names, kind)
        if not nb or hb <= 0:
            return 1.0
        f_base = hb / nb
        return float(np.clip((h + SHRINK * f_base) / (n + SHRINK) / f_base, 0.25, 4.0))
    agg = np.sqrt(ratio(POST_FOLD, "raise") * ratio(POST_BET, "bet"))
    return float(np.clip(s0 - k * np.log(agg), 0.0, 1.0))


LOOSE_BLUFF, PRIOR_WEIGHT = 0.35, 5.0   # experiment 42: prior bluff share of an opponent's bets, worth 5 shown bets


def evidence_strictness(hands, prof=None, base=None):
    """Experiment 42: trust in the value-heavy reading of this opponent's bets and raises, from what they showed
    down. Their bluff share among shown bets / raises is estimated with a prior of LOOSE_BLUFF worth PRIOR_WEIGHT
    bets; strictness = how far it sits below LOOSE_BLUFF (0 with no showdowns or when they show bluffs; 0.5 after
    five value showdowns; ~0.7 after ten)."""
    from bot import bluffmodel
    labels = [y for r in hands for _, y in bluffmodel.examples_from_record(r, "opp")]
    est = (sum(labels) + LOOSE_BLUFF * PRIOR_WEIGHT) / (len(labels) + PRIOR_WEIGHT)
    s = float(np.clip(1 - est / LOOSE_BLUFF, 0.0, 1.0))
    if prof is not None and base is not None:
        # Capped by their measured aggression: a player betting / raising far above normal can't be read as
        # value-heavy off a few (self-selected) showdowns: at twice the normal rate the cap is 0.31, at e times 0.
        s = min(s, strictness(prof, base, 1.0, 1.0))
    return s


class Preflop:
    """Equilibrium and locked re-solves of the main 100bb preflop spot."""
    def __init__(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import match_strategy as ms
        from preflop.shortstack import solve_short
        self.ms, self.solve_short = ms, solve_short
        pre = json.load(open(ms.POLICY / "preflop.json"))
        self.s, self.t, self.r = pre["open"], pre["threebet_to"], pre["iso_to"]
        self.w = np.array([ms.n_combos(h) for h in ms.HANDS], float)
        self.strength = (ms.E * self.w[None, :]).sum(1) / self.w.sum()
        self.eq = self._solve()

    def _solve(self, lock=None, iters=2000, no_jam=None):
        ms = self.ms
        return self.solve_short(ms.E, ms.W, self.s, self.r, ms.STACK, iters, r_oop=ms.R_OOP, alpha=ms.ALPHA, t3=self.t,
                                adj_ip=ms.A_IP, adj_oop=ms.A_OOP, levels=ms.levels_for(self.s), f4=ms.F4,
                                jam4_depth=ms.JAM4_DEPTH, lock=lock, no_jam=no_jam)

    def lock_for(self, node, who, freqs):
        """Per-hand strategy at `node` giving these action frequencies over the hands that reach it. Up to the
        equilibrium's own frequency an aggressive action goes to the strongest hands (then calls, then folds);
        of the aggression beyond it (e.g. shoving 11% where equilibrium shoves 0%), LOOSE_SHARE is spread evenly
        over the whole range and the rest goes to the strongest hands: over-aggression is partly loose, not only
        premiums (one shown 100bb shove was 22)."""
        reach = self.eq.reach(node)[0 if who == "btn" else 1] * self.w
        acts = list(self.eq.actions(node))
        order = np.argsort(-self.strength)
        total = reach.sum()
        out = {a: np.zeros(len(self.w)) for a in acts}
        ranked = [a for a in AGGRESSION if a in acts and freqs.get(a, 0) > 0]
        if total <= 1e-9 or not ranked:
            return None
        eq_f = {a: float(np.dot(reach, v) / total) for a, v in self.eq.actions(node).items()}
        loose = {a: LOOSE_SHARE * max(0.0, freqs[a] - eq_f.get(a, 0.0)) for a in ranked
                 if a in ("all-in", "4-bet", "3-bet", "re-raise", "raise")}
        spread = sum(loose.values())
        for a, x in loose.items():
            out[a] += x                                           # the same share of every hand
        freqs = {a: freqs[a] - loose.get(a, 0.0) for a in ranked}
        reach = reach * (1 - spread)
        quota = {a: freqs[a] * total for a in ranked}
        k = 0
        for i in order:
            left = reach[i]
            while left > 1e-12 and k < len(ranked):
                take = min(left, quota[ranked[k]])
                out[ranked[k]][i] += take / max(reach[i], 1e-12)
                quota[ranked[k]] -= take
                left -= take
                if quota[ranked[k]] <= 1e-12:
                    k += 1
            if reach[i] <= 1e-12:                                 # unreachable: keep the equilibrium action
                for a in acts:
                    out[a][i] = self.eq.actions(node)[a][i]
            elif left > 1e-12:
                out[ranked[-1]][i] += left / reach[i]
        return out

    def adapt(self, prof):
        """Best-response nodes for the bot, and the blend weight, against this opponent profile."""
        lock, used = {}, {}
        n_total = 0
        for name, (node, amap) in PRE_NODES.items():
            obs = prof["nodes"].get(name)
            if not obs or node not in self.eq.strat:
                continue
            who = "btn" if node.startswith("btn") else "bb"
            reach = self.eq.reach(node)[0 if who == "btn" else 1] * self.w
            eq_f = {a: float(np.dot(reach, v) / max(reach.sum(), 1e-12)) for a, v in self.eq.actions(node).items()}
            n = obs["n"]
            f = {a: eq_f[a] * PRIOR / (n + PRIOR) for a in eq_f}
            for k, a in amap.items():
                if k in obs and a in f:
                    f[a] += obs[k] * n / (n + PRIOR)
            tot = sum(f.values())
            f = {a: v / tot for a, v in f.items()}
            lk = self.lock_for(node, who, f)
            if lk is not None:
                conf = min(P_MAX, n / (n + N_CONF))
                lock[node] = (lk, conf)
                used[node] = {"n": n, "confidence": round(conf, 2), "freqs": {a: round(v, 3) for a, v in f.items()},
                              "equilibrium": {a: round(v, 3) for a, v in eq_f.items()}}
                n_total = max(n_total, n)
        if not lock:
            return None
        # The tree has both seats: lock the opponent's BTN nodes to get the bot's BB answers, and their BB
        # nodes to get the bot's BTN answers (two solves).
        mine, evs = {}, {}
        for opp_seat in ("btn", "bb"):
            part = {k: v for k, v in lock.items() if k.startswith(opp_seat)}
            if not part:
                continue
            res = self._solve(part, no_jam="bb" if opp_seat == "btn" else "btn")
            evs[opp_seat] = res.btn_ev()
            mine |= {name: {a: v.tolist() for a, v in res.actions(name).items()}
                     for name in res.strat if not name.startswith(opp_seat)}
        lam = 1.0                       # the restricted Nash response already weighs the evidence
        return {"nodes": mine, "lam": lam, "locked": used, "btn_ev_eq": self.eq.btn_ev(), "btn_ev_br": evs}


class Adapter:
    """Keeps the engine's adaptation up to date in a background thread as hands are stored."""
    def __init__(self, engine):
        self.engine = engine
        self.base = json.load(open(BASELINE)) if BASELINE.exists() else None
        self.pre = None
        self.done = {}                        # opponent id -> hands count at the last adaptation
        self.lock = threading.Lock()
        self.state = {}
        self.states = {}                      # opponent id -> what the engine should use against them

    def apply(self, opp_id):
        """Load this opponent's adaptation into the engine (defaults if there is none yet). The caller holds the
        decision lock, so two tables never interleave."""
        from bot.engine import STRICT_PRIOR
        from bot.opponent import canonical
        st = self.states.get(canonical(opp_id)) if opp_id else None
        e = self.engine
        if st is None:
            e.set_adaptation({}, 0.0, {"bluff": 1.0, "defend": 1.0})
            e.opp_profile, e.size_value, e.bluff_model, e.strict = None, None, None, STRICT_PRIOR
            return
        e.set_adaptation(st["nodes"], st["lam"], st["post"])
        e.opp_profile, e.size_value, e.strict = st["profile"], st["size_value"], st["strict"]
        e.bluff_model = st["bluff_model"]

    def update(self, opp_id, force=False):
        from bot.opponent import canonical
        opp_id = canonical(opp_id)
        hands = load(opp_id)
        if not hands or (not force and len(hands) - self.done.get(opp_id, -MIN_NEW) < MIN_NEW):
            return
        self.done[opp_id] = len(hands)
        threading.Thread(target=self._run, args=(opp_id, hands), daemon=True).start()

    def _run(self, opp_id, hands):
        if not self.lock.acquire(blocking=False):
            self.done[opp_id] -= MIN_NEW                       # busy: try again on the next hand
            return
        try:
            prof = profile(hands)
            if self.pre is None:
                self.pre = Preflop()
            pre = self.pre.adapt(prof)
            post = postflop_exploit(prof, self.base) if self.base else {"bluff": 1.0, "defend": 1.0}
            fn, scale = size_value(prof)
            from bot import bluffmodel
            bm, n_shown = bluffmodel.for_opponent(hands)        # experiment 41: retrained on every update
            # Kept per opponent (several tables at once) and loaded into the engine before each of their decisions.
            self.states[opp_id] = {
                "nodes": pre["nodes"] if pre else {}, "lam": pre["lam"] if pre else 0.0, "post": post,
                "profile": prof, "size_value": fn, "bluff_model": bm,
                "strict": evidence_strictness(hands, prof, self.base) if self.base else 0.0}
            self.state = {"opponent": opp_id, "hands": len(hands), "lam": round(pre["lam"], 3) if pre else 0,
                          "postflop": dict(post, sizes=scale), "locked": pre["locked"] if pre else {}}
        finally:
            self.lock.release()


def baseline(deals):
    """The bot's own profile in self-play (both seats), the reference for the postflop ratios."""
    from bot.engine import Engine
    from bot.selfplay import DECK, play_hand
    rng = np.random.default_rng(11)

    class Rec:
        def __init__(self, eng):
            self.eng, self.hist = eng, None

        def act(self, seat, hole, board, history, facing):
            self.hist = history
            d = self.eng.decide(seat, hole, board, history)
            return d.action, d.to

    a, b = Rec(Engine(seed=3)), Rec(Engine(seed=4))
    recs = []
    for k in range(deals):
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        play_hand({"btn": a, "bb": b}, {"btn": c[:2], "bb": c[2:4]}, c[4:])
        hist = [[[x.player, x.kind, x.to] for x in st] for st in (a.hist or b.hist)]
        for opp in ("btn", "bb"):
            recs.append({"number": k, "opp_pos": opp, "hero_pos": "bb" if opp == "btn" else "btn", "history": hist,
                         "stacks": {"hero": 100, "opp": 100}, "shown": {}, "board": ""})
    prof = profile(recs)
    prof["shown"] = []
    BASELINE.write_text(json.dumps(prof, indent=1))
    return prof


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "baseline":
        p = baseline(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
        print(json.dumps({k: v for k, v in p["nodes"].items()}, indent=0)[:2000])
        return
    prof = profile(load(sys.argv[2] if len(sys.argv) > 2 else None))
    base = json.load(open(BASELINE))
    pre = Preflop().adapt(prof)
    print(f"{prof['hands']} hands" if pre else "no preflop data")
    if pre:
        for node, u in pre["locked"].items():
            print(f"  opponent at {node} (n={u['n']}, confidence {u['confidence']:.0%}): read {u['freqs']}  vs equilibrium {u['equilibrium']}")
        w = np.array([1.0])
        pp = Preflop()
        w = pp.w
        for node in ("btn_root", "bb_vs_limp", "bb_vs_raise", "btn_vs_iso", "btn_vs_3bet_small"):
            if node in pre["nodes"]:
                who = 0 if node.startswith("btn") else 1
                rw = pp.eq.reach(node)[who] * w
                f = {a: f"{np.dot(rw, v) / rw.sum():.0%}" for a, v in pre["nodes"][node].items()}
                e = {a: f"{np.dot(rw, v) / rw.sum():.0%}" for a, v in pp.eq.actions(node).items()}
                print(f"  bot at {node}: best response {f}  (equilibrium {e})")
        print(f"  model BTN EV: equilibrium {pre['btn_ev_eq'] * 100:+.2f} bb/100")
    print("postflop:", postflop_exploit(prof, base), "| bluff share by size vs expected:", size_value(prof)[1])


if __name__ == "__main__":
    main()
