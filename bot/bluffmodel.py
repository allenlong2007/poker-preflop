"""Experiment 41: a learned bluff predictor (logistic regression) for the opponent's postflop bets and raises.

What counts as a bluff: when the bet is made, the bettor has no pair beyond the board and no strong draw (flush draw
or open-ended straight draw). Features of a bet: street, size (share of the pot), raise or bet, how many times this
player already bet / raised this hand, board texture (flush possible, straight possible, paired), position.

  prior     fit on self-play bets where every hand's cards are known (the bot against itself and against a
            loose-aggressive version of itself): the structure (bigger bets, raises, later streets, scare boards)
  opponent  refit for each opponent on their own shown-down bets, as a MAP estimate with a Gaussian prior centred
            on the prior's weights (a tight prior on the slopes, a looser one on the bias): with a handful of
            showdowns it moves little, with many it becomes their own model

predict() gives P(bluff) for a bet; bot/potodds.py uses it as the bluff share of that bet's range.

Usage:
    python -m bot.bluffmodel prior [deals]        (self-play data and the prior fit; ~5 min)
    python -m bot.bluffmodel show <opponent id>
"""
import json
import sys
from pathlib import Path

import eval7
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "output" / "match" / "bluff_prior.json"
NAMES = ["bias", "turn", "river", "log size", "raise", "barrels", "flush possible", "straight possible", "paired",
         "in position"]
SLOPE_SD, BIAS_SD = 0.35, 1.0           # prior spread around the self-play weights for an opponent's refit
L2 = 1e-3


def _cards(s):
    return [eval7.Card(s[i:i + 2]) for i in range(0, len(s), 2)]


def features(street, frac, raise_, barrels, board, in_position):
    suits = [c.suit for c in board]
    ranks = sorted({c.rank for c in board})
    flush = max(suits.count(x) for x in set(suits)) >= 3
    straight = any(sum(1 for r in ranks if lo <= r <= lo + 4) >= 3 for lo in range(-1, 9)) or \
        (12 in ranks and sum(1 for r in ranks if r <= 3) >= 2)
    paired = len(ranks) < len(board)
    return np.array([1.0, street == 2, street == 3, np.log(max(frac, 0.05)), raise_, barrels, flush, straight,
                     paired, in_position], float)


def is_bluff(hole, board):
    from preflop import fullgame as fg
    made = (eval7.evaluate(list(hole) + board) >> 24) - (eval7.evaluate(board) >> 24)
    if made >= 1:
        return 0.0
    return 0.0 if len(board) < 5 and fg._has_draw(tuple(hole), board) else 1.0


def examples_from_record(r, who):
    """(features, label) for every bet / raise by `who` ("opp" or "hero") whose cards are known."""
    pos = r["opp_pos"] if who == "opp" else r["hero_pos"]
    hole = r["shown"].get(who) if who == "opp" else (r.get("hero_hole") or r["shown"].get("hero"))
    if not hole or not r.get("board"):
        return []
    hole, board = _cards(hole), _cards(r["board"])
    out = []
    pot = _preflop_pot(r["history"][0])
    barrels = 0
    for s, street in enumerate(r["history"][1:], start=1):
        b = board[:2 + s]
        if len(b) < 2 + s:
            break
        put = {"btn": 0.0, "bb": 0.0}
        for p, k, to in street:
            if k in ("bet", "raise") and to:
                if p == pos:
                    prev = max(v for q, v in put.items() if q != p)
                    frac = (to - prev) / max(pot + 2 * prev, 1e-9) if k == "raise" else to / pot
                    out.append((features(s, frac, k == "raise", barrels, b, pos == "btn"), is_bluff(hole, b)))
                    barrels += 1
                put[p] = to
            elif k == "call":
                put[p] = max(put.values())
        pot += sum(put.values())
    return out


def _preflop_pot(pre):
    put = {"btn": 0.5, "bb": 1.0}
    for p, k, to in pre:
        if k == "raise" and to:
            put[p] = to
        elif k == "call":
            put[p] = max(put.values())
    return sum(put.values())


def fit(X, y, w0=None, sd=None, iters=400):
    """Logistic regression, MAP with a Gaussian prior N(w0, sd^2) (or plain L2 when w0 is None); Newton steps."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    d = X.shape[1]
    w0 = np.zeros(d) if w0 is None else np.asarray(w0, float)
    prec = np.full(d, L2) if sd is None else 1.0 / np.asarray(sd, float) ** 2
    w = w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)) if len(X) else np.zeros(0)
        g = (X.T @ (p - y) if len(X) else 0) + prec * (w - w0)
        H = (X.T * (p * (1 - p))) @ X if len(X) else np.zeros((d, d))
        H = H + np.diag(prec)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return w


class Model:
    def __init__(self, w):
        self.w = np.asarray(w, float)

    def predict(self, street, frac, raise_, barrels, board, in_position):
        x = features(street, frac, raise_, barrels, board, in_position)
        return float(1 / (1 + np.exp(-x @ self.w)))


def prior():
    return Model(json.load(open(PRIOR))["w"]) if PRIOR.exists() else None


def for_opponent(hands):
    """The prior refit on this opponent's shown-down bets; (model, number of labeled bets)."""
    p = prior()
    if p is None:
        return None, 0
    # Shown cards are selected outcomes. Keep the frozen prior in production;
    # opponent refits are evaluated separately in calibration_report().
    data = [e for r in hands for e in examples_from_record(r, "opp")]
    return p, len(data)


def calibration_report(hands):
    """Shadow-only chronological holdout, one loss observation per revealed hand.

    Brier loss measures probabilistic accuracy (calibration AND discrimination).
    Improvement intervals use bounded per-hand paired losses, never independent
    street labels. A fixed first-half fit cannot see the later validation labels.
    Selected reveals cannot qualify this model for full-range live serving.
    """
    from bot.reads import MIN_HANDS, interval, unique_hands
    p = prior()
    report = {"applied": False, "reason": "selected_reveals_not_range_calibrated",
              "train_hands": 0, "validation_hands": 0,
              "metric": "per_hand_mean_brier", "sampling": "opponent_selected_reveals"}
    if p is None:
        return dict(report, reason="missing_prior")
    hands = unique_hands(hands)
    # Archive order is temporal only within one monotonically numbered source.
    numbers = [h.get("number") for h in hands]
    if (len({h["opp_id"] for h in hands}) > 1
            or any(not isinstance(n, (int, float)) for n in numbers)
            or any(b <= a for a, b in zip(numbers, numbers[1:]))):
        return dict(report, reason="chronology_unverified")
    labeled = [(r, examples_from_record(r, "opp")) for r in hands]
    labeled = [(r, data) for r, data in labeled if data]
    split = len(labeled) // 2
    train, validation = labeled[:split], labeled[split:]
    report.update(train_hands=len(train), validation_hands=len(validation))
    if min(len(train), len(validation)) < MIN_HANDS:
        return dict(report, reason="too_few_independent_reveals")
    # Weight each training hand equally, using one deterministic first bet label.
    X, y = zip(*(data[0] for _, data in train))
    sd = np.full(len(p.w), SLOPE_SD)
    sd[0] = BIAS_SD
    candidate = Model(fit(X, y, p.w, sd))
    prior_loss, candidate_loss, prior_residual, candidate_residual = [], [], [], []
    for _, data in validation:
        Xv, yv = map(np.asarray, zip(*data))
        pp = 1 / (1 + np.exp(-Xv @ p.w))
        pc = 1 / (1 + np.exp(-Xv @ candidate.w))
        prior_loss.append(float(np.mean((pp - yv) ** 2)))
        candidate_loss.append(float(np.mean((pc - yv) ** 2)))
        prior_residual.append(float(np.mean(pp - yv)))
        candidate_residual.append(float(np.mean(pc - yv)))
    diff = np.asarray(prior_loss) - candidate_loss
    # Paired loss differences lie in [-1, 1]; rescale to [0, 1].
    lo, hi = interval(float(((diff + 1) / 2).sum()), len(diff))
    report.update(prior_brier=float(np.mean(prior_loss)), candidate_brier=float(np.mean(candidate_loss)),
                  prior_mean_residual=float(np.mean(prior_residual)),
                  candidate_mean_residual=float(np.mean(candidate_residual)),
                  brier_improvement=float(diff.mean()), brier_improvement_interval=[2 * lo - 1, 2 * hi - 1],
                  improvement_supported=lo > .5)
    return report


def build_prior(deals):
    from bot.engine import Engine
    from bot.selfplay import DECK, EngineAgent, play_hand

    class Rec:
        def __init__(self, agent):
            self.a, self.h = agent, None

        def act(self, seat, hole, board, history, facing):
            self.h = history
            return self.a.act(seat, hole, board, history, facing)

    rng = np.random.default_rng(21)
    X, y = [], []
    # Near-equilibrium play only: the loose-aggressive test player's extra bets are random hands, which would
    # teach that big bets and raises are bluffs, the opposite of how people play.
    pairs = [(EngineAgent(Engine(seed=31)), EngineAgent(Engine(seed=32))), (EngineAgent(Engine(seed=33)), EngineAgent(Engine(seed=34)))]
    for k in range(deals):
        a, b = (Rec(x) for x in pairs[k % 2])
        c = [DECK[i] for i in rng.choice(52, size=9, replace=False)]
        play_hand({"btn": a, "bb": b}, {"btn": c[:2], "bb": c[2:4]}, c[4:])
        hist = a.h if a.h is not None else b.h
        if hist is None:
            continue
        board = "".join(str(x) for x in c[4:])
        hh = [[[x.player, x.kind, x.to] for x in st] for st in hist]
        for pos, hole in (("btn", c[:2]), ("bb", c[2:4])):
            r = {"opp_pos": pos, "hero_pos": "bb" if pos == "btn" else "btn", "history": hh, "board": board,
                 "shown": {"opp": "".join(str(x) for x in hole)}}
            for xx, yy in examples_from_record(r, "opp"):
                X.append(xx)
                y.append(yy)
    w = fit(X, y)
    PRIOR.write_text(json.dumps({"w": w.tolist(), "names": NAMES, "n": len(y), "bluff_rate": float(np.mean(y))},
                                indent=1))
    return w, len(y), float(np.mean(y))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "prior":
        w, n, rate = build_prior(int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
        print(f"{n} self-play bets, {rate:.0%} bluffs; weights:")
        for k, v in zip(NAMES, w):
            print(f"  {k:18} {v:+.3f}")
        return
    from bot.opponent import load
    m, n = for_opponent(load(sys.argv[2]))
    p = prior()
    b = _cards("Kh8d3c2s5h")
    print(f"{n} shown-down bets from this opponent")
    for street, frac, rz in ((3, 0.33, 0), (3, 0.75, 0), (3, 1.5, 0), (3, 1.0, 1), (2, 0.75, 0), (1, 0.33, 0)):
        bb = b[:2 + street]
        print(f"  street {street} {'raise' if rz else 'bet'} {frac:.2f} pot: P(bluff) prior {p.predict(street, frac, rz, 1, bb, True):.0%}"
              f" -> this opponent {m.predict(street, frac, rz, 1, bb, True):.0%}")


if __name__ == "__main__":
    main()
