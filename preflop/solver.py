"""Heads-up preflop solver: Button (small blind) vs Big Blind, equity-only payoffs.

Game tree (all sizes are "raise TO" amounts in big blinds, stacks = `stack`):

    BTN:  fold | raise to s                     (never limp)
    BB:   fold | call | 3-bet to t
    BTN:  fold | call | 4-bet all-in             (only after a 3-bet)
    BB:   fold | call                            (only after the all-in)

Whenever the hand reaches a call, we assume the cards are dealt out with no more
betting and each player wins pot x equity. That is the big simplification:
real postflop play (position, bluffs, folding equity out) is ignored.

Solved with CFR+ (counterfactual regret minimisation), vectorised over the
169 hand classes with numpy. Every number is BTN's net profit in bb per hand.
"""
import numpy as np


def _regret_match(R):
    pos = np.maximum(R, 0)
    tot = pos.sum(1, keepdims=True)
    n = R.shape[1]
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1 / n)


class Payoffs:
    def __init__(self, E, s, t, stack, r_ip=1.0, r_oop=1.0):
        self.s, self.t, self.stack = s, t, stack
        # Equity realization: in a pot that still has postflop play, the BTN (in
        # position) wins a bit more than its raw equity and the BB a bit less.
        # BTN's share of the pot = E*r_ip / (E*r_ip + (1-E)*r_oop). 1.0/1.0 = raw equity.
        share = E * r_ip / (E * r_ip + (1 - E) * r_oop)
        self.call_open = 2 * s * share - s      # BB calls the open
        self.call_3bet = 2 * t * share - t      # BTN calls the 3-bet
        self.call_jam = 2 * stack * E - stack   # BB calls the all-in (no postflop, raw equity)


def _values(P, s2, s3, s4):
    """BTN's value matrices [i, j] at the BB-jam, BTN-3bet and BB-open nodes."""
    V4 = s4[None, :, 0] * P.t + s4[None, :, 1] * P.call_jam
    V3 = s3[:, None, 0] * -P.s + s3[:, None, 1] * P.call_3bet + s3[:, None, 2] * V4
    V2 = s2[None, :, 0] * 1.0 + s2[None, :, 1] * P.call_open + s2[None, :, 2] * V3
    return V2, V3, V4


def solve(E, W, open_size=2.25, threebet_size=None, stack=100, iters=3000, r_ip=1.0, r_oop=1.0):
    """Return the equilibrium strategy for one open size.

    open_size     -- BTN raises to this many bb
    threebet_size -- BB re-raises to this (default 4x the open)
    r_ip, r_oop   -- equity realization for BTN / BB in non-all-in pots (1.0 = raw equity)
    """
    t = threebet_size or 4 * open_size
    P = Payoffs(E, open_size, t, stack, r_ip, r_oop)
    W = W / W.sum()                 # joint probability of each (BTN, BB) class pair
    n = len(W)

    R = [np.zeros((n, k)) for k in (2, 3, 3, 2)]     # regrets per decision node
    S = [np.zeros((n, k)) for k in (2, 3, 3, 2)]     # cumulative average strategy
    for it in range(1, iters + 1):
        s1, s2, s3, s4 = (_regret_match(r) for r in R)
        V2, V3, V4 = _values(P, s2, s3, s4)

        # Action values at each node, weighted by how often the opponent gets there.
        u1 = np.stack([-0.5 * W.sum(1), (W * V2).sum(1)], 1)
        M2 = W * s1[:, 1:2]
        u2 = -np.stack([M2.sum(0), (M2 * P.call_open).sum(0), (M2 * V3).sum(0)], 1)
        M3 = W * s2[None, :, 2]
        u3 = np.stack([-P.s * M3.sum(1), (M3 * P.call_3bet).sum(1), (M3 * V4).sum(1)], 1)
        M4 = W * (s1[:, 1] * s3[:, 2])[:, None]
        u4 = -np.stack([P.t * M4.sum(0), (M4 * P.call_jam).sum(0)], 1)

        own_reach = [np.ones(n), np.ones(n), s1[:, 1], s2[:, 2]]
        for k, (u, sig) in enumerate(zip((u1, u2, u3, u4), (s1, s2, s3, s4))):
            R[k] = np.maximum(R[k] + u - (sig * u).sum(1, keepdims=True), 0)   # CFR+
            S[k] += it * own_reach[k][:, None] * sig                          # linear averaging

    avg = [_regret_match(s) for s in S]
    return Result(P, W, *avg)


class Result:
    def __init__(self, P, W, btn_open, bb_vs_open, btn_vs_3bet, bb_vs_jam):
        self.P, self.W = P, W
        self.btn_open = btn_open        # [fold, raise]
        self.bb_vs_open = bb_vs_open    # [fold, call, 3bet]
        self.btn_vs_3bet = btn_vs_3bet  # [fold, call, jam]
        self.bb_vs_jam = bb_vs_jam      # [fold, call]

    def btn_ev(self):
        """BTN's expected profit in bb per hand when both sides play this strategy."""
        V2, _, _ = _values(self.P, self.bb_vs_open, self.btn_vs_3bet, self.bb_vs_jam)
        V1 = self.btn_open[:, None, 0] * -0.5 + self.btn_open[:, None, 1] * V2
        return float((self.W * V1).sum())

    def exploitability(self):
        """How much a perfect counter-strategy wins vs us, averaged over both seats (bb/hand).

        0 means a true equilibrium; it shrinks as CFR runs longer.
        """
        P, W = self.P, self.W
        s1, s2, s3, s4 = self.btn_open, self.bb_vs_open, self.btn_vs_3bet, self.bb_vs_jam
        _, _, V4 = _values(P, s2, s3, s4)
        # Best-responding BTN.
        M3 = W * s2[None, :, 2]
        best3 = np.max(np.stack([-P.s * M3.sum(1), (M3 * P.call_3bet).sum(1), (M3 * V4).sum(1)]), 0)
        raise_v = (W * (s2[None, :, 0] + s2[None, :, 1] * P.call_open)).sum(1) + best3
        br_btn = np.maximum(-0.5 * W.sum(1), raise_v).sum()
        # Best-responding BB (values from BB's side).
        M2 = W * s1[:, 1:2]
        fold_to_3b = s3[:, None, 0] * -P.s + s3[:, None, 1] * P.call_3bet
        M4 = M2 * s3[:, None, 2]
        jam_v = np.maximum(-P.t * M4.sum(0), -(M4 * P.call_jam).sum(0))
        threebet_v = -(M2 * fold_to_3b).sum(0) + jam_v
        br_bb = (0.5 * W * s1[:, None, 0]).sum(0) + np.max(
            np.stack([-M2.sum(0), -(M2 * P.call_open).sum(0), threebet_v]), 0)
        return float((br_btn + br_bb.sum()) / 2)
