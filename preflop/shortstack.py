"""Short-stack preflop solver: the BTN may also limp or open all-in.

Game tree ("to" amounts in bb, effective stack S, blinds 0.5 / 1):

    BTN:  fold | limp | raise to s | all-in
      limp   -> BB: check (limped pot of 2, seen down with realization) | raise to r | all-in
                  raise  -> BTN: fold | call (pot 2r) | all-in -> BB: fold | call
                  all-in -> BTN: fold | call
      raise  -> BB: fold | call (pot 2s) | all-in -> BTN: fold | call
      all-in -> BB: fold | call

At 30bb and below the BB's best 3-bet is all-in (experiment 19), so that is its only 3-bet here.
Pots that see a flop use the same realization formula as solver.py; all-in pots use raw equity.
Solved with vectorised CFR+ over the 169 hand classes; values are the BTN's profit in bb per hand.
"""
import numpy as np

from .solver import _regret_match

BTN_ACTIONS = ("fold", "limp", "raise", "all-in")


class Node:
    def __init__(self, name, player, children):
        self.name, self.player, self.children = name, player, children   # children: {action: Node | matrix}


def build_tree(E, s, r, stack, r_ip=1.0, r_oop=1.0, alpha=1.0, limp=True, jam=True):
    x = E ** alpha * np.asarray(r_ip, float).reshape(-1, 1)
    y = (1 - E) ** alpha * np.asarray(r_oop, float).reshape(1, -1)
    share = x / (x + y)
    n = len(E)
    const = lambda v: np.full((n, n), float(v))
    seen = lambda pot_to: 2 * pot_to * share - pot_to       # BTN profit when both put in pot_to and see a flop
    allin = 2 * stack * E - stack

    vs_jam = lambda name, win: Node(name, "bb", {"fold": const(win), "call": allin})
    root = {"fold": const(-0.5)}
    if limp:
        vs_limp = {"check": seen(1.0)}
        if r < stack:
            vs_limp["raise"] = Node("btn_vs_iso", "btn", {"fold": const(-1), "call": seen(r),
                                                         "all-in": vs_jam("bb_vs_limp_reraise", r)})
        vs_limp["all-in"] = Node("btn_vs_limp_jam", "btn", {"fold": const(-1), "call": allin})
        root["limp"] = Node("bb_vs_limp", "bb", vs_limp)
    root["raise"] = Node("bb_vs_raise", "bb", {
        "fold": const(1), "call": seen(s),
        "all-in": Node("btn_vs_3bet", "btn", {"fold": const(-s), "call": allin})})
    if jam:
        root["all-in"] = vs_jam("bb_vs_open_jam", 1)
    return Node("btn_root", "btn", root)


def _nodes(node):
    yield node
    for c in node.children.values():
        if isinstance(c, Node):
            yield from _nodes(c)


class ShortResult:
    def __init__(self, root, W, strat):
        self.root, self.W, self.strat = root, W, strat     # strat: {node name: (n, k) average strategy}

    def actions(self, name):
        node = next(nd for nd in _nodes(self.root) if nd.name == name)
        return dict(zip(node.children, self.strat[name].T))

    def value(self, node=None):
        node = node or self.root
        sig = self.strat[node.name]
        V = 0
        for a, (act, c) in enumerate(node.children.items()):
            vc = self.value(c) if isinstance(c, Node) else c
            V = V + (sig[:, a][:, None] if node.player == "btn" else sig[:, a][None, :]) * vc
        return V

    def btn_ev(self):
        return float((self.W * self.value()).sum())

    def reach(self, name):
        """(BTN reach per hand, BB reach per hand) of a node under the average strategy."""
        def walk(node, pb, po):
            if node.name == name:
                return pb, po
            sig = self.strat[node.name]
            for a, c in enumerate(node.children.values()):
                if isinstance(c, Node):
                    nb, no = (pb * sig[:, a], po) if node.player == "btn" else (pb, po * sig[:, a])
                    found = walk(c, nb, no)
                    if found is not None:
                        return found
            return None
        n = len(self.W)
        return walk(self.root, np.ones(n), np.ones(n))

    def exploitability(self):
        """Average gain of a best response for each seat vs this strategy (bb/hand); 0 = equilibrium."""
        W = self.W

        def br(node, M, hero):
            # M: W weighted by the opponent's reach. Returns the hero's best-response value per hand.
            if not isinstance(node, Node):
                return (M * node).sum(1) if hero == "btn" else -(M * node).sum(0)
            sig = self.strat[node.name]
            kids = list(node.children.values())
            if node.player == hero:
                return np.max([br(c, M, hero) for c in kids], 0)
            if hero == "btn":
                return sum(br(c, M * sig[None, :, a], hero) for a, c in enumerate(kids))
            return sum(br(c, M * sig[:, a][:, None], hero) for a, c in enumerate(kids))

        game = self.btn_ev()
        return float((br(self.root, W, "btn").sum() - game + br(self.root, W, "bb").sum() + game) / 2)


def solve_short(E, W, s=2.0, r=3.0, stack=15, iters=3000, r_ip=1.0, r_oop=1.0, alpha=1.0, limp=True, jam=True):
    """Equilibrium of the short-stack tree for one open size s and one BB raise-vs-limp size r."""
    root = build_tree(E, s, r, stack, r_ip, r_oop, alpha, limp, jam)
    W = W / W.sum()
    n = len(W)
    nodes = list(_nodes(root))
    R = {nd.name: np.zeros((n, len(nd.children))) for nd in nodes}
    S = {nd.name: np.zeros((n, len(nd.children))) for nd in nodes}

    def value(node, sig):
        if not isinstance(node, Node):
            return node
        out, kids = 0, []
        for a, c in enumerate(node.children.values()):
            vc = value(c, sig)
            kids.append(vc)
            p = sig[node.name][:, a]
            out = out + (p[:, None] if node.player == "btn" else p[None, :]) * vc
        node._kids = kids
        return out

    def update(node, sig, pb, po, it):
        p = sig[node.name]
        if node.player == "btn":
            u = np.stack([(W * po[None, :] * vc).sum(1) for vc in node._kids], 1)
            own = pb
        else:
            u = -np.stack([(W * pb[:, None] * vc).sum(0) for vc in node._kids], 1)
            own = po
        R[node.name] = np.maximum(R[node.name] + u - (p * u).sum(1, keepdims=True), 0)    # CFR+
        S[node.name] += it * own[:, None] * p                                             # linear averaging
        for a, c in enumerate(node.children.values()):
            if isinstance(c, Node):
                if node.player == "btn":
                    update(c, sig, pb * p[:, a], po, it)
                else:
                    update(c, sig, pb, po * p[:, a], it)

    for it in range(1, iters + 1):
        sig = {k: _regret_match(v) for k, v in R.items()}
        value(root, sig)
        update(root, sig, np.ones(n), np.ones(n), it)
    return ShortResult(root, W, {k: _regret_match(v) for k, v in S.items()})
