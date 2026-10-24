"""Conservative, time-uniform gates for observed action frequencies.

One bounded observation per distinct hand/node, including repeated actions. The
Hoeffding bound spends alpha/(n*(n+1)) at look n and alpha/128 per channel.
Coverage assumes a stable conditional mean across hands; it is not a poker EV
or calibration guarantee. Showdown-selected labels never authorize live reads.
"""
from collections import defaultdict
from math import isfinite, log, sqrt

VERSION = 1
ALPHA = .05
CHANNELS = 128
MIN_HANDS = 40
MAX_WIDTH = .35
MARGIN = .05
PRE = {
    "btn_first_in": ("fold", "limp", "raise", "all-in"),
    "bb_vs_limp": ("check", "raise", "all-in"),
    "bb_vs_open": ("fold", "call", "raise", "all-in"),
    "btn_vs_iso": ("fold", "call", "raise", "all-in"),
    "btn_vs_3bet": ("fold", "call", "raise", "all-in"),
    "bb_vs_4bet": ("fold", "call", "raise", "all-in"),
}
ACTIONS = dict(PRE)
for _street in ("flop", "turn", "river"):
    ACTIONS[f"{_street}_vs_bet"] = ("fold", "call", "raise")
    for _node in ("lead", "bet_when_checked_to"):
        ACTIONS[f"{_street}_{_node}"] = ("bet", "check")
ACTIONS["flop_cbet"] = ("bet", "check")
assert sum(map(len, ACTIONS.values())) <= CHANNELS


def interval(total: float, n: int, allocation: int = 1) -> tuple[float, float]:
    """Simultaneous across registered channels and every positive sample count."""
    if n <= 0:
        return 0.0, 1.0
    mean = total / n
    if not isfinite(mean) or not 0 <= mean <= 1:
        raise ValueError("read observations must be bounded in [0, 1]")
    radius = sqrt(log(2 * CHANNELS * allocation * n * (n + 1) / ALPHA) / (2 * n))
    return max(0.0, mean - radius), min(1.0, mean + radius)


def jam_key(node: str, stack: float, prefix: list) -> str:
    """Exact effective depth and action amounts; no near-jam labels or size bins."""
    import json
    actions = [(p, k, float(to or 0).hex() if k == "raise" else "0") for p, k, to in prefix]
    return json.dumps([node, float(stack).hex(), actions], separators=(",", ":"))


def unique_hands(hands: list[dict]) -> list[dict]:
    """Fail closed for absent identity; deduplicate by archive's hero/hand key."""
    from bot.opponent import canonical
    out, seen, opponents = [], set(), set()
    for h in hands:
        identity = h.get("id")
        if identity is None or not h.get("opp_id"):
            continue
        opponents.add(canonical(h["opp_id"]))
        if len(opponents) > 1:
            raise ValueError("opponent evidence cannot pool unrelated identities")
        key = (h.get("hero_id"), str(identity))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def evidence_profile(hands: list[dict]) -> dict:
    """Keep raw descriptive counts, plus distinct-hand evidence for serving."""
    from bot.opponent import profile
    hands = unique_hands(hands)
    result = profile(hands)
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    jams = {}
    for hand in hands:
        pre, opp = hand["history"][0], hand["opp_pos"]
        for i, (player, kind, to) in enumerate(pre):
            if player != opp:
                continue
            node = None
            if i == 0 and opp == "btn":
                node = "btn_first_in"
            elif i == 1 and opp == "bb" and pre[0][0] == "btn":
                node = {"call": "bb_vs_limp", "raise": "bb_vs_open"}.get(pre[0][1])
            if node:
                depth = min(hand["stacks"].values())
                key = jam_key(node, depth, pre[:i])
                item = jams.setdefault(key, {"n": 0, "hits": 0, "index": len(jams) + 1})
                item["n"] += 1
                item["hits"] += int(kind == "raise" and to is not None and to >= depth - 1e-9)
        for name, obs in profile([hand])["nodes"].items():
            if name not in ACTIONS:
                continue
            counts[name] += 1
            norm = sum(obs.get(a, 0) for a in ACTIONS[name])
            for action in ACTIONS[name]:
                sums[name][action] += obs.get(action, 0) / max(norm, 1e-12)
    result["read_evidence"] = {
        "version": VERSION, "method": "hand-clustered-anytime-hoeffding",
        "family_alpha": ALPHA, "channel_budget": CHANNELS,
        "min_hands": MIN_HANDS, "max_width": MAX_WIDTH, "practical_margin": MARGIN,
        "assumption": "stable conditional action frequencies across hands",
        "nodes": {name: {"n": n, "totals": dict(sums[name])} for name, n in counts.items()},
        "jams": jams, "decisions": {},
        "showdown": {"applied": False, "reason": "selected_reveals_not_range_calibrated"},
    }
    return result


def gate(prof: dict, name: str, reference: dict) -> tuple[dict, dict]:
    """Project the fixed reference to the nearest confidence-compatible simplex.

    Abstain until an interval excludes a practically equivalent reference. This
    uses the closest supported frequencies, rather than the noisy point estimate.
    """
    report = {"applied": False, "reason": "missing_independent_evidence", "n_hands": 0,
              "reference": reference}
    evidence = prof.get("read_evidence", {})
    if evidence.get("version") != VERSION or name not in ACTIONS:
        return dict(reference), report
    obs = evidence.get("nodes", {}).get(name, {})
    n = obs.get("n", 0)
    report["n_hands"] = n
    actions = ACTIONS[name]
    base = {a: reference.get(a, 0.0) for a in actions}
    norm = sum(base.values())
    if norm <= 0:
        report["reason"] = "missing_reference"
        return dict(reference), report
    base = {a: v / norm for a, v in base.items()}
    bounds = {a: interval(obs.get("totals", {}).get(a, 0), n) for a in actions}
    report.update(intervals=bounds, estimates={a: obs.get("totals", {}).get(a, 0) / max(n, 1) for a in actions})
    if n < MIN_HANDS:
        report["reason"] = "too_few_distinct_hands"
    elif max(hi - lo for lo, hi in bounds.values()) > MAX_WIDTH:
        report["reason"] = "uncertainty_too_wide"
    elif not any(lo > base[a] + MARGIN or hi < base[a] - MARGIN for a, (lo, hi) in bounds.items()):
        report["reason"] = "no_supported_practical_difference"
    else:
        # Euclidean projection onto box-constrained probability simplex.
        left, right = -1.0, 1.0
        for _ in range(70):
            shift = (left + right) / 2
            total = sum(min(hi, max(lo, base[a] + shift)) for a, (lo, hi) in bounds.items())
            if total > 1:
                right = shift
            else:
                left = shift
        rates = {a: min(hi, max(lo, base[a] + (left + right) / 2)) for a, (lo, hi) in bounds.items()}
        report.update(applied=True, reason="supported_practical_difference", rates=rates)
        evidence["decisions"][name] = report
        return rates, report
    evidence["decisions"][name] = report
    return dict(reference), report


def serving_profile(prof: dict, baseline: dict | None = None) -> dict:
    """Only approved action-rate nodes reach legacy range/slowplay consumers."""
    defaults = {}
    for street, bet, raise_ in (("flop", .40, .12), ("turn", .33, .09), ("river", .33, .07)):
        defaults[f"{street}_vs_bet"] = {"fold": (1 - raise_) / 2, "call": (1 - raise_) / 2, "raise": raise_}
        for suffix in ("lead", "bet_when_checked_to", "cbet"):
            defaults[f"{street}_{suffix}"] = {"bet": bet, "check": 1 - bet}
    nodes = {}
    for name, ref in defaults.items():
        raw = (baseline or {}).get("nodes", {}).get(name, {})
        reference = {a: raw.get(a, 0) for a in ACTIONS.get(name, ())} if raw.get("n") else ref
        rates, report = gate(prof, name, reference)
        if report["applied"]:
            nodes[name] = dict(rates, n=report["n_hands"])
    return {"hands": prof.get("hands", 0), "nodes": nodes, "read_evidence": prof.get("read_evidence", {})}


def approved_nodes(prof: dict | None) -> dict:
    """Serving consumers must not fall back to legacy unqualified raw counts."""
    prof = prof or {}
    evidence = prof.get("read_evidence", {})
    if evidence.get("version") != VERSION:
        return {}
    decisions = evidence.get("decisions", {})
    return {name: obs for name, obs in prof.get("nodes", {}).items()
            if decisions.get(name, {}).get("applied") and name in ACTIONS}


def main() -> None:
    """Read-only local report; never trains or promotes an exact-card policy."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from bot.adapt import BASELINE, Preflop, postflop_exploit
    from bot.bluffmodel import calibration_report
    from bot.opponent import canonical, load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opponent", nargs="?", help="omit to report each identity separately")
    args = parser.parse_args()
    hands = load(args.opponent)
    grouped = defaultdict(list)
    for hand in hands:
        grouped[canonical(hand["opp_id"])].append(hand)
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    pre = Preflop()
    reports = {}
    for identity, records in sorted(grouped.items()):
        prof = evidence_profile(records)
        pre.adapt(prof)
        served = serving_profile(prof, baseline)
        reports[identity] = {
            "hands": prof["hands"], "data_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "postflop": postflop_exploit(served, baseline) if baseline else None,
            "evidence": prof["read_evidence"], "calibration": calibration_report(records)}
    root = Path(__file__).resolve().parents[1]
    sources = ("bot/reads.py", "bot/adapt.py", "bot/bluffmodel.py", "bot/engine.py", "bot/potodds.py")
    source_hash = hashlib.sha256(b"".join((root / p).read_bytes() for p in sources)).hexdigest()
    print(json.dumps({"schema_version": VERSION, "source_sha256": source_hash,
                      "opponents": reports}, indent=2))


if __name__ == "__main__":
    main()
