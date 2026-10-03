"""The 169 starting-hand classes and the 1,326 combos behind them."""
import itertools

RANKS = "AKQJT98765432"   # high to low, also the order of the 13x13 grid
SUITS = "shdc"


def hand_name(r, c):
    """Name of the cell at row r, column c of the 13x13 grid.

    Diagonal = pairs, above the diagonal = suited, below = offsuit.
    """
    if r == c:
        return RANKS[r] * 2
    hi, lo = min(r, c), max(r, c)
    return RANKS[hi] + RANKS[lo] + ("s" if r < c else "o")


# All 169 classes in grid order (row by row). Index i <-> grid cell (i // 13, i % 13).
HANDS = [hand_name(r, c) for r in range(13) for c in range(13)]
INDEX = {h: i for i, h in enumerate(HANDS)}


def combos(name):
    """Every 2-card combo of a class, e.g. 'AKs' -> [('As','Ks'), ('Ah','Kh'), ...]."""
    a, b = name[0], name[1]
    if a == b:
        return [(a + s1, b + s2) for s1, s2 in itertools.combinations(SUITS, 2)]
    if name[2] == "s":
        return [(a + s, b + s) for s in SUITS]
    return [(a + s1, b + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]


def n_combos(name):
    return 6 if name[0] == name[1] else 4 if name[2] == "s" else 12


def overlaps(c1, c2):
    return bool(set(c1) & set(c2))
