"""Check the equity engine against well-known reference numbers."""
from preflop.cards import HANDS, combos, n_combos
from preflop.equity import class_vs_class, vs_random


def test_169_classes_1326_combos():
    assert len(HANDS) == 169
    assert sum(n_combos(h) for h in HANDS) == 1326
    assert all(len(combos(h)) == n_combos(h) for h in HANDS)


def test_aa_vs_random_about_85():
    assert abs(vs_random("AA") - 0.852) < 0.005


def test_aa_vs_kk_about_82():
    eq, _ = class_vs_class("AA", "KK")
    assert abs(eq - 0.82) < 0.01


def test_small_pair_vs_overcards_coin_flip():
    eq, _ = class_vs_class("22", "AKo")
    assert 0.48 < eq < 0.55


def test_card_removal_weights():
    # AA vs AKs: AsAh blocks 2 of the 4 AKs combos -> 6 * 2 = 12 combo pairs.
    _, w = class_vs_class("AA", "AKs", iters=1)
    assert w == 12
