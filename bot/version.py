"""The bot's version: one per round of changes from a live game's feedback (v0 = the first version that played a
live game). The PokerNow nickname is NAME; bump VERSION and tag the commit (`git tag vN`) with every new round.

    v0  2ef14b3  first live version: 3x open, 5 s actions
    v1  10a5257  2.25x open, fewer limps, postflop retrained on the new ranges (exp 33-34)
    v2  efe9a0c  real 4-bets, stack-depth preflop, opponent tracking and adaptation (exp 35-37)
    v3  9e8f9e1  all-in call fix, pot odds on every call, separate player ids, 4.5 s actions (exp 38-39)
    v4  d9606f0  raise-size-aware river calls, river sizing, short-stack mode, learned bluff model (exp 40-41)
    v5  31471e1  strictness against big bets learned per opponent from showdowns (exp 42)
    v6  (tag)    never just call with the nuts (incl. the nut flush): raise or jam; multi-table; shove-node fixes (exp 43)
"""
VERSION = 6
NAME = f"HUBot v{VERSION}"
