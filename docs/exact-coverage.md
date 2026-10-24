# Exact-only serving and coverage training

Default `Engine` and the PokerNow server require a validated exact policy. They do not use preflop hand-class averages, representative-flop group averages, turn/river texture or 12-bucket tables, nearest-stack policies, deep-stack cells, or raise-sequence rules. `exact_required=False` explicitly enables the historical research benchmark; it is not the live default. Deep-cell substitution is removed even in that mode.

## What is and is not complete

Exact representation covers literal private cards/suits, each visible board card, kicker/blocker differences, position, effective depth, integer chip contributions, minimum raises and the full legal raise history on every street. Only private-card order and simultaneous flop order are normalized. No future board or opponent private cards enter an information key. Covered turn/river/re-raise decisions bypass all legacy overrides.

**Training coverage is not complete.** Exact keys do not make an unvisited state trained. For one fixed hero combo alone there are 19,600 flops, 921,200 ordered turn boards and 42,375,200 ordered river boards before positions, stack depths, bet amounts and histories. A finite observed-state manifest is not the full game. All reports explicitly retain `full_game_complete:false`.

A missing, unpromoted, undertrained, invalid, unsupported or unreadable policy withholds the action. The server returns `decision:null` with an explanation, so the client cannot execute a bucket-based substitute. The exact request is queued; cached polls neither resample nor inflate counts. Restart an already running server to load the change. The default strict library engine also has a persistent queue; explicit `coverage_path=None` disables its persistence for controlled experiments.

No exact artifact is currently promoted. An overnight run may fill only part of its requested frontier, and visit counts do not establish convergence or playing strength.

## Durable observations and fallback frequencies

`output/pokernow/coverage.sqlite` stores exact request identities, context, occurrence count and reason, plus serving attempts and source counts. Known cards are validated; shown opponent cards and future board cards are omitted. Archive import is idempotent and does not count as a new live attempt. Unsupported but reconstructable archived requests remain queued and are reported as excluded training gaps.

```sh
.venv/bin/python -m bot.coverage
.venv/bin/python -m bot.coverage --export output/exact/pending.json
```

The report gives absolute counts and `fraction_of_attempts` for each reason, with street, exact depth and prior bet/raise-count slices. Reasons are nonexclusive: fractions can sum above 100%. Source counts distinguish exact, withheld and explicitly permitted legacy play. Decision audit diagnostics also contain the actual fallback labels, strategy mode and coverage snapshot. Withheld audit events retain version/run/hand links.

Exact failures are labeled `exact:<status>`. Explicit legacy benchmarks separately count hand-group/bucket use, representative flops, successful flop group averaging, solver lookup attempts that required a fallback, nearest preflop sizes, postflop size translation, nearest stacks, borrowed depth nodes, missing-node aliases, near-jams interpreted as all-ins, deep preflop raise aliases and rule-based postflop raise sequences. Uniform legacy cells are reported heuristically; the old arrays lack exact training-visit metadata. Their deep-stack substitute is disabled.

## Queue preparation

```sh
.venv/bin/python scripts/prepare_coverage.py \
  --output output/exact/coverage-2026-10-24/targets.json
```

Preparation uses the persistent queue, existing decision logs and raw hand archives. A hand contributes a request immediately before each recorded hero action, using only that street's visible board and earlier actions. Raw archives are never rewritten. Missing/unsupported contexts are counted; they cannot be called covered.

The manifest includes input hashes, a config/target hash, exact target counts and rejected-record reasons. Default precision is 0.001bb. Its raise menu includes minimum legal raises, all-ins, 14 pot increments (10/20/25/33⅓/40/50/60/75/100/125/150/200/300/400%) and every observed legal raise total. This preserves observed off-menu sizes and repeated raises without snapping or a raise-depth cutoff. The fraction menu remains an action abstraction; `train --exhaustive` enumerates every legal chip total at configured precision with much greater cost.

`prepare --expand-private` explicitly enumerates every board-compatible literal private combo at requested public roots. It expands that finite frontier, not all boards/histories, and can make the manifest and training much larger. Without the flag, requested hero combos are the coverage denominator; other exact combos can still be trained during sampled trajectories.

The 2026-10-24 archive plan initially contains 761 supported exact targets: 357 preflop, 185 flop, 131 turn, 88 river; 247 depths; 68 extra observed sizing totals; histories through six prior bets/raises. Sixteen histories fail current legality reconstruction and remain unresolved. 935 old decision rows lack action-time contexts; raw hand archives recover the supported targets above. These are dataset measurements, not coverage percentages for the full game.

## Guided full-game training

```sh
.venv/bin/python scripts/train_exact.py train \
  --path output/exact/coverage-2026-10-24/overnight/candidate.sqlite \
  --coverage-targets output/exact/coverage-2026-10-24/targets.json \
  --hours 8 --batch 50 --max-gb 16 --target-min-updates 1000
# Resume after the previous writer stops, reusing RNG/config/target manifest:
.venv/bin/python scripts/train_exact.py train \
  --path output/exact/coverage-2026-10-24/overnight/candidate.sqlite --hours 8 --max-gb 16
```

`CoverageTrainer` starts every trajectory at the blinds in the `whole_game` namespace. Targets choose the exact depth. Round-robin target scheduling avoids relying on random deals to revisit a specific cell. Both players receive updates; no isolated subgame with assumed uniform posterior ranges is installed.

Chance proposal: 80% condition on the requested literal hero combo and visible board; 20% sample the uniform full deal at that depth. If the uniform event probability is `p`, the proposal is `q=.8+.2p` on the event and `.2p` otherwise. Initial regret and average contributions receive `p/q`. Marginal event probabilities use `1/C(52,2)` for the hero combo and, for observed boards, factors `1/C(50,3)`, then `1/47` and `1/46`. Opponent cards and remaining runouts are sampled compatibly. This proposal may know a training runout, but the policy key and policy inputs cannot.

History proposal: while the sampled history matches the requested prefix, use 90% proposal probability on its next action plus 10% of the normal behavior distribution. The existing outcome-sampling prefix/suffix corrections account for that behavior probability. After divergence or the target prefix, use normal exploration. A guided amount missing from the configured legal menu is an error. See the underlying sampling structure in [OpenSpiel's outcome-sampling implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/outcome_sampling_mccfr.py).

The sampler has full support, but importance variance can be high. This is a targeted estimator, not proof of equilibrium. Training source/config/runtime and exact target-table fingerprints protect resume; a supplied plan cannot silently change precision or sizing menus. Existing checkpoints with empty extra-size menus retain their read-only config fingerprints; old training code cannot be silently resumed after algorithm changes.

## Progress and completion

Status includes aggregate exact infosets/updates and queued-cell readiness by street, depth and prior raise count. A target is counted ready only if it has the update floor, matching legal actions and finite, nonnegative, positive average-strategy mass, matching candidate policy-read requirements. `update_floor_targets` is separate. A zero-own-reach historical line can have many updates but no average mass; it remains uncovered.

Jobs stop at the requested frontier's readiness floor, time/iteration budget, disk/free-space guard, signal or failure. The status says why. The same atomic checkpoints, single-writer leases, saved RNG and frozen snapshots as standard exact training apply. Final frozen snapshots are saved beside the candidate as `latest.sqlite`.

Candidate readiness does not mean live eligibility. A frozen candidate still needs independent held-out evaluation and the existing promotion gates before the strict live engine can use it. No automatic promotion or live source change occurs. New queue entries after a manifest is frozen await a new manifest/job rather than silently changing a running training distribution.
