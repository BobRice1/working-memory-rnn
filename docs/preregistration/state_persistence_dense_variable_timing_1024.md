# Dense state-persistence neighbourhood evaluation

Frozen before outcome inspection: 29 July 2026.

## Purpose

Map the four descriptive signatures across a denser carried-state persistence
grid on the current balanced 10-circular / 10-N-back checkpoint pools. This is
the supervisor-requested neighbourhood analysis around `0.95`, not a new blind
candidate screen and not a matched-cost Gaussian comparison.

## Frozen grid

```text
0.80, 0.85, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94,
0.95, 0.96, 0.97, 0.98, 0.99, 1.00
```

`0.80` and `0.85` are coarser low-end anchors. Values from `0.88` to `1.00`
step by `0.01`. Values above `1.00` are excluded from this run.

## Scope

- Circular family: retained variable-timing seeds `20260801`--`20260806` and
  `20260808`--`20260811`, with randomized distractor timing and 1,024 trials
  per cell.
- N-back family: screened seeds `20260912`--`20260921`, with 1,024 sequences
  per cell.
- Operator: `state_persistence` only.

## Outcomes

Report seed-level trajectories and mean ± SD for:

1. clean delay-20 settling change
2. long-minus-short delay selectivity
3. distractor-minus-clean selectivity
4. 2-back-minus-0-back load selectivity

Also report the slowing-with-preservation pass count at each strength using the
existing rule (latency-valid, settling delta `> 0`, clean proportional cost
`<= 0.20`).

## Claim boundary

Descriptive neighbourhood mapping only. It does not establish biological
mechanism or specificity versus matched Gaussian disruption.
