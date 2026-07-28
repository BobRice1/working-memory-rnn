# Post-hoc state-persistence 0.80 distractor-selectivity result

Date: 2026-07-28

## Scope

This is a post-hoc exploratory extension of the completed candidate screen. It
evaluates state persistence `p = 0.80`, which was not in the original fixed
grid, on the same five retained distractor-trained circular checkpoints:

- `20260731`
- `20260732`
- `20260733`
- `20260735`
- `20260736`

Each checkpoint and condition used 1,024 trials. The original candidate grid
and outputs were not modified.

## Scoring

For each checkpoint, clean and distractor impairment were calculated relative
to their own native delay-20 baselines:

```text
condition impairment =
    (perturbed angular error - native angular error) / native angular error

distractor selectivity =
    distractor impairment - clean impairment
```

A positive selectivity value means the perturbation caused proportionally more
additional damage on distractor trials. A negative value means clean trials
were proportionally more affected.

## Result

| Checkpoint | Clean impairment | Distractor impairment | Distractor selectivity |
|---:|---:|---:|---:|
| 20260731 | +0.695 | +0.758 | +0.063 |
| 20260732 | +2.698 | +1.687 | -1.011 |
| 20260733 | +2.331 | +2.002 | -0.329 |
| 20260735 | +1.801 | +1.256 | -0.546 |
| 20260736 | +0.620 | +0.557 | -0.063 |
| **Mean** | **+1.629** | **+1.252** | **-0.377** |

Across checkpoints, distractor selectivity was:

- mean `-0.377`;
- SD `0.426`;
- student-t 95% interval `[-0.906, 0.151]`;
- positive in `1/5` checkpoints.

Mean raw delay-20 angular error changed as follows:

| Condition | Native error | Error at `p = 0.80` |
|---|---:|---:|
| Clean | 3.379 degrees | 8.490 degrees |
| Distractor | 4.147 degrees | 9.196 degrees |

Fixation accuracy remained at least `0.978`, so the error increase was not
explained by fixation-gate failure.

## Interpretation

State persistence `0.80` produced broad circular-memory degradation rather
than selective distractor vulnerability. Clean error increased by about 163%
relative to its native baseline, exceeding the 20% preservation ceiling in all
five checkpoints. Distractor error remained higher than clean error in raw
degrees on average, but its proportional increase from its already higher
native baseline was smaller.

This result does not strengthen the distractor-selectivity case for reduced
persistence. It instead suggests that moving from `p = 0.95` to `p = 0.80`
leaves the selective, accuracy-preserving regime and enters a general-damage
regime. Because `p = 0.80` was chosen after inspecting the original screen,
this extension is descriptive and post hoc.

## Reproduction record

- Configuration:
  `configs/state_persistence_080_distractor_selectivity.yaml`
- Full grid:
  `outputs/state_persistence_080_distractor_selectivity/metrics/state_persistence_080_grid.csv`
- Run metadata:
  `outputs/state_persistence_080_distractor_selectivity/metrics/state_persistence_080_metadata.json`
