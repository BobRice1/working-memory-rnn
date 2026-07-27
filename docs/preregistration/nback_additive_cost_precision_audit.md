# N-back additive-cost precision audit

## Audit status

Recorded on 2026-07-27 after the registered baseline-only precision run.

Frozen design commit: `3da286e`

Implementation commit: `0e5e813`
Device: NVIDIA GeForce RTX 3060 Laptop GPU

Perturbations, P5, strength grids, calibration, or 2-back outcomes run during
this phase: **none**.

## Registered data collection

All ten frozen checkpoints were evaluated on exactly 8,192 fresh,
unperturbed 0-back sequences:

```text
64 batches * 128 sequences = 8192 sequences per checkpoint
10 checkpoints * 8192 = 81920 total sequence observations
```

Every sequence log-loss value was finite and non-negative. The complete
`10 x 8192` matrix, `10 x 64` task-seed matrix, and 10,000 bootstrap maximum
SDs were persisted.

## Checkpoint descriptions

| Seed | Mean CE | Sample SD | Median | IQR | P95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260912 | 0.003899 | 0.001220 | 0.003715 | 0.001516 | 0.006177 | 0.011587 |
| 20260913 | 0.002018 | 0.000690 | 0.001886 | 0.000703 | 0.003189 | 0.009074 |
| 20260914 | 0.004121 | 0.001359 | 0.003892 | 0.001566 | 0.006395 | 0.020671 |
| 20260915 | 0.003430 | 0.001241 | 0.003266 | 0.001378 | 0.005328 | 0.022529 |
| 20260916 | 0.001421 | 0.000674 | 0.001271 | 0.000760 | 0.002622 | 0.009949 |
| 20260917 | 0.002262 | 0.001037 | 0.002052 | 0.000831 | 0.003926 | 0.042368 |
| 20260918 | 0.003029 | 0.001177 | 0.002857 | 0.001333 | 0.005100 | 0.016675 |
| 20260919 | 0.002424 | 0.001796 | 0.002122 | 0.001070 | 0.004333 | 0.086007 |
| 20260920 | 0.004721 | 0.001985 | 0.004396 | 0.001463 | 0.006861 | 0.060414 |
| 20260921 | 0.002937 | 0.001163 | 0.002715 | 0.001145 | 0.004848 | 0.019343 |

These results confirm that baseline cross-entropy is small and varies
materially across checkpoints, supporting the pre-registered decision to use
an additive rather than proportional cost.

## Family-wide precision result

The registered 10,000-draw bootstrap produced:

```text
sigma_upper = 0.0022746990417660466
```

Applying the frozen conservative formula:

```text
kappa = 2
z = 1.96
h = 0.005

n_required = 3.9754841629433284
```

After the registered minimum and batch-rounding rules:

```text
n_cost_check = 1024 sequences per checkpoint and cell
```

The minimum of 1,024, rather than the estimated requirement, determines the
future cost-check size.

## Acceptance decision

All 15 persisted validity checks passed:

- exact checkpoint order;
- exact sequence count for every checkpoint;
- finite, non-negative sequence units;
- finite `sigma_upper` and `n_required`;
- exact bootstrap draw count with finite values;
- cost-check size at or above the minimum, on a complete 128-sequence batch,
  and below the 8,192 maximum;
- complete, unique seed map;
- persisted arrays, descriptions, and seed map.

The baseline-only precision phase therefore **passes**.

## Frozen consequence

Future N-back calibration and held-out cost validation must use:

```text
additive target:        0.050 nats
acceptable band:       [0.040, 0.060] nats
maximum CI half-width:  0.005 nats
candidate-P5 gap:       0.005 nats
n_cost_check:           1024 sequences
```

Before any strength is evaluated, the exact additive calibration runner,
stochastic stream mapping, calibration-bank sequence count, held-out
bootstrap offsets, and confirmatory outcome procedure must be pre-registered
and committed.

## Authoritative artifacts

```text
outputs/nback_additive_cost_precision/metrics/
  nback_additive_cost_precision_summary.json
  nback_additive_cost_precision_checkpoint_descriptions.csv
  nback_additive_cost_precision_seed_map.csv

outputs/nback_additive_cost_precision/arrays/
  nback_additive_cost_precision_arrays.npz
```

Verified SHA-256 hashes:

```text
summary:
bec42d778327aa2e8f1aab99ab04133735660deac944c395b210566ea891b62a

checkpoint descriptions:
941b7c649e59a0d96d0d1928ecbdc10f04918b345dd860b50028666164b2b16e

seed map:
be8e7c48a8182e18513de10721749facebf9912c7c54f0c4df1dcd9236e56526

arrays:
b5198fd2164bb5726d3d3de72b15327d75475194ded873702d98194a9d738d1d
```

The manifest and all three artifact hashes stored inside the summary were
independently recomputed and matched.

## Claim boundary

This pass establishes attainable precision for small matched 0-back
confidence costs. It is not evidence that any perturbation reproduces a
human psilocybin signature and does not support a biological mechanism claim.
