# Persistence hidden-state analysis results

Descriptive Approach-B analysis of `state_persistence` on the variable-timing
circular family (10 seeds × 3 persistence values × 1,024 paired trials).
**Claim boundary:** mechanism description only; no matched-cost Gaussian comparator.

## Design

- Seeds: `[20260801, 20260802, 20260803, 20260804, 20260805, 20260806, 20260808, 20260809, 20260810, 20260811]`
- Persistence: `[1.0, 0.95, 0.9]`
- Base config SHA256: `C568B49FBF17504D6047454E150C00C54F3E8C9503CE9E4EDD50C2CDA5FA554D`
- `FINAL_SEED_BASE`: `202607300`
- Distractor metric: onset-aligned; recovery averaged over onset buckets with
  in-delay post window ≥ 1 step (late onsets dropped from recovery only).
- PCA: per-seed basis fit on persistence-1.00 clean delay tanh states (no arctanh);
  0.95/0.90 mean trajectories projected into that frozen basis.

## Pooled means (across 10 seeds)

| Persistence | Mean clean drift (°) | Mean clean decode err (°) | Mean step speed | Peak attraction | Recovery fraction |
|---|---:|---:|---:|---:|---:|
| 1.00 | 1.6339 | 0.9752 | 0.0855 | 0.0845 | 0.3074 |
| 0.95 | 1.7568 | 1.2503 | 0.0787 | 0.0944 | 0.2925 |
| 0.90 | 2.2215 | 1.9513 | 0.0752 | 0.1058 | 0.2741 |

## Artifacts

- Metrics CSV: `outputs/persistence_hidden_state_analysis/metrics/persistence_hidden_state_metrics.csv` (30 rows)
- Summary JSON: `outputs/persistence_hidden_state_analysis/metrics/persistence_hidden_state_summary.json`
- PCA figure: `outputs/persistence_hidden_state_analysis/figures/persistence_pca_delay_trajectories.png`
- Drift/recovery figure: `outputs/persistence_hidden_state_analysis/figures/persistence_drift_recovery_summary.png`

## Sanity checklist (executed in notebook)

- Pairing: identical angles / distractor angles / relative starts across gains.
- No-op: persistence 1.00 hidden states allclose to native `model(inputs)`.
- Shapes: hidden `[T, 1024, 64]`, weights `[64, 2]`, decoded `[T, 1024]`.
- Range: hidden within tanh support; onset bank covers 0..15 and sums to 1024.
- PCA: frozen per-seed basis; object id unchanged across gains.
- Row count: 30 (= 10 × 3).
