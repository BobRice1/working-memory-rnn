# Model Run Archive

On 2026-07-13, the outputs were organized around three dissertation
progression stages:

1. `outputs/categorical_working_memory/`: categorical `tanh` delayed-response baseline.
2. `outputs/circular_working_memory/`: continuous circular population-code model.
3. `outputs/fixation_circular_working_memory/`: canonical fixation-gated
   circular model.

The following runs were preserved, not deleted, under `outputs/archive/`:

| Archived run | Status | Configuration provenance |
|---|---|---|
| `baseline_delay_relu` | Superseded ReLU categorical baseline | Removed from the active set in commit `408e824`; recoverable from Git history |
| `baseline_delay_stable` | Superseded randomized-delay ReLU variant | Removed from the active set in commit `408e824`; recoverable from Git history |
| `tuned_delay_stable` | Intermediate circular attractor-oriented model | `configs/archive/tuned_delay_stable.yaml` |
| `tuned_delay_response_gate` | Superseded hold-then-report prototype | Config is embedded in its checkpoint; its uncommitted YAML and old code are not in current Git history |
| `tuned_delay_fixation_gate` | Fixed-delay Yang-style precursor | `configs/archive/tuned_delay_fixation_gate.yaml` |

Archiving changes organizational status only. Checkpoints, metrics, arrays,
figures, and other artifacts remain in their run directories beneath
`outputs/archive/`. Historical changelog entries retain the paths used when the
runs were originally produced.

The categorical and early circular stages are now historical. The
fixation-gated circular family remains current, alongside the separately
trained N-back family. Archived runs may be cited as development history but
are not current variants.

On 2026-07-14, the initial dense structured-noise figure suite and long report
were archived under `outputs/archive/noise_structure_initial_figure_suite_2026-07-14/`
without deleting the underlying five-seed experiment metrics.

Historical changelog entries and ignored output directories may still contain
the earlier `yang_fixation_circular_working_memory` name. Those paths identify
the artifacts as originally generated and are not current configuration names.
See `docs/repository-map.md` for the current executable path.
