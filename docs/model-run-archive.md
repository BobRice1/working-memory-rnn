# Model Run Archive

On 2026-07-13, the outputs were organized around three active dissertation
progression stages:

1. `outputs/baseline_delay/`: categorical `tanh` delayed-response baseline.
2. `outputs/tuned_delay/`: continuous circular population-code model.
3. `outputs/tuned_delay_fixation_gate_stable/`: stabilized Yang-style
   fixation-gated circular model.

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

The active progression supports the dissertation narrative from categorical
memory, through continuous circular coding, to the selected Yang-style task.
Archived runs may be cited as development history but are not current variants.
