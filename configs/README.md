# Configuration Index

Configurations remain at their recorded paths because tests, preregistrations,
run manifests, and historical reproduction commands refer to them directly.
Placement in the root of this directory does not imply that a configuration is
currently recommended.

## Current Task and Evaluation Configurations

| Configuration | Status | Purpose |
|---|---|---|
| `fixation_circular_distractor_working_memory.yaml` | Current distractor baseline | Five retained one-item circular checkpoints trained on balanced clean/distractor batches. |
| `fixation_circular_working_memory.yaml` | Prior clean baseline | Five-seed circular family retained for the completed candidate screen and earlier results. |
| `nback_working_memory_screened_final.yaml` | Current baseline pool | Competence-screened N-back checkpoint family used by the completed candidate screen. |
| `full_candidate_perturbation_trained_distractor_1024.yaml` | Current completed evaluation | Candidate-only 1,024-trial circular rerun on the trained-distractor checkpoints, paired with the unchanged completed N-back results. |
| `full_candidate_perturbation_1024.yaml` | Prior completed evaluation | Original candidate-only screen using the clean-trained circular family and evaluation-only distractors. |

## Supporting Completed Pipelines

| Configuration | Status | Purpose |
|---|---|---|
| `nback_additive_perturbation.yaml` | Historical completed pipeline | Frozen additive N-back perturbation record. Its one-off runner was retired from the active package and remains available in Git commit `d64b2cf`. |
| `exploratory_psilocybin_signature_pilot.yaml` | Superseded pilot | Three-seed, 256-trial exploratory screen superseded by the 1,024-trial evaluation. |
| `state_persistence_080_distractor_selectivity.yaml` | Post-hoc completed extension | Separate 1,024-trial evaluation of state persistence `0.80` on the five retained distractor-trained circular checkpoints. |

## Historical Baseline Development

| Configuration | Status | Purpose |
|---|---|---|
| `categorical_working_memory.yaml` | Historical | Initial four-class delayed-response baseline. |
| `circular_working_memory.yaml` | Historical | Early fixed-delay circular working-memory baseline. |

Additional earlier circular configurations are physically grouped in
`configs/archive/`.

## Failed Multicondition Family

These configurations belong to the failed two-slot load/distractor task. They
are retained because the failure and rescue ladder are part of the
pre-registered scientific record. They must not be presented as successful
perturbation baselines.

- `multicondition_working_memory.yaml`
- `multicondition_working_memory_distribution_loss.yaml`
- `multicondition_working_memory_distribution_role.yaml`
- `multicondition_working_memory_distribution_role_h128.yaml`

## N-Back Development and Rescue History

The following configurations record the progression that produced the
competence-screened checkpoint pool:

- `nback_working_memory.yaml`
- `nback_working_memory_balance_rescue.yaml`
- `nback_working_memory_final.yaml`
- `nback_working_memory_budget_rescue.yaml`
- `nback_working_memory_budget_final.yaml`

Use the screened-final configuration for current evaluation unless a historical
run is being reproduced deliberately.

## Change Policy

Do not silently modify a configuration tied to a preregistration, completed
run, or recorded hash. Create a new configuration for a new experiment and
record its relationship to the earlier specification in `docs/changelog.md`.
