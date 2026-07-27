# Configuration Index

Configurations remain at their recorded paths because tests, preregistrations,
run manifests, and historical reproduction commands refer to them directly.
Placement in the root of this directory does not imply that a configuration is
currently recommended.

## Current Task and Evaluation Configurations

| Configuration | Status | Purpose |
|---|---|---|
| `fixation_circular_working_memory.yaml` | Current baseline | Five-seed fixation-gated circular working-memory family. |
| `nback_working_memory_screened_final.yaml` | Current baseline pool | Competence-screened N-back checkpoint family used by the completed candidate screen. |
| `full_candidate_perturbation_1024.yaml` | Current completed evaluation | Candidate-only 1,024-trial screen across the circular and N-back families. |

## Supporting Completed Pipelines

| Configuration | Status | Purpose |
|---|---|---|
| `nback_additive_perturbation.yaml` | Historical completed pipeline | Frozen additive N-back perturbation record. Its one-off runner was retired from the active package and remains available in Git commit `d64b2cf`. |
| `exploratory_psilocybin_signature_pilot.yaml` | Superseded pilot | Three-seed, 256-trial exploratory screen superseded by the 1,024-trial evaluation. |

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
