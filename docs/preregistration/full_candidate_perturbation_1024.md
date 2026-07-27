# Full candidate-only perturbation evaluation

Frozen before outcome evaluation: 27 July 2026.

This descriptive follow-up expands the completed three-seed pilot to all five
original circular-task checkpoints and all ten competent N-back checkpoints.
Each condition uses 1,024 trials or sequences. Existing checkpoints are reused;
there is no retraining.

The fixed candidate grids are those in
`configs/full_candidate_perturbation_1024.yaml`: synaptic-drive gain,
heterogeneous drive gain, sensory input gain, circular distractor-window gain,
recurrent gain, state persistence, and conserved time constant. Gaussian noise,
cost matching, calibration, hybrids, and confirmatory hypothesis tests are
outside this run.

Every perturbed cell is compared with its checkpoint's native unperturbed
baseline on identical generated task samples. The outcomes remain settling
with relative preservation, long-minus-short delay selectivity,
distractor-minus-clean selectivity, and 2-back-minus-0-back discriminability
impairment.

Checkpoint is the replication unit. Results will report means, checkpoint
points, consistency counts, and dose trajectories without confirmatory
p-values. The circular distractor condition remains an out-of-distribution
probe and its native baseline was weak in the pilot; additional trials improve
precision but do not correct that validity limitation.
