# Project Changelog

This changelog tracks two related histories:

<details>
<summary>2026-07-28 - Rerun candidate grid with randomized distractor timing</summary>

Action:

- Evaluated the seven candidate perturbation families on all 10 retained
  variable-timing circular checkpoints with 1,024 trials per cell.
- Assigned the five-step distractor per trial using one frozen, randomized,
  exactly balanced bank over relative starts `0`--`15` in the 20-step delay.
  Each start occurred 64 times.
- Paired target angles, distractor angles, and distractor starts across
  baseline and perturbation settings.
- Reused the unchanged completed 10-seed N-back signature table.
- Left common-window distractor drift/recovery fields undefined for the
  mixed-onset cell instead of applying an incorrect midpoint window.

Result:

- State persistence `0.95` remained the only profile meeting the complete
  descriptive majority pattern.
- Across the 10 circular checkpoints, mean clean delay-20 proportional error
  change was `-0.003 +/- 0.114`, restricted settling change was
  `+0.032 +/- 0.062` steps, long-minus-short delay selectivity was
  `+0.137 +/- 0.143`, and randomized-timing distractor selectivity was
  `+0.035 +/- 0.063`.
- Slowing with the clean-preservation rule held in `6/10`, delay selectivity
  was positive in `9/10`, and distractor selectivity was positive in `8/10`.
  The unchanged N-back load contrast was positive in `10/10`.
- Student-t 95% intervals excluded zero only for circular delay selectivity
  (`[0.035, 0.240]`); settling (`[-0.013, 0.076]`) and distractor selectivity
  (`[-0.010, 0.080]`) remained uncertain.
- Native randomized-timing distractor error averaged `4.382` degrees and all
  baseline distractor cells passed fixation and settling validity.
- Fifteen of 1,890 circular rows were latency-invalid, all in
  heterogeneous-drive settings. Every persistence `0.95` clean delay-20 cell
  was valid.
- Circular runtime was 1,129 seconds (18 minutes 49 seconds) on CUDA.

Interpretation:

- The persistence `0.95` candidate pattern survives a larger independent
  circular family and a distractor outcome distributed across the delay.
- This is stronger descriptive replication than the historical five-seed
  midpoint result, but the settling and distractor intervals still include
  zero and the N-back component is reused rather than newly blinded.
- Because both the checkpoint family and distractor outcome changed, numerical
  differences from the historical rerun cannot be assigned to either change
  alone.
- This remains a computational sufficiency candidate, not a unique or
  biological psilocybin mechanism. Matched-cost Gaussian specificity remains
  outstanding.

Recorded outputs:

- `outputs/full_candidate_perturbation_variable_timing_1024/circular_variable_timing/metrics/circular_variable_timing_grid.csv`
- `outputs/full_candidate_perturbation_variable_timing_1024/circular_variable_timing/metrics/circular_variable_timing_metadata.json`
- `outputs/full_candidate_perturbation_variable_timing_1024/summary/cross_task_signature_summary.csv`
- `docs/reports/full_candidate_perturbation_variable_timing_1024_results.md`

</details>

<details>
<summary>2026-07-28 - Freeze 10-seed circular perturbation rerun</summary>

Action:

- Added a separate perturbation configuration for the 10 retained
  variable-timing circular checkpoints.
- Kept the previous seven operator grids, 1,024 trials per cell, clean delay
  contrasts, held-out evaluation seeds, and completed 10-seed N-back signature
  table unchanged.
- Added a dedicated runner and explicit checkpoint-manifest loading.
- Replaced the historical midpoint-only distractor outcome with a per-trial,
  stratified-uniform random assignment across all 16 valid onset positions for
  a 20-step delay and five-step distractor.
- Frozen onset assignments are paired across baseline and every perturbation;
  each onset occurs exactly 64 times in the 1,024-trial bank.

Interpretation:

- This is a descriptive family rerun, not a new blind candidate nomination or
  a matched-cost Gaussian comparison.
- The circular family and distractor outcome both differ from the historical
  five-seed midpoint-only rerun, so between-run changes cannot be attributed to
  the checkpoint family alone.

Planned outputs:

- `outputs/full_candidate_perturbation_variable_timing_1024/`

</details>

<details>
<summary>2026-07-28 - Train 10-seed variable-timing circular family</summary>

Action:

- Executed the frozen variable-timing circular protocol on CUDA.
- Evaluated candidates on paired 1,024-trial banks for clean trials and
  distractor onset fractions `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`.
- Retained the first 10 candidates passing every frozen competence gate.
- The initial execution completed seeds `20260801`--`20260804` and was stopped
  during seed `20260805` before that seed wrote a checkpoint. The resumed run
  reused and re-evaluated the four complete checkpoints and restarted seed
  `20260805`; no partial checkpoint was retained.

Result:

- Eleven candidates were evaluated. Ten passed:
  `20260801`--`20260806`, `20260808`--`20260811`.
- Seed `20260807` failed clean and distractor competence with approximately 91
  degrees mean angular error and was excluded under the frozen rule.
- Across the retained checkpoints, clean error was `2.972 +/- 0.527` degrees
  and the mean across distractor timings was `3.813 +/- 0.641` degrees
  (mean +/- seed SD).
- The worst timing-specific error per checkpoint averaged
  `3.899 +/- 0.638` degrees and ranged from `3.086` to `5.294`, comfortably
  below the 15-degree per-timing gate.
- The within-checkpoint range across the five timings averaged
  `0.190 +/- 0.107` degrees (range `0.083`--`0.417`). For comparison, the
  historical midpoint-trained family had a descriptive mean timing range of
  `0.693` degrees under its separate timing analysis.
- Minimum fixation accuracy across conditions averaged `0.9766`, above the
  0.90 gate.

Interpretation:

- The new family is competent across the full trained timing distribution and
  is suitable as the 10-checkpoint circular pool for the planned balanced
  comparison.
- The smaller timing range is descriptive evidence that variable-timing
  training reduced the pronounced timing dependence of the old family. It is
  not a randomized causal comparison because the families use independent
  seeds and separate held-out banks.
- The failed seed is a training non-convergence outcome, not evidence about
  distractor timing.

Recorded outputs:

- `outputs/fixation_circular_variable_distractor_working_memory/metrics/fixation_circular_variable_distractor_working_memory_pool.csv`
- `outputs/fixation_circular_variable_distractor_working_memory/metrics/fixation_circular_variable_distractor_working_memory_pool_summary.json`
- `docs/reports/variable_distractor_timing_training_result.md`

</details>

<details>
<summary>2026-07-28 - Freeze variable-timing circular training protocol</summary>

Action:

- Added a new, independent circular checkpoint-family configuration with
  delay-relative distractor starts at `0.00`, `0.25`, `0.50`, `0.75`, and
  `1.00`.
- Kept clean and distractor batches balanced 1:1. Within distractor batches,
  each timing is presented once per shuffled five-batch block.
- Froze an ordered 15-seed candidate schedule (`20260801`--`20260815`) and a
  first-10-passing retention rule. The final family therefore contains 10
  independently initialized competent checkpoints; later candidates are used
  only if an earlier candidate fails.
- Added paired held-out evaluation at every trained timing using 1,024 trials
  per condition and a separate frozen evaluation seed bank.
- Froze competence gates of clean mean angular error no greater than 10
  degrees, mean angular error no greater than 15 degrees at each individual
  distractor timing, and fixation accuracy of at least 0.90 in every
  condition.
- Added timing-specific training-history fields, pool metrics, and focused
  tests.

Interpretation:

- The per-timing thresholds are preregistered model-validity and competence
  thresholds informed by task structure; they are not literature-derived
  behavioural effect sizes.
- Timing range is reported descriptively and is not used to select
  checkpoints.
- This family is separate from the five midpoint-trained checkpoints and does
  not retroactively alter their timing-generalisation result.

Planned output:

- `outputs/fixation_circular_variable_distractor_working_memory/`

</details>

<details>
<summary>2026-07-28 - Evaluate circular distractor-timing generalisation</summary>

Action:

- Added `configs/circular_distractor_timing_generalisation.yaml`,
  `src/wm_rnn/circular_distractor_timing.py`, and focused tests.
- Evaluated the five retained distractor-trained circular checkpoints with
  frozen weights at delay-relative distractor starts `0`, `4`, `8`, `11`, and
  `15`, plus a clean condition.
- Used 1,024 paired trials per checkpoint and condition. Target banks were
  exactly equal across all conditions and distractor-angle banks were exactly
  equal across all five timing conditions.
- Added an explicit end-of-delay boundary rule: peak drift and end attraction
  remain defined, but recovery is `NaN` when no post-distractor delay window
  exists.
- Recorded the analysis in
  `docs/reports/circular_distractor_timing_generalisation.md`.

Result:

- The trained midpoint had the lowest distractor cost in every checkpoint.
- Mean midpoint cost was `0.768` degrees. Timing-minus-midpoint costs were
  `+0.206` degrees at the start, `+0.043` at quarter, `+0.042` at
  three-quarter, and `+0.693` at the end.
- Each displacement was positive in `5/5` checkpoints. Student-t 95%
  intervals excluded zero for start, three-quarter, and end; the quarter
  interval was `[-0.00007, 0.086]`.
- All 30 cells passed fixation and latency validity, and every cell had a
  settled fraction of `1.00`.
- Archived midpoint response errors reproduced within `4.91e-7` degrees.

Interpretation:

- The checkpoints show timing-specific filtering with a pronounced
  late-delay asymmetry, not fully timing-invariant filtering.
- The result cannot distinguish an absolute timing strategy from
  state-dependent susceptibility or reduced recovery time before response.
- This is a post-result descriptive robustness analysis, not confirmatory
  evidence or a biological psilocybin mechanism.

Recorded outputs:

- `outputs/circular_distractor_timing_generalisation/metrics/timing_metrics.csv`
- `outputs/circular_distractor_timing_generalisation/metrics/timing_comparisons.csv`
- `outputs/circular_distractor_timing_generalisation/metrics/timing_summary.json`

</details>

<details>
<summary>2026-07-28 - Evaluate post-hoc persistence 0.80 distractor selectivity</summary>

Action:

- Added `configs/state_persistence_080_distractor_selectivity.yaml` without
  modifying the completed candidate-screen grid.
- Evaluated state persistence `0.80` on the five retained distractor-trained
  circular checkpoints using 1,024 trials per condition and the existing
  frozen evaluation banks.
- Recorded the result in
  `docs/reports/state_persistence_080_distractor_selectivity.md`.

Result:

- Mean clean proportional impairment was `+1.629`; all five checkpoints failed
  the original 20% clean-preservation ceiling.
- Mean distractor proportional impairment was `+1.252`.
- Distractor-minus-clean selectivity was `-0.377 ± 0.426`, with a student-t 95%
  interval of `[-0.906, 0.151]` and a positive direction in `1/5` checkpoints.
- Persistence `0.80` therefore produced broad circular-memory degradation, not
  increased distractor selectivity.

Interpretation:

- This is a post-hoc descriptive extension, not part of the original fixed
  strength grid and not confirmatory evidence.

Recorded outputs:

- `outputs/state_persistence_080_distractor_selectivity/metrics/state_persistence_080_grid.csv`
- `outputs/state_persistence_080_distractor_selectivity/metrics/state_persistence_080_metadata.json`

</details>

<details>
<summary>2026-07-28 - Add task coverage to perturbation meeting notes</summary>

Action:

- Added explicit task-application bullets for every perturbation in
  `docs/reports/meeting.md`.
- Distinguished operators evaluated across both circular and N-back families
  from the circular-only distractor-window manipulation and the not-yet-run
  Gaussian comparator.

Interpretation:

- The meeting notes now make clear which task and condition received each
  intervention without implying that the planned Gaussian control was part of
  the completed candidate screen.

</details>

<details>
<summary>2026-07-28 - Convert supervisor-meeting script to implementation-explicit speaking notes</summary>

Action:

- Reworked `docs/reports/meeting.md` into sectioned bullet-point speaking notes.
- Added the exact model-level implementation, evaluated strength grid,
  literature motivation, and interpretation boundary for P1--P7, including the
  task-specific channel boundaries for sensory and distractor-window gain.
- Clarified that state persistence was recovered by correctly relabelling the
  earlier whole-update operator after an implementation audit; its relevance to
  psilocybin-associated behaviour comes from the effects it reproduced, not
  from a prior biological mapping.

Interpretation:

- The meeting notes now distinguish literature-motivated translations,
  operational competing hypotheses, and mechanistic or generic-disruption
  controls.

Recorded output:

- `docs/reports/meeting.md`

</details>

<details>
<summary>2026-07-28 - Add supervisor-meeting script for perturbation results</summary>

Action:

- Expanded `docs/reports/meeting.md` from a prompt list into a spoken meeting
  script covering the circular distractor task, context-cued N-back
  implementation, baseline competence, task-family relationship, candidate
  perturbations, literature motivation, results, limitations, and next steps.
- Added a short fallback summary and three explicit questions for supervisor
  feedback.

Interpretation:

- The script preserves the report's claim boundary: persistence `0.95` is a
  descriptive computational candidate, not a biological model of psilocybin,
  and matched-cost Gaussian specificity remains outstanding.

Recorded output:

- `docs/reports/meeting.md`

</details>

<details>
<summary>2026-07-28 - Replace dose language with perturbation-strength terminology</summary>

Action:

- Replaced “dose-like”, “dose selection”, and “dose differences” in the
  supervisor report and Q&A with perturbation-strength, parameter-level, or
  scaling terminology.
- Updated and regenerated the persistence-strength figure so its visible title
  reads “Response across carried-state persistence scales”.
- Rebuilt `docs/reports/Perturbations.pdf`.

Interpretation:

- Persistence values such as `0.95` remain abstract RNN scaling parameters and
  are not presented as pharmacological doses or dose analogues.

</details>

<details>
<summary>2026-07-28 - Soften candidate-report title for supervisor handoff</summary>

Action:

- Retitled the supervisor report as “Exploring Candidate RNN Perturbations for
  Psilocybin-Related Working-Memory Effects”.
- Rebuilt `docs/reports/Perturbations.pdf` without changing its scientific
  content or interpretation.

</details>

<details>
<summary>2026-07-28 - Package perturbation report for supervisor handoff</summary>

Action:

- Renamed the compiled candidate-report deliverable to
  `docs/reports/Perturbations.pdf`.
- Added `docs/reports/Perturbations_QA.md` with anticipated supervisor
  questions and evidence-calibrated answers.
- Removed generated LaTeX build intermediates from `docs/reports/` while
  retaining `.tex` sources, bibliography sources, and final PDF deliverables.

Interpretation:

- The handoff now separates the concise scientific report from supporting
  discussion preparation while preserving the source needed to rebuild it.

Recorded outputs:

- `docs/reports/Perturbations.pdf`
- `docs/reports/Perturbations_QA.md`
- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`

</details>

<details>
<summary>2026-07-28 - Tighten scientific provenance in candidate supervisor write-up</summary>

Action:

- Revised `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
  after a scientific review.
- Disclosed that persistence `0.95` was nominated in the earlier three-seed
  pilot and that three of the ten N-back checkpoints overlap with that
  nomination analysis.
- Added direct citations for the slowing, load-selectivity, and filtering
  analogues; separated trained-RNN slow-manifold evidence from biological
  persistent-firing evidence.
- Replaced “strongest computational account” with descriptive-rule wording and
  added the omitted circular-training noise and gradient-clipping settings.

Interpretation:

- The numerical results are unchanged. The revised wording makes clear that
  this is a circular-family retest with partially overlapping N-back evidence,
  not a fresh blind screen or a matched-cost specificity result.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`

Verification:

- `latexmk -pdf -interaction=nonstopmode
  full_candidate_perturbation_scientific_writeup.tex` completed with resolved
  citations and references.
- The rebuilt eight-page PDF was visually inspected.

</details>

<details>
<summary>2026-07-28 - Add supervisor-facing limitations and next-steps section</summary>

Action:

- Added a dedicated `Limitations and next steps for supervisor discussion`
  section to
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex`.
- Split the new content into two explicit blocks:
  - key limitations of the current candidate-versus-native evidence,
  - concrete immediate next experiments and analysis decisions to discuss.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`

Verification:

- `latexmk -pdf full_candidate_perturbation_scientific_writeup.tex` completed.

</details>

<details>
<summary>2026-07-28 - Remove task schematic figure from candidate report</summary>

Action:

- Removed Figure 1 (task schematic) from
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex`.
- Updated Methods wording to avoid dangling references to `fig:tasks`.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf` (7 pages)

Verification:

- `pdflatex` completed successfully.

</details>

<details>
<summary>2026-07-28 - Clarify perturbation equation symbols</summary>

Action:

- Added explicit symbol definitions to the perturbation section in:
  - `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
  - `docs/dissertation/chapters/04_methods.tex`
- Defined shared meanings for `x_t`, `h_t`, `z_t`, `W_*`, `b_*`, `\alpha`,
  `g`, `\mathbf{g}`, `\sigma`, `\varepsilon_t`, `p`, and `s` so equations are
  interpretable without external code context.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`
- `docs/dissertation/main.pdf`

Verification:

- `pdflatex` completed for both report and dissertation.

</details>

<details>
<summary>2026-07-27 - Add distractor+N-back schematic notebook cell</summary>

Action:

- Added notebook cells in `notebooks/02_working_memory_task_schematic.ipynb`
  that regenerate a four-panel schematic in the original visual style:
  circular trial with mid-delay distractor (A–B) and 2-back N-back example
  (C–D).

Recorded outputs:

- `outputs/figures/schematics/working_memory_tasks_schematic.png`
- `outputs/figures/schematics/working_memory_tasks_schematic.pdf`

</details>

<details>
<summary>2026-07-27 - Add competence-gate rationales to candidate write-up</summary>

Action:

- Stated N-back competence thresholds and operational rationales for circular
  retention, fixation, settling, preservation, and N-back gates in
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex`.
- Mirrored the same gate text in dissertation Methods and removed the residual
  N-back pilot-reuse sentence there.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`
- `docs/dissertation/chapters/04_methods.tex`

Verification:

- Write-up `pdflatex` completed successfully.

</details>

<details>
<summary>2026-07-27 - Remove prior-iteration phrasing from candidate write-up</summary>

Action:

- Rewrote Methods/Results/Discussion/Conclusion in
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex` so the
  report stands alone without references to earlier report versions, pilots,
  or “now established” validity language.
- Removed the remaining three-seed / earlier exploratory-screen disclosure
  from Methods.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`

Verification:

- Write-up `pdflatex` completed successfully.

</details>

<details>
<summary>2026-07-27 - Tighten candidate-screen write-up for supervisor send</summary>

Action:

- Removed the Introduction from
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex`.
- Softened the title to mark a native-baseline persistence screen.
- Clarified that non-P3b operators apply every timestep on the frozen forward
  pass; mirrored that sentence in dissertation Methods.
- Renamed the first Results heading to “descriptive profile”.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf` (7 pages)
- `docs/dissertation/chapters/04_methods.tex`

Verification:

- Write-up `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed.

</details>

<details>
<summary>2026-07-27 - Generate distractor-circular and N-back baseline figures</summary>

Action:

- Added `src/wm_rnn/baseline_competence_figures.py` to regenerate parallel
  competence and hidden-state figure suites for the current distractor-trained
  circular family and the screened N-back family.
- Generated pool competence, delay/load, PCA, stability, and
  decoded-angle / match-probability panels.
- Inserted dissertation Results section
  `docs/dissertation/chapters/05_results_baseline.tex` and renumbered later
  result/discussion chapter files.

Recorded result:

- Circular representative seed `20260735`: clean delay errors
  `1.71/1.86/2.10/2.61°` at delays 10/20/40/80; distractor delay-20
  `2.92°`; delay-period hidden decode error `0.41°`.
- N-back representative seed `20260916`: 0-back accuracy/discriminability
  `1.000`; 2-back `0.986/0.971`.

Recorded outputs:

- `outputs/baseline_competence_figures/circular_trained_distractor/`
- `outputs/baseline_competence_figures/nback_screened_final/`
- `docs/reports/figures/baseline_competence/*.png`
- `docs/dissertation/chapters/05_results_baseline.tex`
- `docs/dissertation/main.pdf`

Verification:

- `pytest tests/test_baseline_competence_figures.py` passed.
- `python -m wm_rnn.baseline_competence_figures --family all` completed.
- Dissertation `pdflatex` recompiled after chapter insert.

</details>

<details>
<summary>2026-07-27 - Define RNN architecture explicitly in Methods</summary>

Action:

- Split Methods into a dedicated Network architecture subsection plus Training
  and checkpoint replication.
- Defined drive, leaky native update, linear readout, dimensions, zero
  initialisation, and nonlinearity-after-mixing caveat.

Recorded outputs:

- `docs/dissertation/chapters/04_methods.tex`
- `docs/dissertation/main.pdf`

Verification:

- `pdflatex` completed successfully.

</details>

<details>
<summary>2026-07-27 - Polish dissertation Intro, Aims, Methods, Discussion</summary>

Action:

- Trimmed Introduction so literature detail lives in the lit-review section.
- Added a provisional claim-map table to Aims.
- Clarified checkpoint definition and prior structured-noise background in
  Methods.
- Expanded Discussion around the strong N-back / weak circular split, human
  signature comparison, claim boundary, and next specificity step.
- Tightened Conclusion to match present evidence weight.

Recorded outputs:

- `docs/dissertation/chapters/01_introduction.tex`
- `docs/dissertation/chapters/03_aims.tex`
- `docs/dissertation/chapters/04_methods.tex`
- `docs/dissertation/chapters/08_discussion.tex`
- `docs/dissertation/chapters/09_conclusion.tex`
- `docs/dissertation/main.pdf` (13 pages)

Verification:

- `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed with no citation errors.

</details>

<details>
<summary>2026-07-27 - Draft dissertation literature-review section</summary>

Action:

- Replaced the literature-review scaffold in
  `docs/dissertation/chapters/02_literature.tex` with a quality-weighted
  synthesis drafted from the vault core set.
- Expanded `docs/dissertation/references.bib` with the newly cited core
  sources.

Recorded result:

- Literature section now covers recurrent WM dynamics, acute behavioural
  signatures, psychedelic dynamics theories/models, serotonergic gain
  mechanisms, and the modelling gap.
- Direct evidence, modelling analogies, and speculative translations are
  separated; Cano-Colino distractor constraint and Stoliker/Bredenberg
  structured-variability points are included.

Recorded outputs:

- `docs/dissertation/chapters/02_literature.tex`
- `docs/dissertation/references.bib`
- `docs/dissertation/main.pdf`

Verification:

- `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed with resolved citations.

</details>

<details>
<summary>2026-07-27 - Switch dissertation scaffold to two-column REVTeX</summary>

Action:

- Converted `docs/dissertation/main.tex` from single-column `report` class to
  two-column `revtex4-2` (APS/PRL reprint style).
- Demoted chapter/section hierarchy to section/subsection/subsubsection and
  widened floats for two-column layout.

Recorded outputs:

- `docs/dissertation/main.tex`
- `docs/dissertation/chapters/*.tex`
- `docs/dissertation/main.pdf`

Verification:

- PDF recompiled in two-column REVTeX form.

</details>

<details>
<summary>2026-07-27 - Create full-dissertation LaTeX scaffold from candidate report</summary>

Action:

- Created `docs/dissertation/` as the growing full-dissertation draft.
- Migrated the candidate-screen report into chapters for introduction,
  methods, results I, discussion, and conclusion.
- Added scaffold chapters for literature review, aims/claim boundary,
  specificity results, and dynamical explanation.
- Left the short report in `docs/reports/` as a paper-style snapshot with a
  pointer to the scaffold.
- Added vault note `wiki/writing/dissertation-scaffold.md`.

Recorded outputs:

- `docs/dissertation/main.tex`
- `docs/dissertation/chapters/*.tex`
- `docs/dissertation/main.pdf` (29 pages)
- `docs/dissertation/README.md`
- `wiki/writing/dissertation-scaffold.md`

Interpretation:

- The dissertation spine now matches the end-goal story: lit review, aims,
  methods, candidate screen, specificity, dynamics, discussion.
- Chapters 6 and 7 remain empty until those experiments are run.

Verification:

- `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed; `main.pdf` is 29 pages.

</details>

<details>
<summary>2026-07-27 - Remove abstract pending end-of-project rewrite</summary>

Action:

- Removed the abstract from
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex` so the
  abstract can be written after the remaining experiments are complete.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`

</details>

<details>
<summary>2026-07-27 - Expand candidate-perturbation Methods with lit-to-operator maps</summary>

Action:

- Rewrote the Candidate perturbations Methods section into P1-P7
  subsubsections, each with the implemented equation, literature motivation,
  and an explicit translation boundary.
- Defined checkpoint as one independently trained frozen network.
- Added catalogue-only P5 to the perturbation table for continuity.
- Clarified the P6 versus P7 retention wording in Results.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`

Interpretation:

- Methods now make clear that P1/P2 are the closest Herzog/REBUS gain
  analogues, P3 is a task-level afferent translation, P4/P7 are mechanistic
  controls, P5 is the deferred generic comparator, and P6 is an operational
  nominee rather than a biological mapping.

Verification:

- PDF recompiled (8 pages after Methods expansion).

</details>

<details>
<summary>2026-07-27 - Revise full-candidate write-up for claim honesty</summary>

Action:

- Revised
  `docs/reports/full_candidate_perturbation_scientific_writeup.tex` after an
  internal review, without changing the underlying numerical artefacts.
- Updated the companion results note to match the revised claim boundary.

Recorded result:

- Reframed the study as exploratory nomination then independent circular
  retest, with partial N-back reuse disclosed.
- Reported seed-level student-$t$ 95% intervals; circular settling/delay/
  distractor intervals include zero, while N-back load selectivity excludes
  zero.
- Replaced "pre-specified majority" wording with an accurate description of the
  majority-direction rule implemented in the summary code.
- Added near-miss settling magnitudes, the persistence-versus-conserved
  time-constant asymmetry argument, seed-denominator disclosure, P5 catalogue
  footnote, 23 non-neutral shared profiles, latency-invalid count, and a
  labelled secondary Gaussian pilot hint.
- Set the next experiment priority to matched-cost specificity against
  Gaussian disruption.

Recorded outputs:

- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`
- `docs/reports/full_candidate_perturbation_1024_results.md`

Interpretation:

- The scientific claim is unchanged in substance but narrower and more honest:
  descriptive native-baseline sufficiency for persistence `0.95`, not robust
  circular intervals, uniqueness against generic damage, or biology.

Verification:

- Exact intervals recomputed from the frozen CSVs.
- Near-miss clean delay-20 settling cells confirmed latency-valid.
- PDF recompiled after the prose revision (7 pages; float/layout warnings
  cleared).

</details>

<details>
<summary>2026-07-27 - Rerun all candidates on the trained-distractor circular family</summary>

Action:

- Added a frozen 1,024-trial evaluation configuration using the five retained
  distractor-trained circular checkpoints.
- Reran all seven circular perturbation families and their full strength grids
  on CUDA.
- Reused the completed N-back signature table because its checkpoints, task,
  perturbations, and evaluation settings were unchanged.
- Regenerated the cross-task summaries and figures, then revised and compiled
  the scientific report.

Recorded result:

- State persistence `0.95` remained the only operator-strength profile meeting
  the complete descriptive majority rule.
- Slowing with relative preservation, long-delay selectivity, and distractor
  selectivity had the predicted sign in 4/5 circular checkpoints; N-back load
  selectivity had the predicted sign in 10/10 checkpoints.
- Mean circular effects were small: proportional clean impairment `+0.049`,
  settling change `+0.104` steps, delay selectivity `+0.089`, and distractor
  selectivity `+0.015`.
- Native circular distractor competence was retained: all baseline
  distractor-condition errors were between 3.46 and 4.83 degrees and all
  trials met the settling criterion.
- The rerun took 590.7 seconds (9 minutes 51 seconds).

Recorded outputs:

- `configs/full_candidate_perturbation_trained_distractor_1024.yaml`
- `outputs/full_candidate_perturbation_trained_distractor_1024/`
- `docs/reports/full_candidate_perturbation_1024_results.md`
- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/full_candidate_perturbation_scientific_writeup.pdf`
- `docs/reports/figures/full_candidate_perturbation/`

Interpretation:

- Training the circular checkpoints with distractors resolves the former
  out-of-distribution validity problem.
- The selected pattern is descriptive and not universal across circular
  checkpoints. It establishes computational sufficiency for further mechanism
  analysis, not a unique perturbation, a population-level human result, or a
  biological model of psilocybin.

Verification:

- Focused runner and evaluator suite: `11 passed`.
- Complete repository suite: `216 passed`.
- The seven-page PDF compiled with resolved citations and references and was
  visually inspected page by page.

</details>

<details>
<summary>2026-07-27 - Train the single-item circular distractor family</summary>

Action:

- Added a balanced two-batch training sampler for the original one-item
  fixation-gated circular task: one clean batch and one distractor-present
  batch are used exactly once per shuffled block.
- Kept the 33-input, 64-hidden-unit `tanh` CTRNN architecture unchanged.
- Used a random five-step tuned distractor centred in the delay, with cue,
  delay, and pre-cue timing sampled from the original training choices.
- Trained six candidate seeds for 4,000 updates on the NVIDIA GeForce RTX 3060
  Laptop GPU and evaluated each on 1,024 paired clean and 1,024 paired
  distractor trials at delay 20.

Recorded result:

- Five checkpoints met the development competence criteria: seeds `20260731`,
  `20260732`, `20260733`, `20260735`, and `20260736`.
- Across retained checkpoints, mean angular error averaged 2.70 degrees for
  clean trials and 3.54 degrees for distractor trials; the mean distractor cost
  was 0.85 degrees.
- Seed `20260734` was preserved but not retained: clean error was 15.75 degrees
  and distractor error was 17.46 degrees, exceeding the configured 10/15-degree
  competence limits despite valid fixation performance.
- All retained checkpoints exceeded 0.97 fixation accuracy in both conditions.
- Total wall time for the initial five candidates and replacement pass was
  approximately 46 minutes 52 seconds.

Recorded outputs:

- `configs/fixation_circular_distractor_working_memory.yaml`
- `outputs/fixation_circular_distractor_working_memory/metrics/fixation_circular_distractor_working_memory_pool_summary.json`
- `outputs/fixation_circular_distractor_working_memory/metrics/fixation_circular_distractor_working_memory_pool.csv`
- `outputs/fixation_circular_distractor_working_memory/seed_sweep/`

Interpretation:

- The project now has a competent circular family in which distractor
  filtering is trained rather than imposed only at evaluation. This supports a
  meaningful clean-versus-distractor perturbation contrast.
- These are baseline competence results, not evidence that any perturbation
  reproduces a human psilocybin signature. The distractor cost is a model
  contrast and is not fitted to a human effect size.

Verification:

- Focused distractor/training suite: `26 passed`.
- Complete repository suite: `213 passed`.
- Python compilation, CLI help, and `git diff --check` passed.

</details>

<details>
<summary>2026-07-27 - Pending - Clarify active repository structure and archive historical guidance</summary>

Purpose:

- Make the current circular-plus-N-back experiment easy to distinguish from
  superseded baselines, failed task rescues, frozen preregistration records,
  and ignored generated data.

Changes:

- Simplified `README.md` to a high-level account of the project title, model,
  task families, perturbations, and scientific claim boundary.
- Added `docs/repository-map.md` and `configs/README.md` as current navigation
  and configuration-status indexes.
- Replaced the stale categorical `docs/model-architecture.md` with a concise
  description of the current circular and N-back families, while preserving
  the original document as
  `docs/archive/categorical-model-architecture.md`.
- Added an archive guide and corrected current circular-model names and paths
  in explanatory documents.
- Updated the two active baseline notebooks from the obsolete
  `yang_fixation_circular_working_memory` paths to
  `fixation_circular_working_memory`.
- Removed ignored `tmp/` and `.pytest_cache/` directories after verifying both
  targets were inside the repository.

Preservation:

- Did not remove or alter checkpoints, metrics, figures, other contents of
  `outputs/`, the local `.venv/`, Git stashes, or the untracked manuscript
  files in `docs/reports/`.
- Left frozen and historically referenced configuration files at their
  original paths; `configs/README.md` classifies them without breaking
  preregistration or test references.

Verification:

- Parsed both edited notebooks as JSON.
- Confirmed all canonical paths named by the new repository map exist.
- `git diff --check` passed.
- The complete repository suite passed: `316 passed`.

</details>

<details>
<summary>2026-07-27 - Pending - Reduce the active Python package to the current experiment</summary>

Purpose:

- Remove accumulated one-off development, rescue, analysis, and reporting
  pipelines from the active package while preserving their outputs,
  documentation, and Git history.

Changes:

- Reduced `src/wm_rnn/` from 51 Python files to 26 including `__init__.py`
  (25 substantive modules).
- Reduced active source from approximately 20,900 to 8,742 nonblank,
  non-comment lines.
- Removed retired fixed-point, stability, PCA, movie, structured-noise,
  failed-Family-B, phased perturbation-scoring, and additive N-back execution
  modules.
- Removed tests dedicated exclusively to those retired implementations.
- Made the current N-back candidate evaluator self-contained for its operator
  profile, checkpoint-manifest, hashing, and atomic JSON needs.
- Moved per-seed N-back configuration construction into the current screened
  checkpoint-pool module.
- Removed the failed Family B acceptance hook from the retained circular seed
  sweep before retiring obsolete baseline-only evaluation utilities.
- Updated the active notebook and repository map so they do not import or
  advertise removed modules.

Preservation:

- All recorded checkpoints, metrics, arrays, figures, PDFs,
  preregistrations, failure audits, and changelog history remain in place.
- The removed executable implementations remain recoverable from Git commit
  `d64b2cf`.

Verification:

- `python -m compileall -q src/wm_rnn` passed.
- The remaining complete suite passed: `208 passed`.
- The edited notebook parses as valid JSON.

</details>

<details>
<summary>2026-07-27 - Pending - Create scientific LaTeX perturbation write-up</summary>

Purpose:

- Convert the full 1,024-sample candidate-only evaluation into a
  dissertation-ready scientific paper with task-first Methods, formal
  perturbation definitions, quantitative Results, and calibrated
  interpretation.

Artifacts:

- Added a seven-page REVTeX manuscript and dedicated bibliography under
  `docs/reports/`.
- Added reproducible publication figures for the two task structures, the full
  29-profile signature screen, and the state-persistence dose/seed response.
- Compiled the final PDF with all citations and cross-references resolved.

Verification:

- Checked all manuscript numbers against the full-run summary and raw
  checkpoint-level contrasts.
- Rendered and visually inspected all seven PDF pages; figures, tables,
  equations, captions, columns, and references are legible without clipping.
- The full repository suite remains green with 316 tests passed.
- The manuscript states that settling is an analogue rather than reaction
  time, that the distractor baseline is weak, and that computational
  sufficiency does not identify a biological psilocybin mechanism.

</details>

<details>
<summary>2026-07-27 - Pending - Freeze full 1,024-sample candidate-only run</summary>

Purpose:

- Expand the exploratory screen to all five circular checkpoints and all ten
  competent N-back checkpoints with 1,024 trials or sequences per cell.

Design:

- Retain the pilot's fixed candidate grids while omitting Gaussian noise,
  cost matching, calibration, hybrids, and confirmatory inference.
- Compare every candidate directly with its checkpoint's native baseline on
  identical task samples.
- Preserve the existing settling, delay, distractor, and N-back load
  contrasts. The weak out-of-distribution distractor baseline remains an
  explicit validity limitation.

Implementation:

- Added a separate full-run config, preregistration, and CUDA orchestration
  entry point.
- Generalized the pilot evaluators without changing their frozen default
  designs.
- Verified all five circular checkpoint hashes, all ten retained N-back
  checkpoints, exact candidate grids, and 1,024-sample task-bank dimensions
  before outcome evaluation. Fourteen focused evaluator tests pass.

Outcome:

- The uninterrupted CUDA run completed in 1,075.9 seconds (17 minutes
  56 seconds), producing 945 circular grid rows and 600 N-back evaluation
  cells.
- State persistence `0.95` remained the sole complete descriptive profile:
  slowing/preservation, delay selectivity, and distractor selectivity were
  positive in 5/5 circular checkpoints, and N-back load selectivity was
  positive in 10/10 checkpoints.
- Mean contrasts were +1.38% delay-20 error, +0.194 settling steps, +0.264
  long-minus-short delay selectivity, +0.159 distractor selectivity, and +0.250
  N-back load selectivity.
- Native distractor performance remained weak (25.8–50.4 degree mean error;
  26.7–35.7% settled), so the distractor component is provisional despite its
  consistent direction.
- Added a full-run report and regenerated the cross-task summary and
  seed-point figure.

</details>

<details>
<summary>2026-07-27 - Pending - Freeze exploratory two-task signature pilot</summary>

Purpose:

- Create a small, outcome-free first pass that can provide preliminary
  dissertation-report material without retraining either model family.

Design:

- Reuse three competent original circular-task checkpoints and three competent
  N-back checkpoints.
- Screen every implemented single perturbation family on fixed strength grids
  with 256 trials or sequences per cell.
- Treat candidate-versus-native contrasts as primary and Gaussian state noise
  as a secondary descriptive control.
- Measure circular-task settling, delay, and distractor contrasts alongside the
  N-back 2-back-versus-0-back contrast.

Implementation:

- Added isolated circular and N-back pilot evaluators with explicit frozen
  checkpoint selection, hash/source validation, fresh deterministic task
  banks, fixed grids, and three technical replicates for heterogeneous gain and
  Gaussian noise.
- The circular runner requires an explicit `--execute` flag; its default
  command only expands the outcome-free design.
- Added 14 focused pilot tests. The complete repository suite passes with
  313 tests passed and one skipped before checkpoint evaluation.

Outcome:

- Both pilots completed on CUDA without retraining. The circular sweep took
  135.6 seconds and N-back took 75.6 seconds.
- State persistence `0.95` was the only setting to meet the formal complete
  pilot pattern: slowing/preservation, delay selectivity, distractor
  selectivity, and N-back load selectivity were each positive in 3/3 seeds.
- The independent audit found that native distractor performance was already
  weak (31.5–53.7 degree mean error; about 26% settled), making that component
  provisional and unsuitable for a strong recovery claim.
- Added a reproducible cross-task summary, seed-point figure, and concise
  tomorrow-ready results report. After summary integration, the full suite
  passes with 315 tests.

Claim boundary:

- This is an exploratory three-checkpoint screen. It can nominate perturbation
  directions for a later independently banked run, but cannot establish
  statistical signature recovery, biological equivalence, or mechanistic
  uniqueness.

</details>

<details>
<summary>2026-07-27 - Current commit - Finalize additive N-back experiment</summary>

Purpose:

- Close the seven-phase registered run, verify every persisted artifact, and
  state the scientific result without converting a validity failure into a
  null effect.

Result:

- All seven phases completed in one uninterrupted, CUDA-synchronized attempt
  per phase, totalling 4498.9199 seconds of active execution time.
- The final checkpoint C2 table is header-only. All 14 non-P5 profile rows have
  `n_checkpoints = 0` and `not_testable_validity`.
- The dose summary contains no valid profile and the P5 registered-strength
  curve is empty because no non-neutral confirmatory or dose outcome was
  authorized.
- Every state-recorded artifact passed an independent SHA-256 recheck.
- Added `docs/reports/nback_additive_perturbation_results.md` with the
  literature/architecture rationale, complete phase audit, claim boundary,
  and clean next-experiment recommendation.

Conclusion:

- The trained N-back CTRNN remains a valid model of the Barrett-style task
  contrast, but this run neither recovers nor rejects the psilocybin-like
  load-selective signature. C2 is not testable because the frozen common-cost
  comparator support failed.

</details>

<details>
<summary>2026-07-27 - ec6a631 - Complete gated N-back dose phase</summary>

Purpose:

- Enforce the rule that stronger-dose ordering is evaluated only after a
  profile passes the complete confirmatory validity chain.

Result:

- All 150 profile/checkpoint dose cells were explicitly recorded as skipped
  with `global_confirmatory_validity_failure`.
- Zero dose cells were valid, and zero dose-array artifacts were generated.
- No stronger candidate or Gaussian-P5 perturbation was evaluated.
- The phase completed in one CUDA-synchronized attempt with 468.7234 seconds
  of active execution time, spent on atomic validity-record persistence rather
  than perturbation inference.

Interpretive boundary:

- Dose ordering is NA, not failed or reversed. It cannot be used to rescue the
  untestable matched-cost C2 comparison.

</details>

<details>
<summary>2026-07-27 - 2cdcaa8 - Complete gated N-back confirmatory phase</summary>

Purpose:

- Verify confirmatory-bank baseline transport and enforce the frozen all-ten
  matched-cost barrier before any non-neutral C2 evaluation.

Result:

- All 20 native checkpoint/load cells passed baseline transport.
- Across checkpoints, 0-back accuracy ranged from `0.999946` to `1.000000`.
  Two-back accuracy ranged from `0.956868` to `0.987142`, and 2-back
  discriminability ranged from `0.918620` to `0.973063`.
- All 300 non-neutral profile/checkpoint/load cells were explicitly recorded
  as skipped with `global_cost_validity_failure`.
- The only 20 confirmatory raw-array artifacts are native baselines. No
  candidate or Gaussian-P5 non-neutral confirmatory array was generated.
- The phase completed on CUDA in one uninterrupted attempt with 961.4707
  seconds of active execution time.

Interpretive boundary:

- The RNN checkpoints remain competent on the independent N-back task bank.
  The confirmatory C2 result is NA because matched-cost comparator validity
  failed, not because the networks failed the working-memory task and not
  because a candidate-versus-P5 effect was null.

</details>

<details>
<summary>2026-07-27 - e63934b - Pass N-back neutral-confirmatory firewall</summary>

Purpose:

- Verify exact neutral behavior again on the independent confirmatory 0-back
  and 2-back task banks before allowing any non-neutral confirmatory outcome.

Result:

- All 300 profile/checkpoint/load cells passed exact equality for logits and
  hidden states, with zero maximum absolute difference and zero additive
  sequence-loss cost.
- The audit covered all 15 profiles, all ten retained checkpoints, and both
  condition codes. Deterministic profiles made eight registered comparisons
  per cell and stochastic profiles made 24.
- The phase completed on the NVIDIA GeForce RTX 3060 Laptop GPU in one
  uninterrupted, CUDA-synchronized attempt with 1328.3175 seconds of active
  execution time.
- No non-neutral confirmatory outcome was evaluated during this firewall.

</details>

<details>
<summary>2026-07-27 - d53b76d - Complete N-back held-out cost check</summary>

Purpose:

- Test calibrated strengths on the independent 1,024-sequence 0-back bank
  using the frozen cost band, paired-bootstrap precision, and candidate-P5
  cost-gap gates.

Result:

- 53 of 150 checkpoint/profile cells were valid after propagating unavailable
  calibrations and applying all held-out gates.
- Gaussian comparator P5 transported on its nine calibrated checkpoints, with
  held-out additive costs from `0.044388` to `0.051666` nats, but remained
  invalid at checkpoint seed `20260913`; P5 therefore failed the required
  all-ten profile barrier.
- Confirmatory profile valid-checkpoint counts were: profile 1, 7/10; profile
  4, 1/10; profile 7, 0/10; profile 9, 2/10; profile 10, 4/10; and profile 12,
  8/10. No confirmatory profile passed all ten checkpoints.
- Beyond unavailable calibrations and the invalid P5 reference, held-out
  failures included candidate-P5 gaps above `0.005` and confidence half-widths
  above `0.005`; the largest observed gap and half-width were `0.0139692` and
  `0.0084422`, respectively.
- There were no nonfinite numerical failures.
- The phase completed on CUDA in one uninterrupted attempt with 377.7230
  seconds of active execution time.

Interpretive boundary:

- The frozen N-back candidate-versus-generic-noise C2 test is not globally
  testable in this run. Subsequent phases will retain explicit NA cells and
  descriptive validity information; they cannot be used to rescue or replace
  the failed matched-cost comparator.

</details>

<details>
<summary>2026-07-27 - 103f69f - Complete N-back additive calibration</summary>

Purpose:

- Run the frozen non-neutral calibration bank after the exact-neutral firewall
  passed, without consulting held-out cost-check or 2-back outcomes.

Result:

- 75 of 150 checkpoint/profile cells reached the registered
  `0.050 +/- 0.0025`-nat additive 0-back cost target.
- Profiles 1, 3, 9, 11, 12, and 13 calibrated on all ten checkpoints.
  Confirmatory profiles among these were 1 (response/drive gain above), 9
  (recurrent gain above), and 12 (conserved-integrator time constant below).
- Confirmatory profiles 4, 7, and 10 calibrated on 1, 0, and 5 checkpoints,
  respectively, and therefore failed the all-ten calibration barrier.
- Gaussian comparator P5 (profile 14) calibrated on 9 of 10 checkpoints.
  Checkpoint seed `20260913` reached only `0.0433148` nats at the registered
  maximum state-noise strength `0.1`, so extrapolation was correctly refused.
- All 75 failed cells were registered unreachable-strength failures; there
  were no nonfinite numerical failures.
- The phase completed on CUDA in one uninterrupted attempt with 1157.2079
  seconds of active execution time.

Interpretive boundary:

- Because P5 did not calibrate on all ten checkpoints, no candidate-versus-P5
  confirmatory profile is yet globally eligible. The frozen held-out phases
  will propagate and report this validity result without widening the grid or
  replacing the checkpoint.

</details>

<details>
<summary>2026-07-27 - e8fb324 - Pass N-back neutral-calibration firewall</summary>

Purpose:

- Execute the first registered additive N-back phase and verify that every
  operator implementation is exactly neutral before permitting any
  non-neutral calibration.

Result:

- Evaluated all 15 registered profiles across all 10 retained checkpoints.
- All 150 cells passed exact equality for logits and hidden states, with zero
  maximum absolute difference and zero additive sequence-loss cost.
- Profiles with stochastic streams were checked across all three registered
  replicates; deterministic profiles used the registered single stream.
- The phase completed on the NVIDIA GeForce RTX 3060 Laptop GPU in one
  uninterrupted, CUDA-synchronized attempt with 201.8215 seconds of active
  execution time.
- No non-neutral calibration artifact existed before the firewall passed.

</details>

<details>
<summary>2026-07-27 - 38f45d8 - Implement phased additive N-back perturbation runner</summary>

Purpose:

- Implement the frozen seven-phase CUDA experiment without evaluating a
  non-neutral perturbation before the runner, validity gates, persistence, and
  outcome calculations were independently tested.

File changes:

- `configs/nback_additive_perturbation.yaml`: Pins the preregistration,
  screened checkpoint pool, precision evidence, sample sizes, additive
  calibration target, and held-out matched-cost gates.
- `src/wm_rnn/nback_additive_perturbation_runner.py`: Implements the registered
  neutral-calibration, calibration, cost-check, neutral-confirmatory,
  confirmatory, dose, and finalize phases with exact seed banks, CUDA
  execution, numerical sentinels, global barriers, immutable runtime hashes,
  raw audit arrays, and descriptive outputs.
- `src/wm_rnn/nback_additive_outcomes.py`: Pools replicate counts and
  sequence-wise cross-entropy, applies baseline-transport gates, and computes
  the registered checkpoint-level C2 contrast and diagnostic t summary.
- `src/wm_rnn/nback_perturbation_state.py`: Extends atomic phase persistence
  with cumulative active-time ledgers, interrupted-attempt records, and
  cell-coupled timing deltas for truthful crash/resume timing.
- Focused tests cover exact neutral equivalence for every profile and
  stochastic replicate, calibration and cost gates, nonfinite sentinels,
  candidate-versus-P5 C2 calculations, descriptive latency/failure outcomes,
  runtime mutation refusal, and timing idempotence.

Validation:

- Focused additive runner and analysis suite: 100 passed.
- Full repository suite: 300 passed.
- No checkpoint, strength, Gaussian perturbation, or 2-back outcome was
  evaluated during implementation or testing.

</details>

<details>
<summary>2026-07-27 - 8d3c9c2 - Implement additive N-back experiment foundations</summary>

Purpose:

- Implement and verify the frozen pre-outcome foundations required before the
  phased N-back perturbation runner may evaluate any non-neutral setting.

File changes:

- `src/wm_rnn/nback_additive_calibration.py`: Encodes the exact profile and
  seed manifests, additive calibration, sequence-wise replicate averaging,
  monotonicity and no-extrapolation gates, deterministic paired bootstrap, and
  held-out P5 cost-match validation.
- `src/wm_rnn/nback_metrics.py`: Enforces exact generated-batch identity and
  finite binary inputs, and adds specificity, balanced accuracy, ordinary
  nonmatch performance, settling validity, and explicit response-failure
  metrics.
- `src/wm_rnn/nback_perturbation.py`: Uses additive sequence log-loss cost and
  retains the registered six-channel P3a boundary and C2 sign convention.
- `src/wm_rnn/nback_perturbation_state.py`: Provides atomic, hash-validated
  phased persistence, artifact identity checks, phase barriers, and safe
  resume.
- Focused tests cover all registered calibration failure modes, stochastic
  seed maps, additive rather than proportional cost, metric guards, P3a
  context protection, and persistence corruption/refusal behavior.

Validation:

- Focused additive, metric, adapter, and persistence tests: 42 passed.
- Full repository suite: 242 passed.
- No checkpoint, strength, Gaussian perturbation, or 2-back outcome was
  evaluated.

</details>

<details>
<summary>2026-07-27 - fc99475 - Preregister additive N-back perturbations</summary>

Purpose:

- Freeze the exact matched-cost N-back candidate-versus-Gaussian experiment
  after baseline precision passed and before constructing a non-neutral
  operator.

Frozen decisions:

- Fifteen stable operator/profile IDs, including six confirmatory profiles
  and P5 as comparator.
- 512 calibration, 1,024 held-out cost-check, and 1,024 confirmatory
  sequences per condition and checkpoint.
- Additive `0.050`-nat calibration with explicit monotonicity, bisection,
  no-extrapolation, precision, and P5-gap gates.
- Collision-free task and stochastic stream maps with nested P2/P5
  replicate averaging.
- N-back C2 as excess normalized load selectivity versus matched P5.
- Cross-task IUT integration with Family A C1/C3, fixed six-profile Holm
  family, validity/NA rules, dose-ordering, and claim boundaries.
- Hash-validated phased persistence and resume.

</details>

<details>
<summary>2026-07-27 - 8b69b8e - Record additive N-back precision pass</summary>

Purpose:

- Freeze the registered baseline-only precision result before constructing
  any perturbation experiment.

Result:

- Collected 8,192 fresh unperturbed 0-back sequence losses from each of ten
  checkpoints.
- Family-wide `sigma_upper` was `0.002274699`.
- The planning formula yielded `n_required = 3.9755`; the registered minimum
  therefore fixes `n_cost_check = 1024`.
- All validity, seed-map, persistence, bootstrap, and planning gates passed.
- Saved artifact hashes were independently verified.
- No operator, P5, strength grid, 2-back outcome, or calibration was run.

</details>

<details>
<summary>2026-07-27 - 0e5e813 - Implement additive N-back cost precision</summary>

Purpose:

- Implement the frozen baseline-only sequence-log-loss precision phase
  without constructing or evaluating a perturbation.

File changes:

- `src/wm_rnn/nback_additive_cost_precision.py`: Strictly loads the ten
  retained checkpoints, generates the registered 64 fresh 0-back batches per
  checkpoint, persists all sequence units and seeds, bootstraps the
  family-wide maximum SD, and derives the held-out cost-check sample size.
- `tests/test_nback_additive_cost_precision.py`: Covers seed addressing,
  descriptive statistics, deterministic chunked bootstrap, planning formula,
  complete audit artifacts, and absence of perturbation imports.

Validation:

- Focused precision tests: 6 passed.
- Full repository suite: 214 passed, 1 skipped.
- The full precision-reference experiment was not run before this
  implementation was committed.

</details>

<details>
<summary>2026-07-27 - 3da286e - Preregister additive N-back cost precision</summary>

Purpose:

- Replace unstable proportional ceiling-error matching with a frozen additive
  sequence-log-loss cost before any perturbation is computed.

Frozen decisions:

- Target `0.050` nats, held-out band `[0.040, 0.060]`, maximum 95% half-width
  `0.005`, and candidate-P5 gap `0.005`.
- Collect 8,192 fresh unperturbed 0-back sequences per checkpoint.
- Bootstrap the family-wide maximum SD and derive `n_cost_check` with the
  registered conservative variance formula.
- Use sequence-level resampling, non-overlapping future task banks, and a hard
  maximum of 8,192 sequences per cell.
- Prohibit operators, P5, 2-back, or strength calibration during this phase.

</details>

<details>
<summary>2026-07-27 - 06d4f67 - Freeze competent N-back checkpoint family</summary>

Purpose:

- Record the registered screened-pool terminal result before any
  calibration or perturbation work.

Result:

- Seeds `20260912-20260921` all passed the unchanged training and untouched
  competence gates.
- The target of ten retained checkpoints was reached after ten attempts,
  with no failures.
- Seeds `20260922-20260926` were left unattempted by the registered stop rule.
- No perturbation outcome was run or inspected.

Decision:

- Freeze the ten passing checkpoints as the N-back inferential family.
- Proceed only to a separately pre-registered baseline-only
  calibration-precision phase.

</details>

<details>
<summary>2026-07-27 - 530d300 - Implement competence-screened N-back pool</summary>

Purpose:

- Implement the frozen perturbation-blind selection procedure without
  changing task, model, training, metrics, or competence criteria.

File changes:

- `configs/nback_working_memory_screened_final.yaml`: Encodes the candidate
  pool, target of ten, disjoint data-bank seed stride, and isolated outputs.
- `src/wm_rnn/nback_screened_pool.py`: Trains candidates in frozen order,
  persists progress after every seed, resumes without rerunning completed
  candidates, records failures, and stops at the target or when impossible.
- `src/wm_rnn/nback_evaluation.py` and `train_nback.py`: Resolve the registered
  checkpoint seed stride while preserving historical configurations.
- Tests cover continuation, termination, training failures, invalid pools,
  disjoint banks, and exact config differences.

</details>

<details>
<summary>2026-07-27 - 060a1c8 - Preregister disjoint N-back bank seeds</summary>

Purpose:

- Prevent adjacent checkpoint-by-condition cells from sharing the same
  numeric random seed.

Frozen correction:

- In the screened pool only, use
  `bank_offset + 2 * checkpoint_seed + condition_index`.
- Preserve historical seed addressing for all prior configurations.
- Change no task, model, training, metric, gate, or selection decision.

</details>

<details>
<summary>2026-07-27 - a31558c - Preregister competence-screened N-back pool</summary>

Purpose:

- Replace the brittle ten-of-ten seed-family rule with a perturbation-blind
  competence-screened pool grounded in the exact Wan et al. N-back precedent.

Frozen decisions:

- Attempt seeds `20260912-20260926` once and in numerical order.
- Use validation offset `1100000` and untouched competence offset `1200000`.
- Retain the first ten checkpoints passing every unchanged competence gate.
- Continue after failures and stop when ten pass or ten passes become
  impossible.
- Inspect no perturbation outcome during checkpoint selection.

</details>

<details>
<summary>2026-07-27 - f225c2b - Record replacement final-family failure</summary>

Purpose:

- Enforce the replacement final-family stop rule after seed `20260911`.

Result:

- Training passed at step 6,000.
- Untouched 0-back, 2-back accuracy, and lure checks passed.
- 2-back discriminability was `0.888`, narrowly below `0.900`.
- Seeds `20260912-20` were not started; no perturbation was run.

Interpretation:

- The 10/10 family rule is stricter than the exact Wan et al. RNN precedent,
  which retained 10 competent networks from 12 trained initializations.
- A preregistered competence-screened pool is the evidence-aligned next
  design; thresholds remain unchanged.

</details>

<details>
<summary>2026-07-27 - d986f1b - Implement replacement final N-back configuration</summary>

Purpose:

- Materialize the frozen replacement final seed banks without training a
  checkpoint.

File changes:

- `configs/nback_working_memory_budget_final.yaml`: Added final seed
  `20260911`, validation offset `900000`, competence offset `1000000`, and
  isolated output paths.
- `tests/test_nback_training.py`: Added an exact config-difference guard.

</details>

<details>
<summary>2026-07-27 - f0082e1 - Preregister replacement final N-back family</summary>

Purpose:

- Freeze an entirely new ten-seed final family after the budget-rescue
  development pass.

Frozen decisions:

- Training seeds `20260911-20`.
- Validation offset `900000`.
- Untouched competence offset `1000000`.
- Exact budget-rescue task, CTRNN, training, metrics, gates, and
  stop-on-first-failure rule.

Interpretation:

- All ten checkpoints must pass before calibration or perturbation outcomes.

</details>

<details>
<summary>2026-07-27 - 8866701 - Record budget-rescue development pass</summary>

Purpose:

- Record the three-seed budget-rescue decision before freezing a replacement
  final family.

Result:

- Seeds `20260827-29` all passed training and untouched competence.
- 2-back accuracy was `0.965-0.970`, discriminability `0.929-0.944`, and lure
  accuracy `0.967-0.997`; 0-back remained effectively perfect.
- All three passed in `7,600-10,000` steps, before the old ceiling.

Interpretation:

- The wider maximum preserves development competence but was not needed by
  these three initializations.
- No perturbation outcome was inspected.

</details>

<details>
<summary>2026-07-27 - 64759b8 - Implement N-back Stage 2 budget rescue</summary>

Purpose:

- Materialize the sole frozen budget extension before training development
  seed `20260827`.

File changes:

- `configs/nback_working_memory_budget_rescue.yaml`: Added the 20,000-update
  Stage 2 maximum, new development seed, new seed-bank offsets, and isolated
  output paths.
- `tests/test_nback_training.py`: Added an exact config-difference guard.

</details>

<details>
<summary>2026-07-27 - 4158040 - Preregister N-back Stage 2 budget rescue</summary>

Purpose:

- Test whether the 12,000-update ceiling truncated late-learning N-back
  initializations.

Sole training change:

- Extend the Stage 2 maximum to 20,000 updates while retaining early stopping
  after two consecutive joint validation passes.

Seed separation:

- New development seeds `20260827-29`.
- New validation offset `700000` and competence offset `800000`.
- Prospective final seeds `20260911-20`.

Held fixed:

- Task, CTRNN, loss, rule allocation, optimizer, metrics, competence gates, and
  perturbation lockout.

</details>

<details>
<summary>2026-07-27 - 068ebe1 - Record final N-back family failure</summary>

Purpose:

- Enforce the final-family stop rule after seed `20260902` exhausted the
  frozen Stage 2 budget.

Result:

- Seed `20260901` passed.
- Seed `20260902` retained perfect 0-back but ended at 2-back accuracy `0.847`,
  discriminability `0.688`, and lure accuracy `0.835`.
- Seeds `20260903-10` were not started; no perturbation was run.

Interpretation:

- The failed seed showed a sharply improving but late 2-back trajectory at the
  12,000-update endpoint.
- The smallest next rescue is a training-budget extension with all scientific
  task and competence decisions held fixed.

</details>

<details>
<summary>2026-07-27 - e1237cb - Add outcome-free N-back perturbation adapters</summary>

Purpose:

- Fit the competent N-back task into the registered generic-noise-controlled
  perturbation regime without running a checkpoint outcome.

File changes:

- `src/wm_rnn/nback_perturbation.py`: Reuses applicable registered operators,
  fixes P3a to six stimulus channels, exposes one log-loss calibration unit per
  sequence, and defines normalized candidate-versus-P5 `C2_NBACK`.
- `tests/test_nback_perturbation.py`: Covers neutral operators, exclusion of
  rule-context channels from sensory gain, sequence-level costs, architecture
  guards, and the synthetic load-contrast sign.

Claim boundary:

- No checkpoint loading, strength calibration, perturbation outcome, or
  confirmatory score is included.

</details>

<details>
<summary>2026-07-27 - a4191b9 - Implement frozen final N-back configuration</summary>

Purpose:

- Materialize the final-family seed banks without training a final checkpoint.

File changes:

- `configs/nback_working_memory_final.yaml`: Added the exact successful rescue
  with final seed `20260901`, validation offset `500000`, competence offset
  `600000`, and isolated output paths.
- `tests/test_nback_training.py`: Added an exact config-difference guard.

</details>

<details>
<summary>2026-07-27 - b339814 - Preregister final N-back checkpoint family</summary>

Purpose:

- Freeze the ten-checkpoint final family after the independent three-seed
  development pass and before final training.

Frozen decisions:

- Training seeds `20260901-20260910`.
- New validation offset `500000` and untouched competence offset `600000`.
- Exact successful rescue task, architecture, training allocation, loss,
  metrics, competence gates, and stop-on-first-failure rule.

Interpretation:

- All ten checkpoints must pass before N-back perturbation calibration or
  outcome inspection.

</details>

<details>
<summary>2026-07-27 - 66a95c8 - Record N-back rescue development pass</summary>

Purpose:

- Record the three-seed development decision before freezing or training the
  final N-back family.

Result:

- Seeds `20260824-26` all passed training validation and untouched competence.
- 0-back accuracy/discriminability were effectively `1.000`.
- 2-back accuracy was `0.959-0.965`, discriminability `0.924-0.938`, and lure
  accuracy `0.971-0.988`.

Interpretation:

- The training-balance rescue transports across initialization.
- Final-family training is now eligible after a separate final seed-bank
  pre-registration.
- No perturbation outcome has been inspected.

</details>

<details>
<summary>2026-07-27 - ecbd01f - Add gated N-back seed-family runner</summary>

Purpose:

- Make remaining development and final N-back training reproducible and
  enforce stop-on-first-failure automatically.

File changes:

- `src/wm_rnn/nback_seed_sweep.py`: Added isolated per-seed output paths,
  competence evaluation, CSV/JSON summaries, and early stopping.
- `tests/test_nback_training.py`: Added per-seed configuration isolation
  coverage.

Context:

- Rescue seed `20260824` passed its two validation checks and untouched
  competence bank before this runner was added.

</details>

<details>
<summary>2026-07-27 - ca74be7 - Implement N-back training-balance rescue</summary>

Purpose:

- Implement the frozen high-load training-allocation rescue before training
  development seed `20260824`.

File changes:

- `src/wm_rnn/train_nback.py`: Added configurable shuffled rule blocks and
  rule-specific class weights.
- `src/wm_rnn/training_utils.py`: Added normalized class-weight support to
  masked cross-entropy without changing unweighted callers.
- `configs/nback_working_memory_balance_rescue.yaml`: Added the frozen
  1:3 rule allocation, 2-back match weight, and new seed banks.
- N-back tests cover exact rule counts, weighted-loss arithmetic, unchanged
  base configuration, and CUDA weighted loss.

</details>

<details>
<summary>2026-07-27 - 7efd0af - Preregister N-back training-balance rescue</summary>

Purpose:

- Test whether inefficient rule allocation and unweighted class prevalence
  caused the initial actual N-back competence failure.

Frozen changes:

- Use one 0-back and three 2-back homogeneous batches per shuffled Stage 2
  block.
- Weight 2-back match errors by `2.0` while retaining the six-to-twelve event
  prevalence and ordinary 0-back cross-entropy.
- Use new development seeds `20260824-26` and new validation/competence banks.

Held fixed:

- Task, 64-unit CTRNN, timings, generator, optimizer, maximum steps, metrics,
  competence gates, and perturbation lockout.

</details>

<details>
<summary>2026-07-27 - aed9cca - Record initial N-back baseline failure</summary>

Purpose:

- Enforce the actual N-back development stop rule after seed `20260821`
  exhausted Stage 2.

Recorded result:

- Stage 1 passed at steps 100 and 200.
- Final held-out 0-back accuracy and discriminability were both `1.000`.
- Final 2-back accuracy was `0.891`, discriminability `0.726`, hit rate
  `0.778`, false-alarm rate `0.052`, and one-back-lure accuracy `0.991`.

Interpretation:

- The actual N-back reformulation acquired the control rule and genuine lure
  rejection but missed the high-load competence gate.
- Equal rule sampling and unweighted 1:2 class prevalence produced an
  inefficient, conservative 2-back learner.
- Remaining development/final seeds and all perturbation outcomes remain
  uninspected. A training-only rescue requires a new pre-registration.

</details>

<details>
<summary>2026-07-26 - 7cfea6c - Implement competence-gated N-back baseline</summary>

Purpose:

- Implement Phase 1 of the frozen actual 0-back/2-back redesign without
  training a development seed.

File changes:

- `src/wm_rnn/nback_task.py`: Added deterministic exact-balance homogeneous
  N-back sequences with rule context, matched warm-up, and one-back lures.
- `src/wm_rnn/nback_metrics.py`: Added item-level Barrett metrics, corrected
  d-prime, per-sequence cross-entropy, and guarded settling metrics.
- `src/wm_rnn/train_nback.py`: Added the two-stage competence-gated trainer.
- `src/wm_rnn/nback_evaluation.py`: Added held-out 0-back/2-back competence
  evaluation and exact acceptance checks.
- `src/wm_rnn/training_utils.py`: Registered the N-back task and its 8-input,
  2-output model dimensions.
- `configs/nback_working_memory.yaml`: Added the frozen development
  configuration.
- `tests/test_nback_*.py`: Added generator, metrics, CPU training, and CUDA
  device coverage.

Verification:

- Focused N-back tests passed: 18 tests.
- Full suite passed: 180 tests.
- CUDA smoke test placed model, inputs, targets, states, logits, and loss on
  the NVIDIA GPU.
- No development seed or perturbation outcome was run.

</details>

<details>
<summary>2026-07-26 - ec6ad30 - Pre-register actual N-back redesign</summary>

Purpose:

- Replace the failed synthetic two-slot Family B baseline with an actual
  shared 0-back/2-back task grounded in human and RNN N-back literature.

File changes:

- `docs/nback_rnn_literature_and_architecture.md`: Recorded the human target,
  published RNN precedents, retained CTRNN rationale, and claim boundary.
- `docs/preregistration/nback_baseline_and_integration_preregistration.md`:
  Froze the task, architecture, competence-gated curriculum, metrics,
  development/final seeds, perturbation contrast, and stop rules.

Interpretation:

- The old R1-R3 failures concern baseline learnability, not perturbation
  mechanisms.
- The actual N-back baseline will supply the 0-back/2-back load dissociation;
  the stable delayed-response family remains responsible for delay and
  distractor signatures.
- No N-back code, training result, or perturbation outcome is included in this
  pre-registration commit.

</details>

<details>
<summary>2026-07-26 - Current commit - Record 128-unit rescue failure</summary>

Purpose:

- Enforce the R3 development stop rule and close the planned baseline rescue
  ladder.

Recorded result:

- Seed `20260807` passed fixation (`0.965-0.966`) but failed six of seven
  memory checks.
- Pooled clean error was `46.56` degrees; load2 clean was `46.83-73.12`
  degrees; load2 distractor was `48.99-75.03` degrees.
- Probability-output resultant lengths were `0.39-0.53`, again showing
  structured but incorrect response distributions.

Recorded outputs:

- `outputs/multicondition_working_memory_distribution_role_h128/seed_sweep/seed_20260807/`
- `docs/preregistration/family_b_hidden128_rescue_audit.md`

Interpretation:

- R3 failed. Seeds `20260808-09` were stopped, final seeds remain untouched,
  and no perturbation outcome was inspected.
- Distribution loss, role labelling, and doubled hidden capacity were each
  insufficient. Further work requires a new training/task-design decision.

</details>

<details>
<summary>2026-07-26 - fb85c49 - Implement 128-unit Family B rescue</summary>

Purpose:

- Implement and verify the frozen R3 capacity rescue before training any
  development seed.

File changes:

- `configs/multicondition_working_memory_distribution_role_h128.yaml`: Added
  the exact R2 configuration with only hidden size and output paths changed.
- `tests/test_multicondition_training.py`: Added an exact config-difference
  guard and a 128-unit CPU training smoke test.

Verification:

- `.venv\Scripts\python.exe -m pytest -q` passed: 162 tests.
- `git diff --check` passed.
- No R3 development checkpoint or perturbation outcome was generated.

</details>

<details>
<summary>2026-07-26 - 3c082ac - Pre-register 128-unit Family B rescue</summary>

Purpose:

- Freeze R3 before changing recurrent capacity or training its development
  seeds.

File changes:

- `docs/preregistration/family_b_hidden128_rescue_preregistration.md`: Fixed
  the sole hidden-size change, unchanged R2 components and competence gates,
  required checks, development/final seed separation, and terminal stop rule.
- `docs/changelog.md`: Recorded the R3 baseline-only pre-registration.

Interpretation:

- R3 tests whether 64-unit capacity caused the competence failure; it does not
  test a perturbation mechanism.
- Any failed development gate exhausts the planned rescue ladder before
  final-family training.

</details>

<details>
<summary>2026-07-26 - ac08bbe - Record stimulus-role rescue failure</summary>

Purpose:

- Enforce the R2 development stop rule after the first preregistered seed.

Recorded result:

- Seed `20260804` passed fixation (`0.967-0.969`) but failed all seven memory
  checks.
- Pooled clean error was `42.46` degrees; load2 clean was `62.79-65.81`
  degrees; load2 distractor was `66.22-67.64` degrees.
- Probability-output resultant lengths were `0.42-0.48`, showing structured
  but incorrect response distributions.

Recorded outputs:

- `outputs/multicondition_working_memory_distribution_role/seed_sweep/seed_20260804/`
- `docs/preregistration/family_b_stimulus_role_rescue_audit.md`

Interpretation:

- R2 failed. Seeds `20260805-06` were stopped, final seeds remain untouched,
  and no perturbation outcome was inspected.

</details>

<details>
<summary>2026-07-26 - cd031aa - Implement Family B stimulus-role rescue</summary>

Purpose:

- Implement and verify the frozen R2 task-interface rescue before training any
  development seed.

File changes:

- `src/wm_rnn/tuned_task.py`: Added an opt-in stimulus-role channel after the
  existing fixation and probe inputs, with explicit stable channel indices.
- `src/wm_rnn/training_utils.py`: Resolved the role-channel option from task
  configuration.
- `configs/multicondition_working_memory_distribution_role.yaml`: Added the
  exact preregistered R2 configuration.
- `tests/`: Added role timing, occupied-slot, distractor, probe-separation,
  input-dimension, and CPU training-smoke coverage.

Verification:

- `.venv\Scripts\python.exe -m pytest -q` passed: 160 tests.
- `git diff --check` passed.
- No development checkpoint or perturbation outcome was generated.

</details>

<details>
<summary>2026-07-26 - cb8cf9a - Pre-register Family B stimulus-role rescue</summary>

Purpose:

- Freeze R2 before adding a task-role input or training its development seeds.

File changes:

- `docs/preregistration/family_b_stimulus_role_rescue_preregistration.md`:
  Fixed the role-channel semantics, unchanged R1 components and competence
  gates, required tests, development/final seed separation, and stop rule.
- `docs/changelog.md`: Recorded the R2 baseline-only pre-registration.

Interpretation:

- R2 tests task-role ambiguity, not a perturbation mechanism.
- Any failed development gate stops R2 before final-family training.

</details>

<details>
<summary>2026-07-26 - de45f30 - Record distribution-loss rescue failure</summary>

Purpose:

- Enforce the R1 development stop rule after the first preregistered seed.

Recorded result:

- Seed `20260801` passed fixation (`0.966-0.968`) but failed every memory gate.
- Pooled clean error was `42.14` degrees; load2 clean was `49.14-54.29`
  degrees; distractor cells reached `46.49-70.28` degrees.
- Resultant lengths of `0.37-0.61` show that probability outputs were not
  uniform, so the loss rescue removed the raw-output loophole without restoring
  correct joint retrieval.

Recorded outputs:

- `outputs/multicondition_working_memory_distribution_loss/seed_sweep/seed_20260801/`
- `docs/preregistration/family_b_distribution_loss_rescue_audit.md`

Interpretation:

- R1 failed. Seeds `20260802-03` were stopped, final seeds remain untouched,
  and no perturbation outcome was inspected.

</details>

<details>
<summary>2026-07-26 - 23cb77b - Implement Family B distribution-loss rescue</summary>

Purpose:

- Implement and verify the frozen R1 baseline rescue before training any
  development seed.

File changes:

- `src/wm_rnn/training_utils.py`: Added normalized circular-distribution
  cross-entropy, separately normalized fixation loss, probability-confidence
  metrics, and config-resolved legacy versus softmax decoding.
- `src/wm_rnn/tuned_task.py`: Added stable opt-in softmax normalization while
  preserving raw-activity decoding by default.
- `src/wm_rnn/train.py`: Added the `circular_distribution` tuned-loss route and
  separate component logging.
- `src/wm_rnn/evaluate.py`, `src/wm_rnn/delay_sweep.py`, and
  `src/wm_rnn/family_b_evaluation.py`: Propagated config-selected normalization
  and reported cross-entropy plus circular resultant length.
- `configs/multicondition_working_memory_distribution_loss.yaml`: Added the
  exact preregistered R1 task, curriculum, objective, and output path.
- `tests/`: Added gradient, aligned-versus-uniform, rotation, loss-separation,
  softmax decoding, confidence-metric, and short training-smoke tests.

Verification:

- `.venv\Scripts\python.exe -m pytest tests -q` passed: 155 tests.
- `git diff --check` passed.
- No development checkpoint or perturbation outcome was generated.

</details>

<details>
<summary>2026-07-26 - 5cc18af - Pre-register Family B distribution-loss rescue</summary>

Purpose:

- Freeze a baseline-only response-loss rescue before implementing it or
  training its development seeds.

File changes:

- `docs/preregistration/family_b_distribution_loss_rescue_preregistration.md`:
  Fixed the circular-distribution objective, probability decoding, unchanged
  competence gates, three development seeds, untouched ten-seed final family,
  required tests, and stop/proceed rules.
- `docs/changelog.md`: Recorded the new baseline-rescue pre-registration.

Interpretation:

- The rescue changes no perturbation and cannot test a psilocybin mechanism.
- Any development-seed failure stops R1 before final-family training.

</details>

<details>
<summary>2026-07-26 - fa00964 - Record Family B competence failure</summary>

Purpose:

- Execute the Phase 4 baseline-only competence gate before inspecting any
  perturbation outcome.
- Exercise the frozen training contingencies without weakening any acceptance
  threshold.

File changes:

- `src/wm_rnn/train.py`: Added a deterministic competence curriculum with
  explicit trial-type counts and optional stage-specific learning rates.
- `configs/multicondition_working_memory.yaml`: Recorded the final tested
  fallback schedule: clean pretraining, balanced clean load training, then all
  four task cells at balanced load, 0.35 distractor prevalence, and a `0.0001`
  final-stage learning rate.
- `tests/test_multicondition_training.py`: Added curriculum, weighted-block,
  and learning-rate transition tests.
- `docs/preregistration/family_b_phase4_competence_audit.md`: Recorded every
  baseline-only attempt, component diagnostic, held-out gate value, and the
  decision to stop before perturbation outcomes.

Verification:

- `.venv\Scripts\python.exe -m pytest tests -q` passed: 148 tests.
- The final held-out Family B evaluation passed fixation (`0.966`) but failed
  seven of eight competence checks; only second-position `load2_clean` passed.
- No perturbation calibration, grid, signature score, or result-contingent
  analysis was run.

Interpretation:

- The component tasks are learnable in isolation, but the frozen 64-unit RNN
  did not jointly retain clean, load, distractor, and probe-dependent behavior.
- Phases 5-10 are upstream-blocked by the Phase 4 stop rule. Proceeding would
  turn baseline task failure into uninterpretable perturbation deficits.

</details>

<details>
<summary>2026-07-26 - 5fa38fc - Implement psilocybin-signature experiment</summary>

Purpose:

- Implement and verify the frozen experiment before training Family B or
  inspecting any perturbation outcome.

File changes:

- `src/wm_rnn/perturbation_metrics.py`: Added pure NumPy functions for
  amplitude-gated settling, D9 settling validity, hidden-state circular
  decoding error, distractor drift and recovery, exploratory response geometry,
  signed circular error, activation slope and saturation, and fixed-bin
  marginal state entropy.
- `tests/test_perturbation_metrics.py`: Added synthetic tests for every Phase 1
  metric, including silent-output rejection, restricted-mean conditioning
  bias, response-failure versus latency classification, fixation failure,
  non-finite-input rejection, bounded-state checks, and circular wrap/sign
  behaviour.
- `src/wm_rnn/tuned_task.py` and `src/wm_rnn/training_utils.py`: Added
  byte-regression-protected distractor and matched-timing two-slot retro-cue
  tasks with explicit per-item retention metadata.
- `src/wm_rnn/perturbation_operators.py`: Added the eight explicit perturbation
  operators, neutral invariants, the P6/P7 distinction, and the result-contingent
  Phase 9 hybrid.
- `configs/multicondition_working_memory.yaml`, `src/wm_rnn/train.py`,
  `src/wm_rnn/family_b_evaluation.py`, and `src/wm_rnn/seed_sweep.py`: Added
  balanced homogeneous Family B training and all-condition acceptance checks.
- `src/wm_rnn/perturbation_calibration.py` and
  `src/wm_rnn/perturbation_experiment.py`: Added proportional matched-cost
  calibration, deterministic D8 separation, high-precision cost/P5 gates,
  decoder reuse, exact grid schemas, metadata hashes, and smoke/full CLIs.
- `src/wm_rnn/perturbation_signature_scoring.py` and
  `src/wm_rnn/perturbation_signature_figures.py`: Added pre-outcome IUT/Holm
  scoring, D10 taxonomy, distinguishability and the frozen figure set.
- `src/wm_rnn/perturbation_assignment_sensitivity.py`: Added the
  result-contingent Phase 8 permutation and alignment utilities.
- `tests/`: Added focused task, operator, calibration, harness, scoring,
  assignment and training tests.
- `docs/changelog.md`: Recorded the verified implementation and the hash of the
  preceding pre-registration commit.

Verification:

- `.venv\Scripts\python.exe -m pytest tests -q` passed: 146 tests.
- `git diff --check` passed.

Interpretation:

- This commit constructs the pre-registered task, perturbation and outcome
  space; it contains no Family B training result or perturbation outcome.
- Restricted-mean settling is interpreted as a latency analogue only after
  fixation and response-failure validity gates pass.

</details>

<details>
<summary>2026-07-26 - 5f888be - Freeze psilocybin-signature experiment pre-registration</summary>

Purpose:

- Freeze the outcome space, matched-cost comparison, validity gates, and
  confirmatory decision rule before implementing the experiment or inspecting
  any perturbation outcome grid.

File changes:

- `docs/preregistration/psilocybin_signature_preregistration.md`: Added the
  commit-stamped Phase 0 pre-registration with an empty descriptive score
  matrix, D1-D10, six primary Family B profiles, explicit descriptive-only
  exclusions, P5-referenced C1-C3 constraints, IUT/Holm inference, power
  boundaries, and the baseline-only cost-precision pilot.
- `docs/changelog.md`: Recorded the permitted baseline-only precision pilot and
  this scoped pre-registration commit.

Interpretation:

- The pre-registration tests whether an RNN perturbation reproduces selected
  psilocybin-related behavioural signatures beyond matched generic Gaussian
  degradation. A match demonstrates computational sufficiency in these task
  analogues, not biological equivalence to psilocybin.

</details>

<details>
<summary>2026-07-26 - Cost-match precision pilot on five frozen checkpoints</summary>

Action:

- Evaluated each of the five frozen fixation-gated checkpoints on 1,024
  deterministic clean delay-20 trials using pilot batch seeds
  `202607050`-`202607065`.
- Computed one response-phase mean angular error per trial, then measured its
  checkpoint-specific mean and standard deviation.
- Used the conservative independent-means formula
  `n = 2 * (1.96 * SD / (0.10 * mean))^2` to assess whether a held-out
  proportional-cost band with half-width 0.10 is estimable precisely.

Recorded result:

- Response-mean error SDs were `6.144`, `1.632`, `1.382`, `2.015`, and
  `1.971` degrees for seeds `20260714`-`20260718`; corresponding means were
  `7.013`, `3.869`, `2.853`, `3.951`, and `3.936` degrees.
- The worst checkpoint required 590 trials, or 640 after rounding to complete
  64-trial batches, for a conservative 95% half-width of 0.10 proportional
  cost. The earlier 256-trial cost gate was under-precise for that checkpoint.

Recorded outputs:

- No persistent analysis artifact; this was a read-only pre-registration
  precision diagnostic recorded here and in the experiment specification.

Interpretation:

- The main outcome grid may remain at 256 trials per cell, but matched-cost
  validation will use at least 1,024 dedicated held-out trials per checkpoint.
- Family B will use a pre-registered baseline-only precision rule to increase
  that count if its trial-level variability exceeds the existing-checkpoint
  pilot.

</details>

<details>
<summary>2026-07-23 - Five-checkpoint gain-modulation perturbation sweeps</summary>

Action:

- Refactored `notebooks/03_gain_modulation_experiment.ipynb` into one compact
  setup and one analysis cell for each of global response gain, fixed
  heterogeneous response gain, and recurrent-drive gain.
- Loaded all five frozen `fixation_circular_working_memory` seed checkpoints
  and evaluated every condition on the same four deterministic 64-trial
  batches (256 complete trials per checkpoint and condition).
- Operationalized response gain as a scalar multiplier on the leaky recurrent
  update immediately before activation; heterogeneous gain as one fixed
  positive unit-level vector with exact population mean one; and recurrent
  gain as a multiplier on only the weight-dependent `W_h h_(t-1)` drive.
- Applied each operator at every timestep of the complete task and retained
  neutral-condition invariants against the native model forward pass.
- Added one figure per gain family combining paired checkpoint-level change in
  mean angular error with baseline and perturbed full-trial trajectories in a
  jointly fitted PCA space. PCA uses the checkpoint closest to the median
  baseline because independently trained hidden-unit bases are not aligned.

Recorded result:

- The complete notebook executed successfully on CUDA for all five
  checkpoints. Unperturbed mean angular error ranged from `2.914` to `7.047`
  degrees across model seeds.
- Global response gain produced a U-shaped cost around the trained value:
  mean paired error changes were `+19.920 ± 7.778`, `+3.071 ± 1.226`, `0`,
  `+1.596 ± 0.214`, and `+5.053 ± 0.552` degrees for gains `0.8` through
  `1.2` (mean ± SEM across five checkpoints).
- Mean-preserving heterogeneous response gain produced a monotonic cost:
  `+0.734 ± 0.056`, `+2.511 ± 0.153`, `+7.270 ± 0.430`, and
  `+12.267 ± 0.707` degrees at log-SDs `0.025`, `0.05`, `0.10`, and `0.15`.
  Three fixed gain-vector seeds were averaged within each checkpoint.
- Recurrent-drive gain also produced an asymmetric U-shaped cost:
  `+15.384 ± 1.127`, `+5.130 ± 0.741`, approximately `0`,
  `+0.477 ± 0.161`, and `+1.566 ± 0.466` degrees for gains `0.8` through
  `1.2`.
- At the moderate positive perturbations plotted above, all five checkpoints
  showed worse angular error, so the effects are replicated model-level
  sensitivities rather than a single-checkpoint accident.

Recorded outputs:

- `notebooks/03_gain_modulation_experiment.ipynb`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/metrics/global_response_gain_sweep.csv`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/metrics/heterogeneous_response_gain_sweep.csv`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/metrics/recurrent_gain_sweep.csv`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/figures/global_response_gain.png`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/figures/heterogeneous_response_gain.png`
- `outputs/fixation_circular_working_memory/perturbation_experiments/gain_modulation/figures/recurrent_gain.png`

Interpretation:

- The trained checkpoints are consistently sensitive to all three gain
  operators, and the paired sweeps establish interpretable dose-response
  curves plus visible population-trajectory reorganization.
- These are meaningful perturbation effects, but they are not yet evidence
  that any operator recovers psilocybin dynamics. The strength values are
  implementation coordinates rather than biological doses, PCA is descriptive
  for one representative checkpoint, and mechanism comparison still requires
  matched-cost calibration and held-out behavioural targets.
- Input/distractor gain remains reserved for a distractor-capable task. Minimal
  hybrid conditions should follow component calibration rather than being
  inferred from clean-task impairment alone.

</details>

<details>
<summary>2026-07-23 - Expanded psilocybin modelling dissertation draft</summary>

Action:

- Expanded the three-page literature summary into a structured dissertation
  draft with an evidence-checked Introduction, reproducible fixation-gated
  baseline Methods and Results, and an integrated independent-Gaussian control.
- Reframed the argument around the central hypothesis that perturbing a trained
  working-memory RNN may recover selective acute psilocybin behavioural
  signatures reported in humans.
- Removed project-management material, the gain evidence matrix, coding
  specification, verification checklist, and explicit next-step language from
  the dissertation prose.

Recorded result:

- Baseline and Gaussian numerical claims were traced to the current
  `fixation_circular_working_memory` artifacts.
- Gain remains a cautiously motivated candidate mechanism in the scientific
  narrative, while its research audit and implementation details remain
  outside the dissertation draft.

Recorded outputs:

- `docs/reports/psilocybin_modelling_summary.tex`
- `docs/reports/psilocybin_modelling_summary.pdf`
- `docs/reports/references.bib`

Interpretation:

- Response, input, and recurrent gain are retained as distinct candidate
  computational mechanisms. None is presented as biologically equivalent to
  psilocybin, and Gaussian RMS is retained as a non-pharmacological control
  parameter.

</details>

<details>
<summary>2026-07-23 - Current commit - Remove legacy prefix from fixation model paths</summary>

Purpose: remove the legacy personal-name prefix from repository path names while preserving the fixation-gated circular working-memory model identity.

File changes:

- `configs/fixation_circular_working_memory.yaml`: Renamed the active config and updated its output directory and run name to the neutral fixation model path.
- `docs/fixation-circular-working-memory.md`: Renamed the model note file to match the neutral path naming.
- `notebooks/01_fixation_circular_working_memory.ipynb`: Renamed the model walkthrough notebook to match the neutral path naming.

</details>

<details>
<summary>2026-07-21 - Stimulus→delay PCA Manim animation (smoother, CUDA, 256 trials)</summary>

Action:

- Rebuilt Panel C animation: 256 trials, 12× linear interpolation between RNN
  steps (~469 frames), 20 fps (~26 s), CUDA model forward + ManimGL OpenGL
  draw, medium-quality MP4 master and compact palette GIF via ffmpeg.
- Updated `notebooks/02_working_memory_task_schematic.ipynb` render cell
  (`N_TRIALS=256`, `INTERP_PER_STEP=12`, `FPS=20`).

Recorded outputs:

- `outputs/figures/schematics/yang_stimulus_to_delay_pca_trajectories.npz`
- `outputs/figures/schematics/yang_stimulus_to_delay_pca.mp4` (~0.4 MB)
- `outputs/figures/schematics/yang_stimulus_to_delay_pca.gif` (~2.7 MB)

</details>

<details>
<summary>2026-07-21 - Stimulus→delay PCA Manim animation in schematic notebook</summary>

Action:

- Added Panel C to `notebooks/02_working_memory_task_schematic.ipynb`: ManimGL
  animation of Yang hidden-state PCA from stimulus onset through the last
  delay step (48 trials; HSV = target angle; label refresh without
  ManimGL `set_text`).
- Built trajectory arrays and rendered GIF via
  `tmp/_render_stimulus_delay_pca.py` / notebook cell (`RENDER_PCA_ANIM`).

Recorded outputs:

- `outputs/figures/schematics/yang_stimulus_to_delay_pca_trajectories.npz`
- `outputs/figures/schematics/yang_stimulus_to_delay_pca.gif`
- `tmp/stimulus_to_delay_pca_scene.py`

</details>

<details>
<summary>2026-07-21 - Working-memory task schematic notebook</summary>

Action:

- Expanded `notebooks/02_working_memory_task_schematic.ipynb` with Panel B:
  fixation input/output step traces plus 32-unit stimulus/output heatmaps
  (HSV orientation × activity) for the same example trial as Panel A.
- Panel A keeps larger frames, black matched fixation crosses, Stimulus
  naming, and HSV ribbon bumps; epoch labels in simulated ms (`dt = 20`).

Recorded outputs:

- `outputs/figures/schematics/working_memory_task_schematic.png`
- `outputs/figures/schematics/working_memory_task_schematic.pdf`

</details>

<details>
<summary>2026-07-20 - Notebook PCA ring plot: Manim → matplotlib</summary>

Action:

- Replaced the ManimGL ring-outline cell in
  `notebooks/01_yang_fixation_circular_working_memory.ipynb` with a matplotlib
  scatter of late-delay/perturbed `start_pc`, colored by source angle.

Recorded outputs:

- notebook cells 16–17
- figure path: `.../figures/{RUN_NAME}_pca_ring_outline.png`

</details>

<details>
<summary>2026-07-20 - Hidden-state decode for Yang fixed-point angles</summary>

Action:

- Added `wm_rnn.hidden_angle_decoder` and switched fixation-gated fixed-point
  analyses to ridge-decode angle from hidden states instead of the silent
  circular population readout.
- Re-ran Yang `fixed_point_analysis` and `fixed_point_landscape` (512×4).

Recorded result:

- Fixed-point analysis mean angle error: ~60° → **1.24°** (p95 ~3.8°).
- Landscape known-angle error: ~56° → **6.9°** (late-delay ~2.8°, perturbed ~8.0°).
- Confirms the prior poor decode was a readout metric mismatch, not absence of
  angular structure in the hidden fixed points.

Recorded outputs:

- `src/wm_rnn/hidden_angle_decoder.py`
- `src/wm_rnn/fixed_point_analysis.py`
- `src/wm_rnn/fixed_point_landscape.py`
- `tests/test_hidden_angle_decoder.py`
- updated Yang fixed-point figures/metrics/arrays

</details>

<details>
<summary>2026-07-20 - Yang fixed-point landscape at archive-matched 512×4</summary>

Action:

- Re-ran `wm_rnn.fixed_point_landscape` for the Yang checkpoint with
  `--n-trajectory-trials 512 --perturbations-per-trial 4 --n-random-starts 0`
  (2560 task-related starts; same trial×pert sampling as the archived
  `tuned_delay_stable` landscape, without random starts).
- Regenerated the notebook Manim ring figure from `start_pc`.

Recorded result:

- 512 late-delay + 2048 perturbed starts saved.
- Converged fraction ≈ 0.557; mean known-angle error remains high (~56°),
  consistent with Yang fixed-point decode differing from the archived model.

Recorded outputs:

- `outputs/.../arrays/yang_fixation_circular_working_memory_fixed_point_landscape.npz`
- `outputs/.../figures/yang_fixation_circular_working_memory_fixed_point_landscape.png`
- `outputs/.../figures/yang_fixation_circular_working_memory_manim_pca.png`

</details>

<details>
<summary>2026-07-20 - Ring outline via late-delay starts (1000 task-related)</summary>

Action:

- Re-ran fixed-point landscape with 200 late-delay trials × 4 perturbations and
  0 random starts (1000 task-related states).
- Updated the notebook Manim cell to plot pre-optimization `start_pc` states
  colored by source angle, instead of collapsed post-optimization fixed points.

Recorded result:

- Ring geometric coverage improved (max PCA arc gap ~4.9°; 0/36 empty bins)
  versus the earlier fixed-point plot (gaps up to ~27°).

Recorded outputs:

- `outputs/.../arrays/yang_fixation_circular_working_memory_fixed_point_landscape.npz`
- `outputs/.../figures/yang_fixation_circular_working_memory_manim_pca.png`
- `notebooks/01_yang_fixation_circular_working_memory.ipynb`

</details>

<details>
<summary>2026-07-20 - Denser Yang fixed-point landscape (~1000 starts)</summary>

Action:

- Re-ran `wm_rnn.fixed_point_landscape` for the single Yang checkpoint with
  `--n-trajectory-trials 150 --perturbations-per-trial 4 --n-random-starts 250`
  (1000 total starts).

Recorded result:

- Saved 1000 fixed-point endpoints in PCA space.
- Converged fraction ≈ 0.647; mean residual ≈ 0.0010.

Recorded outputs:

- `outputs/.../arrays/yang_fixation_circular_working_memory_fixed_point_landscape.npz`
- `outputs/.../figures/yang_fixation_circular_working_memory_fixed_point_landscape.png`
- `outputs/.../metrics/yang_fixation_circular_working_memory_fixed_point_landscape_*.json/csv`

Interpretation:

- Use this denser sample for ring-outline visualization (e.g. notebook Manim
  cell). Random starts inflate decoded-angle error; prefer low-residual points
  for the cleanest ring.

</details>

<details>
<summary>2026-07-20 - Fresh notebook Manim PCA cell</summary>

Action:

- Replaced the packaged/old Manim helpers with a self-contained notebook cell that
  writes a tiny ManimGL scene for this Yang run's hidden-state PCA arrays and
  renders a static PNG (`-w -s`, no MP4).

Recorded outputs:

- `notebooks/01_yang_fixation_circular_working_memory.ipynb`
- `outputs/.../figures/yang_fixation_circular_working_memory_manim_pca.png`

</details>

<details>
<summary>2026-07-20 - Added Yang model walkthrough notebook</summary>

Action:

- Added a model-specific notebook for the canonical Yang-style
  fixation-gated circular working-memory RNN.
- Kept the notebook as a thin orchestration layer that imports existing
  package functions, reads cached metrics and figures, and leaves regeneration
  behind explicit opt-in flags.
- Added an opt-in static PCA-space hidden-state figure cell that renders from
  saved PCA arrays using ManimGL mobjects rather than Matplotlib.

Recorded outputs:

- `notebooks/01_yang_fixation_circular_working_memory.ipynb`

Interpretation:

- The notebook supports dissertation-facing inspection of the latest model
  without moving implementation logic out of the tested Python package.

</details>

<details>
<summary>2026-07-15 - Gaussian-control report and figure refinement</summary>

Action:

- Regenerated the Gaussian-control figure suite with shared Fig. 1 y-axis limits,
  translucent Gaussian geometry overlay and simplified PC labels, and a visible
  horizontal clean baseline in Fig. 4.
- Clarified the report's decoder interpretation and removed the forced page break
  before the bibliography.

Recorded outputs:

- `outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure/gaussian_control/figures/`
- `docs/reports/gaussian_control_results.pdf`

Interpretation:

- The Gaussian perturbation remains a nonspecific degradation control; the edits
  improve visual comparability and make the clean reference explicit.

</details>

<details>
<summary>2026-07-13 - Hidden-state decoder, five-seed baseline, and canonical model names</summary>

Action:

- Renamed the active progression to `categorical_working_memory`,
  `circular_working_memory`, and `yang_fixation_circular_working_memory`.
- Added cross-temporal ridge decoding from hidden states to circular sine/cosine
  targets using independent training and test trials.
- Generalized `seed_sweep` to tuned circular models and trained five independent
  4,000-step Yang-style baselines with response evaluation, hidden-state
  decoding, and delay sweeps at `10`, `20`, `40`, `80`, and `160` steps.

Recorded result:

- Mean response angular error was `4.36 ± 1.55` degrees across seeds.
- Mean fixation accuracy was `0.9677 ± 0.0016`.
- Mean delay hidden-state decoding error was `0.55 ± 0.46` degrees.
- At an untrained 160-step delay, seed-level response error ranged from `5.14`
  to `11.73` degrees.

Recorded outputs:

- `outputs/yang_fixation_circular_working_memory/metrics/yang_fixation_circular_working_memory_cross_temporal_decoder_summary.json`
- `outputs/yang_fixation_circular_working_memory/figures/yang_fixation_circular_working_memory_cross_temporal_decoder.png`
- `outputs/yang_fixation_circular_working_memory/metrics/yang_fixation_circular_working_memory_seed_sweep_summary.json`
- `outputs/yang_fixation_circular_working_memory/seed_sweep/`

Interpretation:

- Circular memory content is accurately decodable during the silent-output delay
  in every independently trained network. The model now has both prerequisites
  for perturbation work: a maintenance-period memory measure and an independent
  baseline ensemble.
- Between-seed variation must remain part of later perturbation comparisons;
  perturbation conclusions should not be based on one selected checkpoint.

</details>

<details>
<summary>2026-07-13 - Archived superseded model runs</summary>

The active dissertation progression is now `baseline_delay`, `tuned_delay`,
and `tuned_delay_fixation_gate_stable`. The five superseded output directories
were moved beneath `outputs/archive/` without deleting checkpoints, metrics,
arrays, or figures. Retained superseded YAMLs were moved beneath
`configs/archive/`; `docs/model-run-archive.md` records provenance. These runs
remain development history rather than active model alternatives.

</details>

1. Git commits: code and documentation changes committed to the repository.
2. Run log: experiment actions performed with the baseline model, including
   training, delay sweeps, multi-seed runs, and plotting.

<details>
<summary>2026-07-13 - Stabilized Yang-style randomized-timing baseline</summary>

Action:

- Added `configs/tuned_delay_fixation_gate_stable.yaml` and support for a
  pre-cue fixation phase plus randomized pre-cue, cue-duration, and discrete
  delay choices.
- Followed the main Yang delayed-response schedule with pre-cue choices `15`,
  `25`, `35`; cue choices `10`, `20`, `30`; delay choices `10`, `20`, `40`,
  `80`; a `25`-step response; and an unscored `5`-step response transition.
- Added normalized weighted tuned MSE (response weight `5`, fixation weight
  `2`), input noise `0.01`, training-only recurrent noise `0.05`, and gradient
  clipping at `1.0`. Weighting is linear rather than reproducing the accidental
  squared-mask effect of the reference TensorFlow implementation.
- Trained for 4,000 steps on CPU and regenerated evaluation, delay-sweep, PCA,
  stability, attractor, fixed-point, landscape, and dynamics artifacts.
- Corrected stability phase boundaries to use generated batch phases, including
  the new pre-cue fixation period.

Recorded result:

- Reference-timing evaluation: mean angular error `3.406` degrees, population
  MSE `0.00806`, fixation MSE `0.01915`, fixation accuracy `0.9667` including
  the deliberately unscored response transition.
- Frozen-delay mean angular error was `3.295`, `3.399`, `3.432`, and `4.174`
  degrees at trained delays `10`, `20`, `40`, and `80`; it remained `5.682`
  degrees at the untrained `160`-step delay.
- Corrected delay settling ratio was `0.366`. The autonomous probe's final
  speed was `0.00253` and mean drift was `16.958` degrees.
- Fixed-point analyses found a near-neutral leading eigenvalue (mean spectral
  radius `0.99974`), contracting secondary direction (`0.88742`), and strong
  leading-eigenvector/tangent alignment (`0.9783`). Landscape convergence was
  `0.766` at residual threshold `0.001`.

Recorded outputs:

- `outputs/tuned_delay_fixation_gate_stable/checkpoints/tuned_delay_fixation_gate_stable.pt`
- `outputs/tuned_delay_fixation_gate_stable/metrics/`
- `outputs/tuned_delay_fixation_gate_stable/arrays/`
- `outputs/tuned_delay_fixation_gate_stable/figures/`

Interpretation:

- Randomized Yang-style timing largely removed the fixed-delay model's timing
  overfitting and produced much stronger settling and local-stability evidence.
- Circular readout angle during fixation/delay is not a valid memory decoder
  because the architecture explicitly trains that readout to remain silent.
  Accordingly, fixed-point decoded-angle error and delay-period output-angle
  traces should not be used to judge memory preservation; hidden-state or
  cross-temporal decoding is required for that claim.

</details>

<details>
<summary>2026-07-13 - Yang-style fixation-gated circular baseline</summary>

Action:

- Replaced the active response-gated prototype configuration with
  `configs/tuned_delay_fixation_gate.yaml`.
- Renamed the opt-in task flag and auxiliary metrics to fixation terminology.
  The fixation input/output are high during cue and delay and low during
  response, following the Yang-style hold-then-report convention.
- Retained the 32-unit circular population readout, silent before response, and
  whole-trial supervision.
- Trained for 2,000 steps on CPU, evaluated 20 batches, and regenerated the
  delay-sweep, PCA, stability, attractor, fixed-point, landscape, and dynamics
  figures for the new checkpoint.

Recorded result:

- Final whole-trial training loss was `0.00168`.
- Mean response angular error was `1.927` degrees, response population MSE was
  `0.00511`, fixation MSE was `0.00946`, and fixation accuracy was `1.000`.
- Longer-delay performance degraded from `1.923` degrees at 20 steps to
  `43.951` degrees at 80 steps.
- The delay settling ratio was `2.356`; the autonomous probe drifted `58.431`
  degrees; no trajectory-seeded fixed points met the `0.001` residual threshold.
  The fixed-delay model therefore performs well at its trained response time but
  does not reproduce the stable tuned model's attractor evidence.
- Delay-period angle traces from the circular readout are not memory-decoding
  measures for this architecture because that readout is explicitly trained to
  remain silent before response.

Recorded outputs:

- `outputs/tuned_delay_fixation_gate/checkpoints/tuned_delay_fixation_gate.pt`
- `outputs/tuned_delay_fixation_gate/metrics/`
- `outputs/tuned_delay_fixation_gate/arrays/`
- `outputs/tuned_delay_fixation_gate/figures/`

Interpretation:

- The Yang-style interface is implemented correctly and enforces fixation then
  report. Variable-delay training or another stability objective is still
  needed before treating this selected interface as the stable attractor
  baseline for perturbation experiments.

</details>

<details>
<summary>2026-07-09 - Response-gated circular delayed-response baseline</summary>

Action:

- Added `configs/tuned_delay_response_gate.yaml` and an opt-in
  `task.response_gated` path for the tuned circular task.
- The final input is a binary response cue, the final output is a binary
  response gate, and the circular target remains zero until the response
  period. `training.score_all_periods` scores the whole trial to penalize an
  early circular report.
- Added response-gate evaluation metrics and focused tests.

Recorded result:

- The 2,000-step fixed-delay run reached `0.0012` final whole-trial MSE.
- Over 20 evaluation batches, mean response angular error was `2.212` degrees,
  response population MSE was `0.00454`, response-gate MSE was `0.00210`, and
  response-gate accuracy was `1.000`.
- CUDA was requested but unavailable in the current Python/PyTorch environment,
  so the run used CPU.

Recorded outputs:

- `outputs/tuned_delay_response_gate/checkpoints/tuned_delay_response_gate.pt`
- `outputs/tuned_delay_response_gate/metrics/tuned_delay_response_gate_train_metrics.json`
- `outputs/tuned_delay_response_gate/metrics/tuned_delay_response_gate_eval_metrics.json`

Interpretation:

- This established the first hold-then-report prototype. It was superseded on
  2026-07-13 by the selected Yang-style fixation-gated convention; its recorded
  checkpoint and metrics remain historical comparison artifacts.

</details>

## Note On Archived Baseline Artifacts (2026-07-01)

As of 2026-07-01, the baseline hidden-state nonlinearity changed from `relu`
to `tanh` (see the "Promoted `tanh` to the canonical baseline activation"
entry below for the full reasoning and results). Every run-log entry below
this note that references `outputs/baseline_delay/...` paths and was
recorded **before** this change describes the original `relu`-based
network. Those historical output artifacts may still exist locally under
`outputs/baseline_delay_relu/` (same internal file names, unchanged), but the
old relu config is no longer part of the active tracked config set. The tracked
`configs/baseline_delay.yaml` file now refers to the current `tanh`-based
baseline. Run-log entries recorded after this note that reference
`outputs/baseline_delay/...` describe the `tanh`-based artifacts.

## Current Baseline Cleanup (2026-07-07)

The active categorical delay baseline is now only `configs/baseline_delay.yaml`
with `model.activation: tanh`. The older `baseline_delay_relu` and
`baseline_delay_stable` configs were removed to avoid presenting relu-based
categorical variants as current baseline models. Historical changelog entries
may still describe those earlier experiments, but they are no longer active
model variants.

## Git Commit History

<details>
<summary>2026-07-13 - bb43031 - Support circular models in seed sweeps</summary>

Generalized independent-seed training, evaluation, decoder analysis, delay
sweeps, and aggregate plotting to continuous circular working-memory models.

File changes:

- `src/wm_rnn/seed_sweep.py`: Added circular metrics and per-seed decoding.
- `src/wm_rnn/plot_seed_sweeps.py`: Added angular-error aggregate plots.
- `tests/test_plot_seed_sweeps.py`: Added tuned-model plotting coverage.

</details>

<details>
<summary>2026-07-13 - 2d0f8c5 - Add cross-temporal hidden-state decoder</summary>

Added held-out circular decoding across training and testing time points so
memory content can be measured while the Yang-style output remains silent.

File changes:

- `src/wm_rnn/cross_temporal_decoder.py`: Added decoder analysis and artifacts.
- `tests/test_cross_temporal_decoder.py`: Added recovery and validation tests.

</details>

<details>
<summary>2026-07-13 - c93feb4 - Rename active working-memory models clearly</summary>

Replaced developmental run names with task- and representation-based names for
the three active dissertation models.

File changes:

- `configs/`: Renamed all active configurations and output/run paths.
- `src/wm_rnn/`: Updated command defaults and fallback run names.
- `tests/`: Updated active configuration references.

</details>

<details>
<summary>2026-07-13 - e7a826b - Archive superseded model variants</summary>

Consolidated the repository around the three active dissertation progression
stages while preserving superseded configurations and generated run artifacts.

File changes:

- `README.md`, `docs/model-architecture.md`: Documented the active progression.
- `configs/archive/`: Retained superseded tuned configurations and provenance.
- `docs/model-run-archive.md`: Added the archived-run manifest.

</details>

<details>
<summary>2026-07-13 - 9e92165 - Update stabilized Yang model summary</summary>

Updated the durable Markdown, LaTeX, and PDF summary for the stabilized
Yang-style model and its corrected analysis results.

File changes:

- `docs/tuned-delay-stable-model-summary.md`: Added the model summary.
- `docs/reports/stable_model_summary.tex`: Updated the report source.
- `docs/reports/stable_model_summary.pdf`: Regenerated the report artifact.

</details>

<details>
<summary>2026-07-13 - 9c853e8 - Align analyses with variable Yang timing</summary>

Made stability and dynamics analyses respect generated phase boundaries for
randomized pre-cue, cue, delay, and response timing.

File changes:

- `src/wm_rnn/dynamics_figures.py`: Used generated phase metadata in figures.
- `src/wm_rnn/stability_analysis.py`: Corrected phase-boundary handling.

</details>

<details>
<summary>2026-07-13 - 92a4517 - Implement Yang-style fixation-gated training</summary>

Implemented fixation-gated circular task dimensions, randomized timing,
weighted loss, training noise, gradient clipping, evaluation, and tests.

File changes:

- `src/wm_rnn/`: Added Yang-style task and training support.
- `configs/`: Added the stabilized active config and retained fixed-delay config.
- `tests/`: Added coverage for fixation gating, timing, and weighted loss.

</details>

<details>
<summary>2026-07-07 - 408e824 - Remove obsolete relu delay variants</summary>

Cleaned the active model set so the only current categorical baseline is
`configs/baseline_delay.yaml` with `model.activation: tanh`.

File changes:

- `configs/baseline_delay_relu.yaml`: Removed from the active config set.
- `configs/baseline_delay_stable.yaml`: Removed from the active config set.
- `README.md`: Removed active relu/stable categorical baseline references.
- `docs/model-architecture.md`: Reframed the categorical baseline as
  `baseline_delay` only.
- `docs/changelog.md`: Added the current baseline cleanup note.
- `docs/reports/stable_model_summary.tex`: Updated the report to compare
  `baseline_delay` against `tuned_delay_stable`.
- `docs/reports/stable_model_summary.pdf`: Recompiled the updated report.

</details>

<details>
<summary>2026-07-09 - Repository memory and maintenance protocol added</summary>

Action:

- Added the vault-level `AGENTS.md` instructions for session context loading,
  supervisor-feedback capture, durable wiki updates, changelog maintenance, and
  repository hygiene.

Interpretation:

- The dissertation vault is the durable source of project memory between Codex
  sessions. This governance change does not modify model code, configurations,
  experiments, or generated outputs.

</details>

<details>
<summary>2026-07-07 - 7c0197a - Add stable model summary report</summary>

Added a LaTeX/PDF summary of the current categorical baseline and tuned stable
model before moving on to noise analysis.

File changes:

- `docs/reports/stable_model_summary.tex`: Added a LaTeX report comparing
  `baseline_delay` and `tuned_delay_stable`, including task descriptions,
  key metrics, attractor-analysis interpretation, and figure captions.
- `docs/reports/stable_model_summary.pdf`: Compiled report with the relevant
  saved figures embedded from the baseline and tuned output directories.

</details>

<details>
<summary>2026-07-07 - 6e948d3 - Add tuned delay attractor model analysis</summary>

Added the tuned continuous population-code model, the stable tuned variant, and
the attractor-analysis tooling used for fixed-point, Jacobian, perturbation, and
movie visualizations.

File changes:

- `configs/tuned_delay.yaml`: Added the fixed-delay continuous circular
  population-code model.
- `configs/tuned_delay_stable.yaml`: Added the stable tuned model trained with
  randomized delay lengths and delay-period scoring.
- `src/wm_rnn/tuned_task.py`: Added the tuned circular task generator.
- `src/wm_rnn/*`: Updated training, evaluation, PCA analysis, delay sweeps,
  stability analysis, attractor probing, fixed-point analysis, dynamics figures,
  and hidden-state movie generation for tuned tasks.
- `tests/`: Added focused tuned-task, tuned-training, evaluation, and PCA tests.
- `docs/model-architecture.md`: Added the tuned model architecture and analysis
  explanation.
- `docs/analysis-figure-rationale.md`: Added figure rationale and literature
  anchors.
- `custom_config.yml` and `requirements-animation.txt`: Added optional Manim
  animation support.

</details>

<details>
<summary>2026-07-07 - 061fde5 - Updated README</summary>

Documentation-only README cleanup after the baseline activation updates.

File changes:

- `README.md`: Updated current model and workflow notes.

</details>

<details>
<summary>2026-07-01 - 9c63925 - Update tanh defaults and baseline documentation</summary>

Aligned defaults and documentation with the `tanh` baseline direction that was
being tested after the original relu baseline showed ramping delay dynamics.

File changes:

- `README.md`: Updated model-variant documentation.
- `configs/baseline_delay_stable.yaml`: Preserved the then-current stable
  comparison configuration state.
- `src/wm_rnn/config.py`: Updated default model activation handling.
- `src/wm_rnn/model.py`: Updated the `RNNConfig` activation default.
- `src/wm_rnn/training_utils.py`: Updated config-to-model activation fallback.

</details>

<details>
<summary>2026-06-29 - ed92033 - feat: add baseline working-memory rnn</summary>

Initial repository build for the baseline delayed-response working-memory RNN.

File changes:

- `.gitignore`: Added Python, environment, checkpoint, output, and temporary-file ignores.
- `README.md`: Added setup, training, evaluation, PCA analysis, and reproducible-run instructions.
- `configs/baseline_delay.yaml`: Added the default baseline task, model, training, evaluation, analysis, and output-path configuration.
- `docs/build-log.md`: Added the original build record and implementation decisions. This file was later removed from Git tracking.
- `pyproject.toml`: Added package metadata, dependencies, setuptools configuration, and initial test configuration.
- `requirements-cuda.txt`: Added the CUDA-oriented dependency path.
- `requirements.txt`: Added the CPU/default dependency path.
- `src/wm_rnn/__init__.py`: Created the package namespace.
- `src/wm_rnn/analysis.py`: Added hidden-state PCA trajectory analysis for trained checkpoints.
- `src/wm_rnn/config.py`: Added default config and YAML config loading.
- `src/wm_rnn/device.py`: Added automatic CPU/CUDA device selection.
- `src/wm_rnn/evaluate.py`: Added checkpoint evaluation and confusion-matrix output.
- `src/wm_rnn/io.py`: Added output-directory and metrics-writing helpers.
- `src/wm_rnn/model.py`: Added the continuous-time ReLU RNN and linear readout.
- `src/wm_rnn/task.py`: Added the categorical cue-delay-response task generator.
- `src/wm_rnn/train.py`: Added the baseline training entry point.
- `src/wm_rnn/training_utils.py`: Added shared tensor conversion, model creation, masked loss, accuracy, and confusion-matrix utilities.
- `tests/*`: Added the original pytest coverage for config/device, model, task, and training pipeline behavior. These were later removed from Git.

</details>

<details>
<summary>2026-06-29 - df8dd04 - docs: add pydoc docstrings</summary>

Documentation-only cleanup across the Python package.

File changes:

- `src/wm_rnn/analysis.py`: Added docstrings for PCA analysis structures and functions.
- `src/wm_rnn/config.py`: Added docstrings for configuration loading and merging.
- `src/wm_rnn/device.py`: Added docstrings for device-selection helpers.
- `src/wm_rnn/evaluate.py`: Added docstrings for evaluation result structures and functions.
- `src/wm_rnn/io.py`: Added docstrings for output and metrics file helpers.
- `src/wm_rnn/model.py`: Added docstrings for RNN config, recurrent layer, and readout model.
- `src/wm_rnn/task.py`: Added docstrings for task config, generated batches, and task generation.
- `src/wm_rnn/train.py`: Added docstrings for training result structures and CLI flow.
- `src/wm_rnn/training_utils.py`: Added docstrings for shared training and evaluation utilities.

</details>

<details>
<summary>2026-07-01 - 71f9fd8 - Removed pytests from git, Added seed sweep for baseline model with plots to visualise memory degradation</summary>

Added the main delay-generalization and multi-seed analysis tooling, while removing the tracked pytest suite.

File changes:

- `README.md`: Added usage instructions for delay sweeps, multi-seed training, and combined seed-sweep plotting.
- `src/wm_rnn/delay_sweep.py`: Added frozen-checkpoint evaluation across multiple delay lengths, writing JSON, CSV, and a delay-accuracy plot.
- `src/wm_rnn/plot_seed_sweeps.py`: Added combined plotting for multiple seed-specific delay sweeps.
- `src/wm_rnn/seed_sweep.py`: Added multi-seed training, evaluation, optional delay sweeps, and seed-level summary outputs.
- `tests/test_config_and_device.py`: Removed from Git.
- `tests/test_model.py`: Removed from Git.
- `tests/test_task.py`: Removed from Git.
- `tests/test_training_pipeline.py`: Removed from Git.

</details>

<details>
<summary>2026-07-01 - dc54e4c - Revise README for</summary>

README simplification pass.

File changes:

- `README.md`: Removed a large amount of earlier README material and kept the documentation more focused on the project workflow.

</details>

<details>
<summary>2026-07-01 - bf7ccc6 - Update README for CUDA training instructions</summary>

Small CUDA setup documentation update.

File changes:

- `README.md`: Revised the CUDA training instructions.

</details>

<details>
<summary>2026-07-01 - 32bdec9 - removed build log</summary>

Removed the build log from Git history going forward.

File changes:

- `docs/build-log.md`: Deleted from Git tracking.

</details>

<details>
<summary>2026-07-01 - ce2dffa - Merge branch 'main' of https://github.com/BobRice1/working-memory-rnn</summary>

Merge commit.

File changes:

- No file-level changes were listed by `git log --name-status` for this merge commit.

</details>

<details>
<summary>2026-07-01 - f673574 - updated gitignore</summary>

Ignore-rule update.

File changes:

- `.gitignore`: Updated ignore patterns.

</details>

<details>
<summary>2026-07-01 - 442e55a - Add hidden-state stability analysis module and changelog documentation</summary>

Added the first direct diagnostic for hidden-state settling versus ramping.

File changes:

- `docs/changelog.md`: Added the project changelog and run log.
- `src/wm_rnn/stability_analysis.py`: Added hidden-state norm and step-to-step speed analysis, phase summaries, a delay settling ratio, JSON/array outputs, and a two-panel stability figure.

</details>

<details>
<summary>2026-07-01 - 57c24a6 - Enhance training configurations with delay scoring and activation variants</summary>

Added training and architecture variants used to test whether the original baseline's long-delay degradation came from the training objective or from the unbounded `relu` activation.

File changes:

- `README.md`: Added stability-analysis instructions and model-variant descriptions.
- `configs/baseline_delay_stable.yaml`: Added the randomized-delay, whole-delay-loss variant.
- `configs/baseline_delay_tanh.yaml`: Added the isolated `tanh` activation variant used for comparison before `tanh` became the canonical baseline.
- `docs/changelog.md`: Added reasoning and run-log entries for stability analysis, the stable training variant, and the isolated `tanh` variant.
- `src/wm_rnn/config.py`: Added the activation field to the default model configuration.
- `src/wm_rnn/model.py`: Made the hidden-state activation configurable (`relu` or `tanh`) instead of hardcoding `relu`.
- `src/wm_rnn/train.py`: Added optional randomized delay lengths and optional whole-delay loss scoring.
- `src/wm_rnn/training_utils.py`: Passed the activation into `RNNConfig` and added a helper for changing delay length.

</details>

<details>
<summary>2026-07-01 - dc918f5 - Promote tanh as the canonical baseline activation</summary>

Archived the original `relu` baseline and promoted the better-settling `tanh` model to the default baseline config.

File changes:

- `configs/baseline_delay.yaml`: Updated the canonical baseline to use `model.activation: tanh`.
- `configs/baseline_delay_relu.yaml`: Added by renaming the previous `baseline_delay_tanh.yaml` path and editing it into an archival config for the original `relu` baseline.
- `docs/changelog.md`: Added the `tanh` promotion rationale and recorded the new canonical baseline reruns.
- `pyproject.toml`: Updated project metadata/configuration state after removing the old test setup.

</details>

<details>
<summary>2026-07-07 - 6e948d3 detail - Tuned continuous population-code model iteration</summary>

Purpose:

- Added the next baseline model iteration requested after supervisor feedback:
  a continuous circular-location delayed-response task using Gaussian /
  von-Mises-like population tuning curves.
- This does not replace the existing categorical baseline. The categorical
  model remains available through `configs/baseline_delay.yaml`; the tuned
  continuous model is configured separately through `configs/tuned_delay.yaml`.
- This is still a baseline working-memory model iteration, not a
  psilocybin-informed perturbation.

Model/task changes:

- `src/wm_rnn/tuned_task.py`: Added the tuned circular task generator.
  Each trial samples an angle in `[0, 2*pi)`, encodes it over evenly spaced
  preferred-angle units using:

  ```text
  activity_i = exp(kappa * (cos(theta - preferred_i) - 1))
  ```

  The input contains the population bump during the cue period only, plus the
  existing fixation/context channel. The target is the remembered population
  bump, scored during the response period.
- `configs/tuned_delay.yaml`: Added the new tuned-delay config with
  `n_tuned_units: 32`, `tuning_kappa: 8.0`, `tanh` recurrent activation, and
  outputs under `outputs/tuned_delay/`.
- `src/wm_rnn/config.py`: Added `task.task_type: categorical` to the default
  config so task dispatch is explicit.
- `src/wm_rnn/training_utils.py`: Added typed dispatch for categorical vs
  tuned task configs, `generate_batch_for_task()`, masked population MSE,
  circular decoding metrics, and tensor conversion that preserves categorical
  integer targets while allowing tuned float population targets.
- `src/wm_rnn/train.py`: Updated training so categorical runs still use masked
  cross-entropy and response accuracy, while tuned runs use masked population
  MSE and report `population_mse` / `final_population_mse` instead of a fake
  categorical accuracy value.
- `src/wm_rnn/evaluate.py`: Updated evaluation so categorical runs still report
  accuracy and write a confusion matrix, while tuned runs report mean angular
  error, median angular error, and population MSE. Tuned evaluation does not
  write a confusion matrix because the output is continuous.
- `src/wm_rnn/analysis.py`: Updated PCA analysis to generate either task type,
  save generic `labels` plus `task_type`, preserve categorical `cues` for
  compatibility, and color tuned trajectories by continuous angle.

Documentation and tests:

- `README.md`: Added `configs/tuned_delay.yaml` as the next model variant and
  briefly explained the continuous population-code task.
- `docs/model-architecture.md`: Added a continuous tuned task section and
  updated later architecture, training, evaluation, and data-flow sections to
  distinguish categorical and tuned branches.
- `tests/`: Added focused tests for tuned population encoding, circular
  wraparound, generated batch semantics, angle decoding, masked MSE, task
  dispatch, categorical regression behavior, tuned evaluation aggregation,
  CLI metric printing, tuned training metrics, and PCA label saving.

Verification:

- Full test suite:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q
  ```

  Result: `22 passed`.
- Tuned smoke verification was run with a short 2-step training configuration:
  training produced `population_mse`, evaluation produced finite angular-error
  and population-MSE metrics, `confusion_path` was `None`, and PCA figure/array
  outputs were written.
- Categorical smoke verification was also run with a short 2-step training
  configuration: categorical history still contained response accuracy,
  evaluation still produced an accuracy metric, and the confusion matrix file
  was still written.
- Temporary smoke output directories were deleted afterward so `outputs/` only
  retains the non-smoke baseline result folders.

How to run the tuned model:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.train --config configs/tuned_delay.yaml --device cpu
.\.venv\Scripts\python.exe -m wm_rnn.evaluate --config configs/tuned_delay.yaml --checkpoint outputs/tuned_delay/checkpoints/tuned_delay.pt --device cpu
.\.venv\Scripts\python.exe -m wm_rnn.analysis --config configs/tuned_delay.yaml --checkpoint outputs/tuned_delay/checkpoints/tuned_delay.pt --device cpu
```

Next action under consideration:

- Train the full `configs/tuned_delay.yaml` model, then inspect angular error,
  population MSE, and PCA trajectories before adding distractors or any
  psilocybin-informed perturbation.
- Decide whether tuned continuous analysis should also get delay-length sweeps,
  drift metrics, or stability analysis adapted from the categorical baseline.

</details>

<details>
<summary>2026-07-07 - 6e948d3 detail - Stable tuned continuous model and attractor-like probe</summary>

Purpose:

- Tested whether the continuous tuned model could maintain a circular
  population-code memory over longer delays and during autonomous hidden-state
  evolution.
- Added a more stable tuned training configuration after the first fixed-delay
  tuned model learned the trained delay but drifted strongly when pushed beyond
  it.

Implementation changes:

- `configs/tuned_delay_stable.yaml`: Added a stable tuned config using
  randomized `20`-`80` step delays, `score_delay_period: true`, `2000` training
  steps, `tanh` recurrent activation, and outputs under
  `outputs/tuned_delay_stable/`.
- `src/wm_rnn/delay_sweep.py`: Extended delay sweeps to tuned tasks. Categorical
  runs still report accuracy; tuned runs report mean, median, p95, and maximum
  angular error plus population MSE.
- `src/wm_rnn/stability_analysis.py`: Updated hidden-state stability analysis
  to generate either categorical or tuned task batches.
- `src/wm_rnn/attractor_probe.py`: Added an autonomous tuned hidden-state probe.
  It starts from late-delay hidden states, removes the cue, runs the recurrent
  dynamics forward, decodes the remembered angle over probe time, and reports
  hidden-state speed, hidden-state displacement, angular drift, and final error.
- `src/wm_rnn/fixed_point_analysis.py`: Added sampled fixed-point and Jacobian
  analysis for tuned checkpoints. It starts from late-delay hidden states,
  optimizes nearby hidden states under blank-delay input to minimize one-step
  recurrent speed, decodes the resulting fixed-point angle, computes the local
  recurrent Jacobian, and summarizes the eigenspectrum.
- `src/wm_rnn/fixed_point_landscape.py`: Added the reference-notebook-style
  landscape view. It fits PCA on task-evoked hidden trajectories, searches for
  fixed points from late-delay, perturbed late-delay, and random bounded hidden
  starts, and plots all fixed-point endpoints in the same PCA state space.
- `src/wm_rnn/dynamics_figures.py`: Added four non-noise mechanism figures:
  fixed-point ring in PCA space, decoded angle over time, Jacobian spectrum,
  and deterministic perturbation recovery.
- `src/wm_rnn/hidden_state_movie.py`: Added an animated tuned-delay hidden-state
  movie that overlays sampled PCA trajectories, the fixed-point ring, cue/output
  population activity, task phase, decoded angle, target angle, and angular
  error. The movie also includes a separate PCA panel for perturbed late-delay
  states returning toward the ring under blank recurrent dynamics, with options
  for dense wide perturbation clouds and interpolated video frames between real
  recurrent model steps. It writes GIF/MP4 movies plus a Manim-ready `.npz` data
  archive.
- `custom_config.yml`: Added a local ManimGL configuration so 3b1b Manim cache
  and rendered assets stay under `outputs/manim/` when Manim is run from this
  repository.
- `requirements-animation.txt`: Added optional animation dependencies for the
  ManimGL workflow without making them mandatory for normal model runs.
- Tuned-delay-stable figure readability pass: added or clarified legends,
  marker labels, phase labels, and degree/residual colorbars so the PCA,
  delay-sweep, stability, attractor-probe, fixed-point, landscape, and dynamics
  figures are easier to interpret without cross-referencing the code.
- Interpretation note: the ring can appear flipped between the static
  ring/landscape figures and the hidden-state movie because each visualization
  currently fits its own two-dimensional PCA projection. PCA component signs are
  arbitrary, and the movie also uses its own sampled trajectory batch. This is a
  visualization-orientation issue, not evidence that the attractor itself
  changed. A future cleanup should save and reuse one shared PCA basis across the
  ring manifold, fixed-point landscape, and movie outputs for consistent figure
  orientation.
- `docs/analysis-figure-rationale.md`: Added the rationale and reference
  anchors for these figures, linking each plot to the working-memory RNN /
  attractor literature it supports.

Stable tuned run:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.train --config configs/tuned_delay_stable.yaml
.\.venv\Scripts\python.exe -m wm_rnn.evaluate --config configs/tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt
.\.venv\Scripts\python.exe -m wm_rnn.delay_sweep --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --delays 20 30 40 60 80 100 120
.\.venv\Scripts\python.exe -m wm_rnn.stability_analysis --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trials 128
.\.venv\Scripts\python.exe -m wm_rnn.attractor_probe --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trials 128 --probe-steps 100
.\.venv\Scripts\python.exe -m wm_rnn.fixed_point_analysis --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trials 64 --max-steps 5000 --lbfgs-steps 200 --learning-rate 0.03 --anchor-weight 0.00001 --residual-threshold 0.001 --device cuda
.\.venv\Scripts\python.exe -m wm_rnn.fixed_point_landscape --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trajectory-trials 64 --n-random-starts 128 --perturbations-per-trial 2 --perturbation-scale 0.15 --max-steps 3000 --lbfgs-steps 100 --learning-rate 0.03 --residual-threshold 0.001 --device cuda
.\.venv\Scripts\python.exe -m wm_rnn.dynamics_figures --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trials 64 --example-trials 12 --recovery-steps 100 --perturbation-scales 0 0.05 0.1 0.2 0.4 0.8 --device cuda
.\.venv\Scripts\python.exe -m wm_rnn.hidden_state_movie --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt --n-trials 512 --example-trials 24 --delay-steps 90 --recovery-source wide-perturbed --recovery-perturbations-per-trial 4 --perturbation-scale 1.0 --recovery-steps 100 --fps 30 --frames-per-step 5 --format mp4 --device cuda
.\.venv\Scripts\python.exe -m wm_rnn.analysis --config configs\tuned_delay_stable.yaml --checkpoint outputs\tuned_delay_stable\checkpoints\tuned_delay_stable.pt
```

Recorded result - trained-delay evaluation:

- Held-out mean angular error: `0.644` degrees.
- Population MSE: approximately `0.000`.

Recorded result - delay sweep:

| Delay steps | Mean angular error | P95 angular error | Population MSE |
| ---: | ---: | ---: | ---: |
| 20 | 0.643 | 1.375 | 0.000069 |
| 30 | 0.791 | 1.662 | 0.000090 |
| 40 | 0.956 | 1.988 | 0.000119 |
| 60 | 1.182 | 2.515 | 0.000178 |
| 80 | 1.453 | 2.958 | 0.000267 |
| 100 | 1.681 | 3.435 | 0.000401 |
| 120 | 1.903 | 3.862 | 0.000489 |

Recorded result - hidden-state stability:

- Early-delay speed: `0.065857`.
- Late-delay speed: `0.002696`.
- Delay settling ratio: `0.040939`.

Recorded result - autonomous hidden-state probe:

- Mean initial probe speed: `0.002005`.
- Mean final probe speed: `0.001455`.
- Speed settling ratio: `0.725529`.
- Mean autonomous drift: `1.509` degrees.
- Median autonomous drift: `1.512` degrees.
- P95 autonomous drift: `2.918` degrees.
- Mean final angular error: `2.108` degrees.
- P95 final angular error: `3.978` degrees.

Recorded result - fixed-point and Jacobian analysis:

- Mean fixed-point residual: `0.000792`.
- Median fixed-point residual: `0.000728`.
- Maximum fixed-point residual: `0.002811`.
- Fraction below the `0.001` residual threshold: `0.766`.
- Mean distance from late-delay trajectory state: `0.130`.
- Mean fixed-point decoding error: `1.356` degrees.
- P95 fixed-point decoding error: `3.244` degrees.
- Mean drift from late-delay decoded angle: `1.118` degrees.
- Mean spectral radius: `1.000036`.
- Maximum spectral radius: `1.005071`.
- Mean second-largest absolute eigenvalue: `0.984019`.
- Mean number of eigenvalues with absolute value above `0.99`: `1.328`.
- Mean number of eigenvalues with absolute value above `1.0`: `0.516`.
- Mean tangent alignment between the sampled ring direction and leading
  eigenvector: `0.945`.

Recorded result - fixed-point landscape visualization:

- PCA explained variance ratio: PC1 `0.348`, PC2 `0.293`.
- Starts searched: `320` total (`64` late-delay, `128` perturbed late-delay,
  `128` random bounded hidden starts).
- Mean fixed-point residual: `0.000470`.
- Median fixed-point residual: `0.000157`.
- Fraction below the `0.001` residual threshold: `0.856`.
- Fraction below threshold by source: late-delay `0.797`, perturbed late-delay
  `0.750`, random `0.992`.
- Mean known-angle error: `3.477` degrees.
- P95 known-angle error: `9.919` degrees.
- Mean known-angle error by source: late-delay `3.188` degrees, perturbed
  late-delay `3.621` degrees.
- Visual result: task trajectories and fixed-point endpoints occupy the same
  ring-shaped structure in PCA space, including endpoints found from random
  bounded hidden-state starts.

Recorded result - additional dynamics figures:

- PCA explained variance ratio for the figure batch: PC1 `0.312`, PC2 `0.277`.
- Mean delay-period decoded angular error: `0.523` degrees.
- Mean response-period decoded angular error: `0.648` degrees.
- Jacobian mean absolute eigenvalue: `0.667`.
- Jacobian maximum absolute eigenvalue: `1.005`.
- Deterministic perturbation recovery after `100` blank-delay steps:

| Hidden perturbation SD | Mean final angular error | P95 final angular error | Mean initial distance to ring | Mean final distance to ring |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.879 deg | 3.807 deg | 0.159 | 0.159 |
| 0.05 | 1.885 deg | 3.901 deg | 0.444 | 0.167 |
| 0.10 | 2.040 deg | 4.296 deg | 0.829 | 0.159 |
| 0.20 | 2.791 deg | 6.355 deg | 1.575 | 0.169 |
| 0.40 | 7.185 deg | 13.764 deg | 2.885 | 0.230 |
| 0.80 | 24.296 deg | 116.251 deg | 4.910 | 0.689 |

Interpretation of these figures:

- The fixed-point ring figure gives a cleaner version of the ring manifold in
  task PCA space, colored by decoded fixed-point angle.
- The decoded-angle plot shows that the model quickly locks onto the target
  angle and keeps it stable through delay and response.
- The Jacobian spectrum plot shows most modes inside the unit circle, with
  near-neutral leading modes around the ring.
- The perturbation recovery plot shows deterministic return toward the sampled
  ring for small and moderate hidden-state perturbations; very large
  perturbations begin to produce substantial angular errors. This is a basin /
  recovery analysis, not noise-driven diffusion analysis.

Comparison with the first fixed-delay tuned model:

| Metric | `tuned_delay` | `tuned_delay_stable` |
| --- | ---: | ---: |
| Mean error at 80-step delay | 25.833 deg | 1.453 deg |
| Mean error at 120-step delay | 56.397 deg | 1.903 deg |
| Delay settling ratio | 0.125353 | 0.040939 |
| Mean autonomous drift over 100 probe steps | 57.243 deg | 1.509 deg |
| Mean final autonomous-probe speed | 0.041944 | 0.001455 |

Interpretation:

- The original fixed-delay tuned model learned the `20`-step continuous memory
  but did not maintain a stable continuous state when the delay was extended or
  when the hidden state was probed autonomously.
- The stable tuned model is a much stronger continuous baseline. It preserves
  angle with low error beyond the randomized training range and shows very small
  autonomous drift after the cue has been removed.
- The fixed-point/Jacobian analysis strengthens this from a trajectory-only
  attractor-like interpretation to sampled direct evidence for a
  ring-attractor-like memory structure: approximate blank-delay fixed points
  preserve the remembered angle, the leading eigenvalue is near neutral, the
  leading eigenvector is aligned with the circular manifold, and secondary
  eigenvalues are smaller. The result is still sampled rather than a global
  proof over all hidden states.
- The landscape analysis adds the missing visual state-space check from the
  reference RNN notebook. It shows that fixed-point searches from random bounded
  hidden states also land on the same ring-shaped fixed-point set, so the ring
  is broader than the exact sampled task trajectory states.

Recorded outputs:

- Checkpoint: `outputs/tuned_delay_stable/checkpoints/tuned_delay_stable.pt`.
- Training metrics/history: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_train_metrics.json`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_train_history.csv`.
- Evaluation metrics: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_delay_sweep_metrics.json`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_delay_sweep.csv`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_delay_sweep.png`.
- Stability summary/arrays/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_stability_summary.json`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_stability.npz`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_stability.png`.
- Attractor-probe summary/trials/arrays/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_attractor_probe_summary.json`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_attractor_probe_trials.csv`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_attractor_probe.npz`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_attractor_probe.png`.
- Fixed-point summary/trials/arrays/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_fixed_point_analysis_summary.json`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_fixed_point_analysis_trials.csv`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_fixed_point_analysis.npz`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_fixed_point_analysis.png`.
- Fixed-point landscape summary/points/arrays/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_fixed_point_landscape_summary.json`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_fixed_point_landscape_points.csv`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_fixed_point_landscape.npz`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_fixed_point_landscape.png`.
- Additional dynamics figures summary/arrays/CSV: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_dynamics_figures_summary.json`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_dynamics_figures.npz`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_perturbation_recovery.csv`.
- Additional dynamics figures: `outputs/tuned_delay_stable/figures/tuned_delay_stable_ring_manifold.png`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_decoded_angle_over_time.png`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_jacobian_spectrum.png`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_perturbation_recovery.png`.
- Hidden-state movie outputs: `outputs/tuned_delay_stable/figures/tuned_delay_stable_hidden_state_movie.mp4`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_hidden_state_movie.gif`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_hidden_state_movie.npz`, `outputs/tuned_delay_stable/metrics/tuned_delay_stable_hidden_state_movie_summary.json`.
- PCA summary/arrays/figure: `outputs/tuned_delay_stable/metrics/tuned_delay_stable_pca_summary.json`, `outputs/tuned_delay_stable/arrays/tuned_delay_stable_hidden_states.npz`, `outputs/tuned_delay_stable/figures/tuned_delay_stable_pca_trajectories.png`.

Next action under consideration:

- Promote `configs/tuned_delay_stable.yaml` as the main continuous baseline for
  future perturbation work, after checking whether the result holds across a
  small seed sweep.
- Repeat the fixed-point/Jacobian analysis across a small seed sweep before
  treating the stable tuned result as seed-general.
- Use the stable tuned checkpoint as the cleaner target for psilocybin-informed
  perturbations, because it gives memory precision, drift, and hidden-state
  stability metrics on a continuous representational variable.

</details>

## Chronological Run Log

<details>
<summary>2026-07-15 - Gaussian-vs-baseline supervisor control pack</summary>

Action:

- Added `wm_rnn.gaussian_baseline_figures` to extract clean vs independent
  Gaussian results from the completed five-seed noise experiment.
- Generated dose–delay behaviour, maintenance metrics, geometry, and a focused
  CUDA settling-time analysis (delay 80; 20° threshold).
- Wrote a short markdown summary for the supervisor meeting.

Recorded result:

- At delay 80 and RMS 0.05, Gaussian raised response error from about 6.5° to
  7.5°, decoder error from 1.3° to 3.7°, and drift from 2.6° to 5.2°, with
  fixation essentially unchanged.
- Settling showed a floor near the unscored transition under clean/RMS 0.05 and
  a clearer slowing at RMS 0.10.

Recorded outputs:

- `src/wm_rnn/gaussian_baseline_figures.py`
- `outputs/.../noise_structure/gaussian_control/figures/`
- `outputs/.../noise_structure/gaussian_control/figure_data/`

Interpretation:

- Confirms Gaussian noise as a weak nonspecific control pack for the meeting,
  separate from the benched structured-noise hierarchy.

</details>

<details>
<summary>2026-07-14 - Marker-facing audit of dissertation noise figures</summary>

Action:

- Rebuilt Fig 1 as equation-forward noise definitions with plain-language meaning
  and matched-RMS example traces.
- Replaced Fig 2 correlation heatmaps with a unit-coupling bar summary; relabelled
  traces and autocorrelation axes in plain language.
- Removed unexplained seed spaghetti from Fig 3B; used solid means with dashed CI
  bounds and darker shaded bands; replaced Fig 3C with labelled bars + seed points
  and an in-figure CI legend.
- Updated Fig 4 to solid means, dashed CI bounds, stronger fill, and neutral epoch
  bands that do not clash with condition colours.
- Converted Fig 5 bottom panels to categorical bars with plain-English y-labels and
  an explicit note that the x-axis is not continuous; added PC axis labels.
- Added full independent / AR(1) / topology covariance equations to the revtex
  results report and recompiled the PDF.

Recorded outputs:

- `src/wm_rnn/noise_structure_dissertation_figures.py`
- `docs/reports/noise_structure_perturbation_results.{tex,pdf}`
- `outputs/.../noise_structure/figures/`

Interpretation:

- Presentation and labelling audit only; primary numerical conclusions unchanged.

</details>

<details>
<summary>2026-07-14 - Dissertation-oriented noise-structure figure set</summary>

Action:

- Archived the prior dense figure suite, plotted-data CSVs, and long report under
  `outputs/archive/noise_structure_initial_figure_suite_2026-07-14/`.
- Added `wm_rnn.noise_structure_dissertation_figures` for a six-figure Masters
  results layout: schematic, validation, hero behaviour, delay timecourse, ring
  geometry, and epoch sensitivity.
- Regenerated PNG/PDF figures and plotted-data CSVs from the frozen full and
  epoch-timing datasets, reusing archived hidden-state analyses.
- Replaced `docs/reports/noise_structure_perturbation_results.tex` with a concise
  revtex4-2 results presentation and recompiled the PDF.

Recorded result:

- Primary comparison at RMS 0.05 retained the established ordering independent <
  temporal < topology, with delay-80 response errors of approximately 6.5°,
  7.5°, 16.3°, and 32.4°.
- The new hero figure centers the dissertation claim; condition-specific dose
  grids and hidden-speed panels were demoted out of the main report.

Recorded outputs:

- `src/wm_rnn/noise_structure_dissertation_figures.py`
- `outputs/.../noise_structure/figures/`
- `outputs/.../noise_structure/figure_data/`
- `docs/reports/noise_structure_perturbation_results.{tex,pdf}`
- `outputs/archive/noise_structure_initial_figure_suite_2026-07-14/`

Interpretation:

- Presentation changed; primary numerical conclusions from the five-seed
  experiment did not. The archived suite remains available for denser analysis
  panels.

</details>

<details>
<summary>2026-06-29 - Baseline model trained once</summary>

Action:

- Trained the default baseline model with `configs/baseline_delay.yaml`.

Configuration:

- Task: categorical delayed-response working-memory task.
- Trained delay length: `20` time steps.
- Number of cue classes: `4`.
- Hidden size: `64`.
- Training steps: `1000`.
- Device: CUDA on `NVIDIA GeForce RTX 3060 Laptop GPU`.

Recorded result:

- Final training accuracy: `1.0`.
- Final training loss: `1.415610206834117e-08`.
- Checkpoint: `outputs/baseline_delay/checkpoints/baseline_delay.pt`.
- Training history: `outputs/baseline_delay/metrics/baseline_delay_train_history.csv`.
- Training metrics: `outputs/baseline_delay/metrics/baseline_delay_train_metrics.json`.

</details>

<details>
<summary>2026-06-29 - Baseline checkpoint evaluated</summary>

Action:

- Evaluated the trained baseline checkpoint on held-out generated delay-task batches.

Recorded result:

- Evaluation accuracy: `1.0`.
- Evaluation batches: `20`.
- Metrics: `outputs/baseline_delay/metrics/baseline_delay_eval_metrics.json`.
- Confusion matrix: `outputs/baseline_delay/metrics/baseline_delay_confusion_matrix.csv`.

</details>

<details>
<summary>2026-06-29 - Hidden-state PCA analysis generated</summary>

Action:

- Ran PCA analysis on hidden-state trajectories from the trained checkpoint.

Recorded outputs:

- PCA summary: `outputs/baseline_delay/metrics/baseline_delay_pca_summary.json`.
- Hidden-state arrays: `outputs/baseline_delay/arrays/baseline_delay_hidden_states.npz`.
- PCA figure: `outputs/baseline_delay/figures/baseline_delay_pca_trajectories.png`.

</details>

<details>
<summary>2026-07-01 - Single-checkpoint delay sweep run</summary>

Action:

- Loaded the trained baseline checkpoint.
- Kept model weights frozen.
- Evaluated response accuracy across longer delay lengths without retraining.

Latest recorded sweep:

| Delay steps | Accuracy |
| ---: | ---: |
| 20 | 1.000 |
| 25 | 1.000 |
| 30 | 0.907 |
| 35 | 0.749 |
| 40 | 0.745 |
| 50 | 0.519 |
| 60 | 0.485 |
| 70 | 0.483 |
| 80 | 0.503 |

Interpretation:

- The model remains reliable around the trained delay length.
- Accuracy begins to degrade around `30` delay steps.
- The major transition occurs around `30-50` delay steps.

Recorded outputs:

- Metrics: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep_metrics.json`.
- CSV: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep.csv`.
- Figure: `outputs/baseline_delay/figures/baseline_delay_delay_sweep.png`.

</details>

<details>
<summary>2026-07-01 - Multiple seeds trained and swept</summary>

Action:

- Trained independent baseline models for seeds `101`, `102`, `103`, and `104`.
- Evaluated each trained model.
- Ran a frozen-weight delay sweep for each seed.

Shared delay sweep:

```text
20 25 30 35 40 50 60 70 80
```

Per-seed training/evaluation result:

| Seed | Final train accuracy | Evaluation accuracy |
| ---: | ---: | ---: |
| 101 | 1.000 | 1.000 |
| 102 | 1.000 | 1.000 |
| 103 | 1.000 | 1.000 |
| 104 | 1.000 | 1.000 |

Per-seed outputs:

- Seed `101`: `outputs/baseline_delay/seed_sweep/seed_101/`.
- Seed `102`: `outputs/baseline_delay/seed_sweep/seed_102/`.
- Seed `103`: `outputs/baseline_delay/seed_sweep/seed_103/`.
- Seed `104`: `outputs/baseline_delay/seed_sweep/seed_104/`.

Summary outputs:

- Summary JSON: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep_summary.json`.
- Summary CSV: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep.csv`.

</details>

<details>
<summary>2026-07-01 - Combined multi-seed delay plot generated</summary>

Action:

- Read the multi-seed summary file.
- Loaded each seed's delay-sweep CSV.
- Plotted all seed curves on one graph with an aggregate curve.

Included seeds:

```text
101 102 103 104
```

Included delays:

```text
20 25 30 35 40 50 60 70 80
```

Recorded output:

- Combined figure: `outputs/baseline_delay/figures/baseline_delay_seed_sweep_delay_curves.png`.

Interpretation:

- This plot is the current best overview of whether delay-related memory degradation is consistent across independently trained models.

</details>

<details>
<summary>2026-07-01 - Hidden-state stability analysis run on the baseline checkpoint</summary>

Action:

- Loaded the trained baseline checkpoint (`outputs/baseline_delay/checkpoints/baseline_delay.pt`), kept weights frozen.
- Generated a fresh 64-trial analysis batch at the trained task settings (`cue_steps=5`, `delay_steps=20`, `response_steps=5`).
- Measured hidden-state norm (magnitude) and step-to-step speed (the size of the change in hidden state between consecutive time steps) at every time step, averaged across trials.

Recorded result:

| Phase | Mean hidden-state norm | Mean step-to-step speed |
| --- | ---: | ---: |
| Cue (steps 0-5) | 1.86 | 0.73 |
| Delay (steps 5-25) | 39.54 | 7.69 |
| Response (steps 25-30) | 232.06 | 41.36 |

- Early-delay speed (first third of the delay period): `1.58`.
- Late-delay speed (last third of the delay period): `17.09`.
- Delay settling ratio (late-delay speed / early-delay speed): `10.83`.

Interpretation:

- The hidden-state norm and step-to-step speed both grow smoothly and
  substantially throughout the delay period and continue accelerating into
  the response period, rather than flattening out.
- A delay settling ratio well above `1` (here, about `10.8x`) means the
  hidden state is moving faster, not slower, by the end of the delay. This
  is the opposite of a settling, attractor-like signature, where step-to-step
  speed would be expected to shrink toward zero as the network approaches a
  stable state.
- This is consistent with the baseline model having learned a ramping or
  phasic solution rather than a tonic fixed-point attractor: the network
  only needs the hidden state to be decodable at the one fixed, trained
  response time, so nothing in the training objective rewards the state for
  settling down earlier.
- This also gives a direct mechanistic explanation for the delay-length
  sweep result recorded above: a growing, never-settled trajectory is
  expected to become progressively less predictable and harder to read out
  correctly the further past the trained delay length it is pushed, which
  matches the observed accuracy collapse beyond `30` delay steps.
- This finding does not mean the baseline model is broken; per Ghazizadeh and
  Ching (2021, "Slow Manifolds in Recurrent Networks Encode Working Memory
  Efficiently and Robustly"; see the dissertation wiki literature notes),
  trained working-memory RNNs can legitimately use fixed-point, limit-cycle,
  or slow-manifold/ramping mechanisms, and behavioral accuracy alone does
  not reveal which one is present. It does mean this baseline currently
  looks like the ramping case rather than a settled attractor case, which is
  a meaningful data point before layering psilocybin-informed perturbations
  (for example reduced recurrent stability) on top of it.

Recorded outputs:

- Figure: `outputs/baseline_delay/figures/baseline_delay_stability.png`.
- Summary: `outputs/baseline_delay/metrics/baseline_delay_stability_summary.json`.
- Arrays: `outputs/baseline_delay/arrays/baseline_delay_stability.npz`.

Next action under consideration:

- Decide whether to retrain with the delay period included in the training
  loss and/or with randomized delay length per batch, specifically to test
  whether either change produces a lower (more settled) delay settling
  ratio, before doing any fixed-point or Jacobian-level analysis.

</details>

<details>
<summary>2026-07-01 - Implementation note: added a whole-delay-loss and randomized-delay training variant, kept as a separate versioned run</summary>

Reasoning:

- The stability analysis above showed the original baseline checkpoint does
  not settle during the delay period; step-to-step speed grows throughout
  the delay instead of shrinking. Two changes were identified as candidate,
  complementary fixes to test before any architecture change:
  1. Score the training loss across the whole delay period, not only the
     response window, so the network is directly rewarded for staying
     decodable at every delay time step, not only at one fixed offset.
  2. Randomize the delay length used in each training batch, so the network
     cannot rely on a solution that only needs to be correct at one exact,
     fixed trained delay.
- These two changes were judged complementary rather than redundant: (1)
  creates pressure to stay correct at every moment, and (2) extends that
  pressure across a range of durations instead of one fixed duration.
- Both changes are implemented as opt-in configuration so the original
  `configs/baseline_delay.yaml` config and its already-recorded results are
  completely unaffected.
- Outputs for this variant are written to a separate top-level output
  directory (`outputs/baseline_delay_stable/`) rather than being mixed into
  `outputs/baseline_delay/`, so figures, metrics, checkpoints, and arrays
  are kept in their own folder per model version and can be compared
  side by side without overwriting or interleaving files. This follows the
  same per-run folder pattern already used by `seed_sweep.py`.

File changes:

- `src/wm_rnn/training_utils.py`: Added `with_delay_steps`, a small helper
  that returns a copy of a task config with a different delay length,
  mirroring the existing `with_batch_size` helper.
- `src/wm_rnn/train.py`: Added two opt-in training behaviors, both disabled
  by default:
  - If `task.delay_steps_min` and `task.delay_steps_max` are both set, each
    training step now samples a random delay length from that inclusive
    range (using a seeded, reproducible random generator) instead of always
    using the fixed `task.delay_steps`.
  - If `training.score_delay_period` is `true`, the training loss is
    computed over the delay period plus the response period, instead of the
    response period only. Response-period accuracy is still always logged
    using the original response-only mask, so training curves stay
    comparable across configs regardless of this setting. The sampled delay
    length for each step is now also recorded in the training history CSV.
  - Final training metrics JSON now also records whether whole-delay
    scoring and delay randomization were used, and the configured delay
    range if randomization was on.
- `configs/baseline_delay_stable.yaml`: Added a new, separate experiment
  config that keeps the model architecture identical to the original
  baseline (`hidden_size=64`, `dt=20.0`, `tau=100.0`) and only changes the
  training objective: `delay_steps_min=15`, `delay_steps_max=45`,
  `score_delay_period=true`, and `steps=2000` (increased from `1000` because
  the task is harder). `paths.output_dir` is set to
  `outputs/baseline_delay_stable` and `paths.run_name` to
  `baseline_delay_stable`, so every output lands in its own folder tree
  separate from the original baseline run. `task.delay_steps` is kept at
  `20` so evaluation, delay-sweep, PCA, and stability analyses still probe
  the same reference delay length as the original baseline for comparison.

</details>

<details>
<summary>2026-07-01 - Implementation note: added a configurable tanh activation and an isolated tanh test variant</summary>

Reasoning:

- The whole-delay-loss / randomized-delay variant improved how far the
  delay length could be pushed before accuracy collapsed, but the hidden
  state was still accelerating rather than settling (delay settling ratio
  about `4.8x`, still above `1`). That pointed at a second, separate cause:
  the recurrent layer's `relu` nonlinearity has no upper bound, so nothing
  in the architecture itself prevents the hidden state from growing forever,
  regardless of how the loss is scored.
- Switching the hidden-state nonlinearity to `tanh` was proposed as a
  structural fix rather than a training-incentive fix: `tanh` bounds hidden
  activity to `(-1, 1)`, so unbounded growth becomes mathematically
  impossible rather than merely discouraged.
- This was deliberately tested in isolation: same fixed `20`-step delay,
  same response-period-only loss, same `1000` training steps as the
  original baseline, changing only the activation function. This isolates
  whether bounding alone (independent of the training-objective changes
  already tested) produces settled hidden-state dynamics.
- `relu` gives only non-negative activity, loosely resembling a firing rate;
  `tanh` allows negative activity and is the standard choice in most
  trained-RNN dynamical-systems literature (for example the kind of
  fixed-point-finding analysis flagged as a future next step), so this is
  recorded as a real modelling choice, not only an implementation detail.

File changes:

- `src/wm_rnn/model.py`: Added an `activation` field to `RNNConfig`
  (`"relu"` by default, or `"tanh"`), with validation against unsupported
  values. `CTRNN` now resolves the configured activation once at
  construction time and applies it in `recurrence()` instead of a hardcoded
  `torch.relu` call. Docstrings updated to describe both activation options.
- `src/wm_rnn/training_utils.py`: `model_config_from_dict` now reads
  `model.activation` from the config dictionary and passes it through to
  `RNNConfig`. This originally fell back to `"relu"` if absent; after
  `tanh` was promoted to the canonical baseline, the fallback was updated to
  `"tanh"`.
- `src/wm_rnn/config.py`: Added an explicit `activation` setting to the
  default model configuration so the setting is discoverable and documented
  even when a YAML config omits it. This was initially added as `"relu"` and
  later updated to `"tanh"` when `tanh` became the canonical baseline.
- `configs/baseline_delay_tanh.yaml`: Added a new experiment config that is
  otherwise identical to `configs/baseline_delay.yaml` (same task timing,
  same `1000`-step response-only training) except `model.activation: tanh`.
  `paths.output_dir` is `outputs/baseline_delay_tanh` and `paths.run_name`
  is `baseline_delay_tanh`, giving this variant its own separate output
  folder tree, consistent with the other variants.

</details>

<details>
<summary>2026-07-01 - Trained and analyzed the whole-delay-loss / randomized-delay variant, compared against the original baseline</summary>

Action:

- Trained a new model with `configs/baseline_delay_stable.yaml`: same
  architecture as the original baseline, but with the training delay length
  sampled per step from `[15, 45]` and the training loss scored across the
  delay period plus the response period, for `2000` steps.
- Evaluated the trained checkpoint, ran the same frozen-weight delay-length
  sweep used for the original baseline (`20, 25, 30, 35, 40, 50, 60, 70,
  80`), ran the PCA trajectory analysis, and ran the hidden-state stability
  analysis, all against this new checkpoint.
- All outputs were written to `outputs/baseline_delay_stable/` (separate
  `checkpoints/`, `metrics/`, `figures/`, and `arrays/` subfolders), so they
  do not overwrite or mix with `outputs/baseline_delay/`.

Recorded result - training and evaluation:

- Final training loss: `0.0038`. Final training (response-period) accuracy:
  `1.000`, reached consistently across randomly sampled delay lengths from
  step `100` onward.
- Held-out evaluation accuracy at the reference `20`-step delay: `1.000`.

Recorded result - delay-length sweep, compared with the original baseline:

| Delay steps | Original baseline accuracy | Stable-variant accuracy |
| ---: | ---: | ---: |
| 20 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 |
| 30 | 0.907 | 1.000 |
| 35 | 0.749 | 1.000 |
| 40 | 0.745 | 1.000 |
| 50 | 0.519 | 1.000 |
| 60 | 0.485 | 0.712 |
| 70 | 0.483 | 0.510 |
| 80 | 0.503 | 0.511 |

Recorded result - hidden-state stability, compared with the original baseline:

| Metric | Original baseline | Stable variant |
| --- | ---: | ---: |
| Early-delay speed | 1.58 | 1.20 |
| Late-delay speed | 17.09 | 5.72 |
| Delay settling ratio (late / early) | 10.83 | 4.76 |

Interpretation:

- The delay-length sweep shows a large, direct improvement: the original
  baseline started degrading past `25-30` steps, while the stable variant
  holds perfect accuracy all the way to `50` steps, which lines up closely
  with the trained random-delay upper bound of `45`. Accuracy still falls
  off beyond that, roughly toward the same chance-level plateau seen in the
  original baseline by `70-80` steps.
- The hidden-state stability analysis shows real but partial improvement:
  the delay settling ratio dropped from about `10.8x` to about `4.8x`, and
  both early- and late-delay speeds are lower. The hidden state is still
  accelerating through the delay rather than flattening toward zero, so
  this is not yet a settled, tonic attractor state; it is a less extreme,
  longer-range version of the same ramping signature seen in the original
  baseline (confirmed visually in
  `outputs/baseline_delay_stable/figures/baseline_delay_stable_pca_trajectories.png`,
  where hidden-state trajectories still diverge outward by cue class rather
  than curving toward fixed points).
- Taken together, this suggests the two training changes successfully
  widened the range of delay lengths the network can handle reliably
  (matching the literal range it was trained on, plus some margin), but did
  not by themselves convert the underlying mechanism into a settled
  attractor. This is consistent with the ramping/phasic account of the
  original baseline rather than a full mechanism change: the network
  appears to have learned a ramp that stays correct over a wider trained
  window, rather than learning to stop ramping.
- This is a meaningful, literature-consistent result rather than a null
  result: Ghazizadeh and Ching (2021) describe slow-manifold/phasic
  solutions as a legitimate, sometimes more efficient alternative to tonic
  fixed-point attractors, and this result is consistent with the network
  favoring that kind of solution even when the training objective directly
  rewards delay-period stability.

Recorded outputs:

- Training metrics: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_train_metrics.json`.
- Training history: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_delay_sweep_metrics.json`, `outputs/baseline_delay_stable/metrics/baseline_delay_stable_delay_sweep.csv`, `outputs/baseline_delay_stable/figures/baseline_delay_stable_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay_stable/figures/baseline_delay_stable_pca_trajectories.png`, `outputs/baseline_delay_stable/arrays/baseline_delay_stable_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay_stable/figures/baseline_delay_stable_stability.png`, `outputs/baseline_delay_stable/metrics/baseline_delay_stable_stability_summary.json`, `outputs/baseline_delay_stable/arrays/baseline_delay_stable_stability.npz`.

Next action under consideration:

- If a genuinely settled, tonic attractor state is desired rather than a
  wider-range ramp, consider a change that directly bounds or penalizes
  hidden-state growth (for example a bounded nonlinearity such as `tanh`,
  or an explicit hidden-state norm penalty) rather than further widening the
  trained delay range.
- Fixed-point or Jacobian-level analysis would give a more direct answer
  than trajectory speed alone, and is a reasonable next step now that two
  checkpoints with meaningfully different delay-generalization behavior
  exist to compare.

</details>

<details>
<summary>2026-07-01 - Trained and analyzed the isolated tanh-activation variant, compared against both prior runs</summary>

Action:

- Trained a new model with `configs/baseline_delay_tanh.yaml`: identical
  training setup to the original baseline (fixed `20`-step delay,
  response-period-only loss, `1000` steps), with only `model.activation`
  changed from `relu` to `tanh`.
- Ran the same evaluation, delay-length sweep (`20, 25, 30, 35, 40, 50, 60,
  70, 80`), PCA trajectory analysis, and hidden-state stability analysis
  used for the previous two runs, against this checkpoint.
- All outputs were written to `outputs/baseline_delay_tanh/`, kept separate
  from both `outputs/baseline_delay/` and `outputs/baseline_delay_stable/`.

Recorded result - training and evaluation:

- Final training loss: `0.0019`. Final training accuracy: `1.000`, reached
  by step `150` and stable for the remainder of training.
- Held-out evaluation accuracy at the reference `20`-step delay: `1.000`.

Recorded result - delay-length sweep, compared with both prior runs:

| Delay steps | Original baseline (relu) | Whole-delay-loss variant (relu) | Tanh variant |
| ---: | ---: | ---: | ---: |
| 20 | 1.000 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 | 1.000 |
| 30 | 0.907 | 1.000 | 1.000 |
| 35 | 0.749 | 1.000 | 1.000 |
| 40 | 0.745 | 1.000 | 1.000 |
| 50 | 0.519 | 1.000 | 1.000 |
| 60 | 0.485 | 0.712 | 1.000 |
| 70 | 0.483 | 0.510 | 1.000 |
| 80 | 0.503 | 0.511 | 1.000 |

Recorded result - hidden-state stability, compared with both prior runs:

| Metric | Original baseline (relu) | Whole-delay-loss variant (relu) | Tanh variant |
| --- | ---: | ---: | ---: |
| Early-delay speed | 1.58 | 1.20 | 0.56 |
| Late-delay speed | 17.09 | 5.72 | 0.03 |
| Delay settling ratio (late / early) | 10.83 | 4.76 | **0.06** |

Interpretation:

- The tanh variant reaches perfect response accuracy at every tested delay
  length, including `80` steps, four times the trained `20`-step delay, and
  it does this using the plain original training setup (fixed delay,
  response-only loss, no extra training steps). This outperforms the
  whole-delay-loss/randomized-delay variant, which needed a harder training
  procedure and still degraded past `50` steps.
- The hidden-state stability numbers now show genuine settling rather than
  a smaller ramp: the delay settling ratio dropped from `10.83` (original)
  and `4.76` (whole-delay-loss variant) to `0.06` for the tanh variant. A
  ratio well below `1` means late-delay speed is much smaller than
  early-delay speed, that is, the hidden state is slowing down and
  flattening out as the delay continues, not accelerating.
- This is visible directly in the recorded figures: hidden-state norm
  (`outputs/baseline_delay_tanh/figures/baseline_delay_tanh_stability.png`)
  rises during the cue period and then visibly plateaus partway through the
  delay instead of continuing to climb; step-to-step speed rises briefly
  after the cue, then decays toward near zero by the end of the delay. The
  PCA trajectory figure
  (`outputs/baseline_delay_tanh/figures/baseline_delay_tanh_pca_trajectories.png`)
  shows trajectories that curve and slow into a settled region per cue
  class, rather than the straight, ever-diverging rays seen in both `relu`
  runs.
- Taken together, this points to the unbounded `relu` nonlinearity, not the
  training objective, as the primary cause of the original baseline's poor
  delay generalization. The whole-delay-loss and randomized-delay changes
  did help the `relu` network cope with a wider range of delays, but they
  could not make it settle, because nothing in that architecture prevents
  continued growth. Bounding the nonlinearity removed the underlying
  capacity for unbounded growth directly, and a settled representation
  emerged from the same simple training setup as the original baseline,
  with no need for the more complex training changes.
- This is the first run in this project to show a hidden-state signature
  consistent with a genuinely settled, tonic, attractor-like mechanism
  rather than a ramping or phasic one, in the terms used by Ghazizadeh and
  Ching (2021). It should still be confirmed with a direct fixed-point or
  Jacobian-level analysis rather than trajectory speed alone before treating
  it as a confirmed attractor.

Recorded outputs:

- Training metrics: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_train_metrics.json`.
- Training history: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_delay_sweep_metrics.json`, `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_delay_sweep.csv`, `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_pca_trajectories.png`, `outputs/baseline_delay_tanh/arrays/baseline_delay_tanh_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_stability.png`, `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_stability_summary.json`, `outputs/baseline_delay_tanh/arrays/baseline_delay_tanh_stability.npz`.

Next action under consideration:

- Decide whether `tanh` should become the new default baseline activation
  (updating `configs/baseline_delay.yaml` and `docs/model-architecture.md`)
  given this result, or whether `relu` should be kept as the documented
  original baseline with `tanh` recorded as a compared variant.
- Run a fixed-point or Jacobian-level analysis on the tanh checkpoint to
  directly confirm attractor structure rather than inferring it from
  trajectory speed and norm alone.
- Consider whether combining `tanh` with the whole-delay-loss/randomized-delay
  training changes gives any further benefit, now that each change has been
  tested in isolation.

</details>

<details>
<summary>2026-07-01 - Implementation note: promoted tanh to the canonical baseline activation</summary>

Reasoning:

- The isolated tanh test above showed perfect response accuracy at every
  tested delay length (up to `80` steps, four times the trained length) and
  a delay settling ratio of `0.06`, using the exact same simple training
  setup as the original `relu` baseline. This was judged strong enough
  evidence to make `tanh` the default baseline activation rather than a
  separately tracked variant, since it is a strictly better-generalizing,
  more settled network produced by the same architecture family and the
  same training procedure, not a different modelling approach.
- At the time of this entry, the original `relu`-based baseline config and
  recorded outputs were archived under clearly named files/folders. This was
  later superseded by the 2026-07-07 active-baseline cleanup, which removed the
  old relu config from the tracked active config set while preserving the
  historical run record in this changelog.
- The standalone `configs/baseline_delay_tanh.yaml` config and its
  `outputs/baseline_delay_tanh/` outputs were removed, because after this
  change they would be an exact duplicate of the new canonical
  `configs/baseline_delay.yaml` / `outputs/baseline_delay/`.

File changes:

- `configs/baseline_delay_relu.yaml`: Added as an exact archival copy of the
  pre-change `configs/baseline_delay.yaml` (fixed `20`-step delay,
  response-only loss, `1000` steps, `model.activation: relu`), with
  `paths.output_dir` set to `outputs/baseline_delay_relu` and
  `paths.run_name` set to `baseline_delay_relu` so it can be re-run
  independently without colliding with the new baseline.
- `configs/baseline_delay.yaml`: Added `model.activation: tanh`. No other
  fields changed; task timing, training steps, and output paths
  (`outputs/baseline_delay/`, run name `baseline_delay`) are unchanged, so
  this file now defines the new canonical baseline in place.
- `configs/baseline_delay_tanh.yaml`: Removed (superseded by the change
  above; would otherwise duplicate `configs/baseline_delay.yaml`).
- `docs/model-architecture.md`: Updated the "Recurrent Core" section to
  describe the hidden-state nonlinearity as configurable with `tanh` as the
  baseline default, documented the `activation` config field, generalized
  the per-step update formula from a hardcoded `relu` call to
  `activation(...)`, and added a short explanation of why `tanh` replaced
  `relu` as the default, pointing to this changelog for full detail.
- `README.md`: Updated the top-line description, and rewrote "Model
  Variants" to describe the then-current `tanh`, archived `relu`, and
  stable-delay comparison configs. This README state was later superseded by
  the 2026-07-07 cleanup, which keeps only `configs/baseline_delay.yaml` as the
  active categorical baseline.

Filesystem changes (outputs, not tracked by Git since `outputs/` is
git-ignored):

- Renamed `outputs/baseline_delay/` to `outputs/baseline_delay_relu/`,
  archiving the original `relu` baseline's checkpoint, metrics, figures,
  arrays, and multi-seed sweep results without modification. File names
  inside this archived folder still carry the original `baseline_delay_*`
  prefix rather than `baseline_delay_relu_*`, since they were generated
  before this archiving step; the containing folder name is authoritative
  for identifying these as the archived `relu` run.
- Deleted `outputs/baseline_delay_tanh/` (superseded by the retrained
  canonical baseline below).
- Retrained a fresh checkpoint into `outputs/baseline_delay/` using the
  updated `configs/baseline_delay.yaml`, and re-ran evaluation, the
  delay-length sweep, PCA analysis, and stability analysis against it (see
  run-log entry below). Because this config is otherwise identical to the
  earlier isolated tanh test, results are the same; this run exists so the
  checkpoint's embedded config and all output file names correctly reflect
  the `baseline_delay` identity going forward.

</details>

<details>
<summary>2026-07-01 - Retrained the canonical baseline_delay checkpoint under tanh and confirmed results</summary>

Action:

- Trained `outputs/baseline_delay/checkpoints/baseline_delay.pt` using the
  updated `configs/baseline_delay.yaml` (`model.activation: tanh`, fixed
  `20`-step delay, response-only loss, `1000` steps).
- Re-ran evaluation, the same delay-length sweep (`20, 25, 30, 35, 40, 50,
  60, 70, 80`), PCA trajectory analysis, and hidden-state stability
  analysis against this checkpoint, all writing into `outputs/baseline_delay/`.

Recorded result:

- Final training loss: `0.0019`. Final training accuracy: `1.000`.
- Held-out evaluation accuracy at the `20`-step reference delay: `1.000`.
- Delay-length sweep: `1.000` accuracy at every tested delay length from
  `20` through `80` steps.
- Hidden-state stability: early-delay speed `0.56`, late-delay speed
  `0.03`, delay settling ratio `0.06`.

Interpretation:

- These numbers are identical (within floating-point rounding) to the
  isolated tanh test recorded earlier, as expected: the only configuration
  difference is the output path and run name, not the model, data, or
  training procedure. This run exists to make `outputs/baseline_delay/`
  and its embedded checkpoint config correctly represent the current
  canonical baseline, rather than to test a new hypothesis.
- `outputs/baseline_delay/` now documents a settled, delay-length-robust
  baseline. `outputs/baseline_delay_relu/` remains available as the
  archived original baseline for any comparison that specifically needs
  the earlier `relu`-based, ramping-dynamics network.

Recorded outputs:

- Training metrics: `outputs/baseline_delay/metrics/baseline_delay_train_metrics.json`.
- Training history: `outputs/baseline_delay/metrics/baseline_delay_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay/metrics/baseline_delay_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep_metrics.json`, `outputs/baseline_delay/metrics/baseline_delay_delay_sweep.csv`, `outputs/baseline_delay/figures/baseline_delay_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay/figures/baseline_delay_pca_trajectories.png`, `outputs/baseline_delay/arrays/baseline_delay_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay/figures/baseline_delay_stability.png`, `outputs/baseline_delay/metrics/baseline_delay_stability_summary.json`, `outputs/baseline_delay/arrays/baseline_delay_stability.npz`.

Next action under consideration:

- Run a fixed-point or Jacobian-level analysis on this checkpoint to
  directly confirm attractor structure, since this is now the checkpoint
  that will anchor future baseline comparisons and psilocybin-informed
  perturbations.
- Superseded by the 2026-07-07 active-baseline cleanup: there is no longer an
  active `baseline_delay_stable` config. Any future whole-delay-loss or
  randomized-delay categorical experiment should be introduced as a new config
  with a clear name and rationale, not as the current baseline.
- Consider whether the existing multi-seed sweep tooling
  (`src/wm_rnn/seed_sweep.py`) should be re-run under the new `tanh`
  baseline to confirm the settled, delay-robust behavior holds across
  seeds, since the original multi-seed results in
  `outputs/baseline_delay_relu/seed_sweep/` were only ever measured for the
  `relu` network.

</details>

<details>
<summary>2026-07-01 - Re-ran the multi-seed delay sweep under the tanh baseline</summary>

Action:

- Trained independent `tanh`-based baseline models for seeds `101`, `102`,
  `103`, and `104` using `configs/baseline_delay.yaml`, evaluated each, and
  ran the same frozen-weight delay sweep (`20, 25, 30, 35, 40, 50, 60, 70,
  80`) used for every other run in this project.
- Plotted all four seed curves together with a mean and `+/- 1 SD` band,
  same as the original `relu` multi-seed plot.
- All outputs were written under `outputs/baseline_delay/seed_sweep/`,
  replacing the prior `tanh`-baseline seed-sweep summary location (the
  original `relu` four-seed results remain untouched and available at
  `outputs/baseline_delay_relu/seed_sweep/`).

Recorded result - training and evaluation:

- All four seeds reached `1.000` final training accuracy and `1.000`
  held-out evaluation accuracy at the `20`-step reference delay.

Recorded result - delay-length sweep, `tanh` seeds compared with the archived `relu` seeds:

| Delay steps | relu seed 101 | relu seed 102 | relu seed 103 | relu seed 104 | relu mean | tanh seed 101 | tanh seed 102 | tanh seed 103 | tanh seed 104 | tanh mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 0.950 | 1.000 | 1.000 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 0.747 | 0.723 | 0.517 | 0.910 | 0.724 | 0.781 | 1.000 | 0.899 | 1.000 | 0.920 |
| 60 | 0.730 | 0.418 | 0.509 | 0.484 | 0.535 | 0.747 | 1.000 | 0.757 | 1.000 | 0.876 |
| 80 | 0.754 | 0.245 | 0.455 | 0.504 | 0.489 | 0.745 | 1.000 | 0.711 | 1.000 | 0.864 |

Interpretation:

- This is a more complete and more honest picture than the single-seed
  result recorded earlier. Under `relu`, all four seeds degrade toward or
  below the neighborhood of chance (`0.25`) by `80` steps, with high
  seed-to-seed variability (`0.245` to `0.754`) and no seed staying
  reliable. Under `tanh`, two of four seeds (`102`, `104`) hold perfect
  `1.000` accuracy through `80` steps, matching the single-seed result
  reported earlier, while the other two (`101`, `103`) settle to roughly
  `0.71`-`0.78` beyond `40` steps rather than continuing to degrade toward
  chance.
- The earlier claim that switching to `tanh` produces generalization "at
  least four times longer than trained" was accurate for the one seed used
  in the initial checkpoint analysis, but should not be read as a
  guarantee for every randomly initialized network. `docs/model-architecture.md`
  has been updated to state this more precisely: strong, well-above-chance
  generalization is typical under `tanh`, and perfect generalization occurs
  for some but not all seeds.
- Even the weaker `tanh` seeds are still a clear improvement over every
  `relu` seed at long delays: the worst `tanh` seed at `80` steps (`0.711`)
  is comparable to or better than the best `relu` seed at `80` steps
  (`0.754`), and the `tanh` mean (`0.864`) is far above the `relu` mean
  (`0.489`) with a much tighter spread. The core conclusion, that bounding
  the nonlinearity substantially improves delay-length generalization, is
  unchanged; what changes is that it should be described as a strong,
  typical improvement rather than a uniform guarantee.
- This also means the `tanh` network's settling behavior may itself vary
  by seed (some seeds may find a fully settled fixed-point-like solution,
  others a partially settled or slower-drifting one). This is a natural
  candidate follow-up: running the hidden-state stability analysis on the
  `101`/`103` (partial) and `102`/`104` (perfect) checkpoints and comparing
  their settling ratios would show whether degree of settling predicts
  degree of long-delay accuracy, rather than assuming it from accuracy
  alone.

Recorded outputs:

- Summary JSON: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep_summary.json`.
- Summary CSV: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep.csv`.
- Combined figure: `outputs/baseline_delay/figures/baseline_delay_seed_sweep_delay_curves.png`.
- Per-seed checkpoints, metrics, and figures: `outputs/baseline_delay/seed_sweep/seed_101/` through `seed_104/`.

Next action under consideration:

- Run the hidden-state stability analysis on each of the four seed
  checkpoints to test whether settling ratio predicts long-delay accuracy
  within the `tanh` baseline, not just between `tanh` and `relu`.
- Revisit whether the still-open `tanh` + whole-delay-loss/randomized-delay
  combination (see the "superseded or worth testing" discussion) would
  reduce this seed-to-seed variability, since that combination directly
  rewards delay-period stability during training rather than relying on
  the bounded nonlinearity alone.

</details>

<details>
<summary>2026-07-07 - Tuned continuous model and stable tuned attractor analyses committed</summary>

Action:

- Added and analyzed the continuous circular population-code task requested as
  the next model iteration after supervisor feedback.
- Trained and analyzed `configs/tuned_delay_stable.yaml` as the stronger
  continuous working-memory baseline.
- Added delay sweeps, stability analysis, autonomous drift probing,
  fixed-point/Jacobian analysis, fixed-point landscape plotting, deterministic
  perturbation recovery, and a hidden-state movie for the tuned model.

Current interpretation:

- The fixed-delay tuned model learns the trained delay but drifts over longer
  delays and autonomous probe periods.
- The stable tuned model preserves the remembered circular angle with low error
  across extended delays and shows sampled ring-attractor-like structure in
  hidden-state analyses.
- The attractor evidence is sampled evidence in projected and probed state
  space, not a complete proof over the full 64-dimensional hidden-state space.

Recorded outputs:

- Stable tuned checkpoint: `outputs/tuned_delay_stable/checkpoints/tuned_delay_stable.pt`.
- Main analysis figures: `outputs/tuned_delay_stable/figures/`.
- Main analysis arrays: `outputs/tuned_delay_stable/arrays/`.
- Main analysis metrics: `outputs/tuned_delay_stable/metrics/`.
- Hidden-state movie: `outputs/tuned_delay_stable/figures/tuned_delay_stable_hidden_state_movie.mp4`.

Visualization note:

- The ring may appear flipped between the static ring/landscape figures and the
  movie because those visualizations currently fit separate two-dimensional PCA
  projections. PCA component signs are arbitrary. This is a display-orientation
  issue, not evidence that the underlying ring attractor changed.

</details>

<details>
<summary>2026-07-07 - Stable model summary report compiled</summary>

Action:

- Added `docs/reports/stable_model_summary.tex`.
- Compiled `docs/reports/stable_model_summary.pdf`.
- Updated the report to compare the current categorical baseline
  `baseline_delay` against the stable tuned continuous model
  `tuned_delay_stable`.

Report interpretation:

- `baseline_delay` is the categorical tanh baseline and remains behaviorally
  strong on the categorical task.
- `tuned_delay_stable` is the better current candidate for continuous
  attractor-style working-memory analysis because it provides a circular memory
  variable, angular error metrics, fixed-point/Jacobian diagnostics, and
  perturbation-recovery analysis.

</details>

<details>
<summary>2026-07-07 - Active categorical baseline cleanup</summary>

Action:

- Removed `configs/baseline_delay_relu.yaml` and
  `configs/baseline_delay_stable.yaml` from the active config set.
- Kept `configs/baseline_delay.yaml` as the only current categorical baseline.
- Updated README, architecture documentation, changelog, and the stable model
  summary report so active comparisons no longer present relu-based categorical
  variants as current models.

Current active model state:

- Categorical baseline: `configs/baseline_delay.yaml`, with
  `model.activation: tanh`.
- Continuous tuned model: `configs/tuned_delay.yaml`.
- Stable continuous tuned model for attractor analyses:
  `configs/tuned_delay_stable.yaml`.

Historical note:

- Older run-log entries still describe the original relu baseline and the
  relu-based randomized-delay/whole-delay-loss categorical variant. Those are
  historical experiment records only. They should not be treated as active model
  variants for new analyses.

</details>

<details>
<summary>2026-07-07 - Stable summary report figure-caption clarifications</summary>

Action:

- Updated the caption for the fixed-point landscape figure in
  `docs/reports/stable_model_summary.tex`.
- Reworded the caption so each panel is explained directly: task trajectories
  plus fixed-point endpoints, decoded-angle coloring, fixed-point residuals, and
  the angular-error histogram. The caption now states that the histogram panel is
  mainly a quality-control view rather than a separate geometry plot.
- Updated the caption for the fixed-point/Jacobian figure in
  `docs/reports/stable_model_summary.tex`.
- Reworded the caption so each panel is explained directly: fixed-point search
  residual, decoded fixed-point error, drift from late-delay trajectory state,
  and Jacobian eigenvalue magnitudes.

Interpretation:

- This is a readability update only. It does not change the underlying analysis
  results or regenerate model outputs.

Recorded outputs:

- Updated report source: `docs/reports/stable_model_summary.tex`.
- Recompiled report PDF: `docs/reports/stable_model_summary.pdf`.

</details>
