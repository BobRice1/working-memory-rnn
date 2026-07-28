# Dissertation scaffold (two-column REVTeX)

This folder is the growing full-dissertation LaTeX draft in APS/REVTeX
two-column style.

## Build

```powershell
cd working-memory-rnn/docs/dissertation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## Section status

| Section | Status |
|---|---|
| I Introduction | Drafted from the old report intro |
| II Literature | Drafted from vault core synthesis |
| III Aims / claim boundary | Drafted from dissertation end-goal note |
| IV Methods | Migrated from the candidate-screen report |
| V Results I (candidate screen) | Migrated; current main evidence |
| VI Results II (specificity) | Empty scaffold — next experiment |
| VII Results III (dynamics) | Empty scaffold — later experiment |
| VIII Discussion | Partial; expand after later results |
| IX Conclusion | Partial; abstract deferred to project end |

## Relation to the short report

`docs/reports/full_candidate_perturbation_scientific_writeup.tex` remains the
compact paper-style snapshot of the candidate screen. New dissertation writing
should happen here.
