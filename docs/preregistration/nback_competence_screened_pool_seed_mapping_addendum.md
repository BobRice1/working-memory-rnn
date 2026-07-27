# N-back screened-pool seed-mapping addendum

## Registration status

Frozen on 2026-07-27 after the competence-screened pool registration and
before training any pool checkpoint. No perturbation outcome has been run or
inspected.

## Issue found during implementation audit

The existing evaluator constructs a condition-bank seed by adding the
checkpoint training seed, a bank offset, and a condition index (`0` or `1`).
With consecutive checkpoint seeds, this makes one checkpoint's 2-back bank
share a numeric RNG seed with the next checkpoint's 0-back bank:

```text
checkpoint k, 2-back: offset + k + 1
checkpoint k+1, 0-back: offset + k + 1
```

The task rules make the resulting batches non-identical, but reusing a random
stream across checkpoint-by-condition cells is unnecessary and weakens the
intended independence of the frozen evaluation banks.

## Frozen correction

For this screened pool only, construct each validation and untouched
competence base seed as:

```text
bank seed = bank offset + 2 * checkpoint training seed
```

The evaluator continues to add condition index `0` for 0-back and `1` for
2-back. Therefore checkpoint `k` uses `offset + 2k` and `offset + 2k + 1`;
checkpoint `k+1` begins at `offset + 2k + 2`. Every
checkpoint-by-condition cell has a distinct numeric seed.

Encode the multiplier as `checkpoint_seed_stride: 2` in both the validation
and evaluation sections of the screened-pool configuration. Existing
configurations omit the field and retain the historical stride of one.

## Scope and firewall

This addendum changes only deterministic RNG-bank addressing. It does not
change the candidate pool, task distribution, sequence count, model,
training, metrics, thresholds, or first-ten-competent selection rule. It was
made from a code-structure audit before observing any screened-pool baseline
or perturbation result.
