# External validation

The frozen shipped model (logistic regression, threshold 0.311) scored on the
other three UCI sites. None of these rows were used for training, tuning or the threshold.

| Site | n | Disease | Recall | Precision | Specificity | ROC-AUC | Missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hungarian | 294 | 106 (36%) | 0.736 | 0.765 | 0.872 | 0.886 | 28 |
| Swiss | 123 | 115 (93%) | 0.800 | 0.979 | 0.750 | 0.733 | 23 |
| VA Long Beach | 200 | 149 (74%) | 0.604 | 0.841 | 0.667 | 0.682 | 59 |
| All three pooled | 617 | 370 (60%) | 0.703 | 0.858 | 0.826 | 0.827 | 110 |

## How to read this

- **Prevalence differs a lot by site** (Swiss and VA are mostly disease-positive), so precision and
  specificity are not comparable with the Cleveland test set. Recall and ROC-AUC travel better.
- **Heavy missingness.** Share of rows with the value missing (or an impossible zero for `chol` and
  `trestbps`). The pipeline fills these with Cleveland's most frequent value (`fbs` and `exang` gaps are
  filled with 0 outside the pipeline, because binary columns are passthrough). That pulls scores
  toward the typical Cleveland patient, and for `ca` and `thal` it can hide exactly the signal the
  model relies on. Treat these as a stress test, not a benchmark.

| Site | ca | thal | slope | chol | fbs | trestbps |
| --- | --- | --- | --- | --- | --- | --- |
| Hungarian | 99% | 90% | 65% | 8% | 3% | 0% |
| Swiss | 96% | 42% | 14% | 100% | 61% | 2% |
| VA Long Beach | 99% | 83% | 51% | 28% | 4% | 28% |
| All three pooled | 98% | 78% | 50% | 33% | 15% | 10% |

- Different clinics code and measure things differently, and the cohorts are small. No claim of
  clinical generalisation follows from this.
