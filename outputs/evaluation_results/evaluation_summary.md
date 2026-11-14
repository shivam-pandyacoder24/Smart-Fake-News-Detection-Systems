# Model evaluation summary

**Selected model:** Logistic Regression

All metrics below are measured on a held-out, stratified test set.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9921 | 0.9923 | 0.9919 | 0.9921 |
| Multinomial Naive Bayes | 0.9605 | 0.9603 | 0.9601 | 0.9602 |

## Confusion matrix label order
Rows are actual labels; columns are predictions. Order: Fake (0), Real (1).

- **Logistic Regression:** `[[3538, 42], [19, 4167]]`
- **Multinomial Naive Bayes:** `[[3421, 159], [148, 4038]]`