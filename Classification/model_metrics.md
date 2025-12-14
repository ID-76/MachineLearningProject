# Model Metrics (Current Runs)

## Overall Ranking (Severe Recall)
- Random Forest (SMOTE): **0.8822**
- CART (SMOTE): **0.8412**
- ID3 (SMOTE): **0.8321**
- Linear SVM (weights 0/1/100/300): **0.7154**

---

## Decision Trees (with and without SMOTE)

| Model | Severe Recall | Recall Sev3 | Recall Sev4 | Macro Recall | Balanced Acc |
| --- | --- | --- | --- | --- | --- |
| ID3 Baseline | 0.7675 | 0.8468 | 0.3357 | 0.4215 | 0.4215 |
| ID3 + SMOTE | **0.8321** | 0.8344 | 0.8196 | 0.4447 | 0.4447 |
| CART Baseline | 0.7889 | 0.8573 | 0.4161 | 0.4488 | 0.4488 |
| CART + SMOTE | **0.8412** | 0.8534 | 0.7750 | 0.4694 | 0.4694 |
| RF Baseline | 0.6836 | 0.7976 | 0.0625 | 0.4600 | 0.4600 |
| RF + SMOTE | **0.8822** | 0.8836 | 0.8750 | 0.4698 | 0.4698 |

Notes:
- Class weights for trees: {0:0, 1:1, 2:5, 3:10}.
- RF uses 200 trees, max_depth=15, min_samples_leaf=50.
- ID3/CART use max_depth=10, min_samples_leaf=20.
- SMOTE applied to oversample classes 3 and 4.

---

## SVM (LinearSVC, final config)

- Class weights: {0:0, 1:1, 2:100, 3:300}
- C=1.0, dual=False, max_iter=3000, with scaling

| Metric | Value |
| --- | --- |
| Recall Sev3 | 0.7681 |
| Recall Sev4 | 0.4286 |
| Severe Recall (3+4) | **0.7154** |
| Macro Recall | 0.3919 |
| Balanced Accuracy | 0.3919 |
| Macro F1 | 0.2399 |
| MCC | 0.1690 |
| PR-AUC (class 4) | 0.0766 |
| MAE (severity) | 0.7047 |
| Quadratic Weighted Kappa | 0.0901 |

Confusion Matrix (rows=true labels, cols=predicted):
```
[[   0   97  192   24]
 [   0 7262 9722 2596]
 [   0  387 2342  320]
 [   0   68  252  240]]
```

---

## Recommendation
- Use **Random Forest + SMOTE** as the top performer for severe recall (0.8822) with balanced Sev3/Sev4 detection.
