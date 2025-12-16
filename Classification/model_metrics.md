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

---

## SVM Tuning Journey (LinearSVC)

**Objective**: Maximize severe detection recall (classes 3 & 4), tolerate precision loss.

**Data & Prep**:
- Stratified subset: 117,509 rows (train 94,007; test 23,502)
- Drop ultra-high-cardinality columns (Description, City, Street, etc.)
- One-hot encode; scale with `StandardScaler(with_mean=False)` in a pipeline
- Stratified train/test split

**Step 1 — Baseline (balanced weights)**
- Config: `class_weight="balanced"`, `C=1.0`, `dual=False`, `max_iter=3000`
- Outcome: Minority recall too low; severe recall notably below final result
- Insight: Automatic balancing insufficient given extreme class 2 dominance

**Step 2 — Aggressive manual class weights**
- Config change: `class_weight={0:0, 1:1, 2:100, 3:300}` (keep `C=1.0`)
- Rationale: Force decision boundary to favor classes 3 and 4
- Final metrics:
	- Recall Sev3: 0.7681
	- Recall Sev4: 0.4286
	- Severe Recall (3+4): 0.7154
	- Macro Recall: 0.3919; Balanced Acc: 0.3919
	- Macro F1: 0.2399; MCC: 0.1690; PR-AUC(class 4): 0.0766
	- MAE: 0.7047; QWK: 0.0901
- Insight: Major gain vs balanced weights; Sev3 strong, Sev4 improved but limited

**Step 3 — Regularization checks**
- Tried varying `C`; kept `C=1.0` for best severe recall stability
- Insight: Weights drove improvements; `C` tweaks did not beat weight gains

**Step 4 — Threshold tuning**
- Not applied (LinearSVC lacks calibrated `predict_proba`; focus moved to trees)

**Conclusion**: Best SVM is weighted LinearSVC above (Severe Recall 0.7154). Trees with SMOTE outperform on severe recall.

---

## Decision Trees Tuning Journey (ID3, CART, Random Forest)

**Objective**: Maximize severe recall via tree models; leverage SMOTE and sensible constraints.

**Data & Prep**:
- Same stratified subset and preprocessing as SVM (drops + one-hot)
- Stratified train/test split

**Phase 1 — Hyperparameters (no SMOTE)**
- Class weights: `{0:0, 1:1, 2:5, 3:10}` (softer than SVM; trees overfit with extreme weights)
- ID3/CART: tune `max_depth ∈ {5,10,15,20}`, `min_samples_leaf ∈ {5,10,20,50}`
- Random Forest: `n_estimators=200`, tune `max_depth`, `min_samples_leaf`
- Best (no SMOTE):
	- ID3: Severe Recall 0.7675 (Sev3 0.8468, Sev4 0.3357)
	- CART: Severe Recall 0.7889 (Sev3 0.8573, Sev4 0.4161)
	- RF: Severe Recall 0.6836 (Sev3 0.7976, Sev4 0.0625)
- Insight: Depth + leaf constraints strongly help rare classes; extreme split/impurity constraints hurt.

**Phase 2 — Harmful constraints (confirmed and reverted)**
- Tried `min_samples_split` and `min_impurity_decrease` increases
- Outcome: Severe recall dropped across models (especially RF Sev4 → 0)
- Action: Reverted to depth + leaf only

**Phase 3 — SMOTE Oversampling (classes 3 & 4)**
- Apply `SMOTE(k_neighbors=5)` on training set; train models on resampled data
- Final SMOTE metrics:
	- ID3 + SMOTE: Severe 0.8321 (Sev3 0.8344, Sev4 0.8196)
	- CART + SMOTE: Severe 0.8412 (Sev3 0.8534, Sev4 0.7750)
	- RF + SMOTE: Severe 0.8822 (Sev3 0.8836, Sev4 0.8750)
- Insight: SMOTE is the single biggest lift; especially dramatic for RF’s Sev4

**Conclusion**: Use **Random Forest + SMOTE** (200 trees, `max_depth=15`, `min_samples_leaf=50`, class weights `{0:0,1:1,2:5,3:10}`) — best severe recall 0.8822.

---

## Initial vs Final (Quick Comparison)

**Decision Trees**
- ID3:
	- Initial (pre-tuning, no SMOTE): Severe≈0.4400 (very low minority recall)
	- Final (SMOTE): Severe=0.8321, Sev3=0.8344, Sev4=0.8196
- CART:
	- Initial (pre-tuning, no SMOTE): Severe≈0.4145 (minority classes under-detected)
	- Final (SMOTE): Severe=0.8412, Sev3=0.8534, Sev4=0.7750
- Random Forest:
	- Initial (pre-tuning, no SMOTE): Severe≈0.2463 (Sev4 near 0)
	- Final (SMOTE): Severe=0.8822, Sev3=0.8836, Sev4=0.8750

**SVM (LinearSVC)**
- Initial (balanced weights): Severe=0.2289, Sev3=0.2499, Sev4=0.1143, Macro Recall=0.3792, Balanced Acc=0.3792
- Final (weights {0:0,1:1,2:100,3:300}, C=1.0): Severe=0.7154, Sev3=0.7681, Sev4=0.4286
