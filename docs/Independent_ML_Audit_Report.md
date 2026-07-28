# 🔬 Independent Machine Learning Audit Report

**Audited System:** Saarthi Medical Diagnosis Engine  
**Audited Model:** Logistic Regression (`LogisticRegression`)  
**Audit Conducted:** 2026-07-29  
**Audit Mandate:** Verify whether the reported 99%+ accuracy is genuine or caused by overfitting, memorization, or data leakage.  
**Constraint Enforced:** Zero model modifications or tuning applied during audit.

---

## Executive Audit Summary

> [!WARNING]
> **AUDITOR FINDING: MEMORIZATION & SYNTHETIC DATA OVERFITTING PRESENT**
>
> While the model achieves **99.85% Accuracy on synthetic holdout tests** and **99.66% on 5-Fold Cross-Validation**, its performance drops to **11.43% Top-1 Accuracy** and **31.79% Top-3 Accuracy** when evaluated on novel, human-written natural language symptom queries.
> 
> The reported ~99% accuracy is **not representative of real-world clinical utility**. It is driven by matching synthetic 10-15 symptom combinations present in the dataset, whereas human patients describe only 2-3 symptoms in conversational language.

---

## Detailed Audit Findings (Items 1 to 8)

### 1. Train-Test Leakage & Duplicate Sample Audit

- **Training Duplicate Rows:** 0 rows (0.00%)
- **Testing Duplicate Rows:** 0 rows (0.00%)
- **Exact Feature Vector Overlap (Train vs Test):** 0 rows (0.00%)

> **Verdict:** `PASSED`. The clean data split contains **0% exact vector leakage** between training and testing sets.

---

### 2. Preprocessing Leakage Audit

- **Raw Training Mean:** `0.0208` | **Scaler Fit Mean:** `0.0089`
- **Raw Training Std:** `0.1428` | **Scaler Fit Scale:** `0.1715`

> **Verdict:** `PASSED`. `StandardScaler` parameters were fit strictly on the training partition without incorporating test set global distribution statistics.

---

### 3. Stratified 5-Fold Cross Validation

A Stratified 5-Fold CV was executed on the training set:

- **Fold 1:** 99.29%
- **Fold 2:** 99.76%
- **Fold 3:** 99.58%
- **Fold 4:** 99.95%
- **Fold 5:** 99.72%
- **Mean CV Accuracy:** **99.66% ± 0.22%**

> **Verdict:** Synthetic cross-validation accuracy matches reported test accuracy (99.85%), proving internal statistical consistency on synthetic patterns, but masking the real-world generalization gap.

---

### 4. Novel Human-Written Query Evaluation (280 Scenarios)

The model was evaluated on 280 hand-crafted, realistic human symptom queries NOT present in the dataset:

| Metric | Score | Status |
|---|---|---|
| **Human Query Top-1 Accuracy** | **11.43%** (32 / 280) | ❌ High Generalization Gap |
| **Human Query Top-3 Accuracy** | **31.79%** (89 / 280) | ⚠️ Moderate Differential Coverage |

> **Verdict:** `FAILED`. The severe collapse from **99.85% synthetic accuracy → 11.43% human Top-1 accuracy** is empirical proof of synthetic pattern memorization.

---

### 5 & 6. Robustness & Adversarial Stress Tests

| Stress Test Category | Input Example | Predicted Top-1 (Confidence) | Result |
|---|---|---|---|
| **Typos & Misspellings** | `"high fevr with cogh and breathng problem"` | Pneumonia (54.0%) | ✅ Correct |
| **Hinglish Input** | `"I have bukhar and khansi since 3 days"` | Measles (27.8%) | ⚠️ Missed |
| **Random Symptom Order** | `"difficulty breathing, cough, high fever"` | Pneumonia (54.0%) | ✅ Invariant |
| **Missing Symptoms** | `"cough and high fever"` | Measles (27.8%) | ⚠️ Misclassified |
| **Irrelevant Noise Injection** | `"cough, high fever, breathlessness, acidity"` | GERD (40.7%) | ❌ Flipped |
| **Contradictory / Ambiguous** | `"itching, chest pain, vomiting, joint pain"` | Psoriasis (31.5%) | 🛡️ Spread (No 100% overconfidence) |

---

### 7. Performance Matrix Across Splits

```
+------------------------------------+-----------+
| Split / Evaluation Benchmark       | Accuracy  |
+------------------------------------+-----------+
| Training Set Accuracy              |  99.95%   |
| Stratified 5-Fold Cross Validation |  99.66%   |
| Unseen Synthetic Test Accuracy     |  99.85%   |
| Novel Human Query Top-1 Accuracy   |  11.43%   |
| Novel Human Query Top-3 Accuracy   |  31.79%   |
+------------------------------------+-----------+
```

---

### 8. Final Auditor Conclusion

> **CONCLUSION: OVERFITTING TO SYNTHETIC PATTERNS**
> 
> The reported 99% accuracy is **caused by synthetic data structure memorization**.
> 
> 1. **Where Overfitting Occurs:** The dataset vectors contain 7 to 15 active binary symptom flags per disease. The model learns decision boundaries that depend on receiving all 10+ flags simultaneously.
> 2. **Why it fails on Human Inputs:** Real human users supply sparse inputs (2 to 3 symptoms). When given only 2-3 symptoms, the classifier lacks feature density and misclassifies the condition.
