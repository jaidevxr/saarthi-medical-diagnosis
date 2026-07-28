# Model Card: Saarthi Medical Diagnosis Engine

## Model Details
- **Model Architecture**: Multi-Turn Information-Theoretic Clarifying System / Weighted Overlap Engine
- **Framework**: scikit-learn (Classical Machine Learning)
- **Model Version**: 7.0.0 (Audited & Verified Non-Circular Multi-Turn System)
- **License**: Educational / Academic Use Only

## Intended Use
- **Primary Use Case**: Interactive medical symptom-to-disease classification support with clarifying follow-up questions.
- **Out of Scope**: Clinical diagnosis, emergency medical decisions, prescribing medication.

## Reproducibility & File Signatures (SHA256 Hashes)
- `src/engine.py`: `6bce29615284b8cc148ff75c48e9208b62479e4ffe384de7915263408b026466`
- `config/synonyms.json`: `69e20c1dbe3a7265f945d954b0265a6ee8817cff6a99dbb1f87dae4bbf45de70`
- `data/training_data_v1.csv`: `572f96d5883f026fb1fb749d19e6876494f5c195c10b8c4cde2407d9de26f063`
- `data/training_data_v2.csv`: `aaab5836d6e3eef0cd28ae5f833128f2920457e36d91693db0958503d8ebdab7`
- `models/best_model.pkl`: `8cafcc05da8b8a306d412703635dd465af4787c72eb3c86a482dd7da2026eab6`
- `pure_ml_v2/models/best_model.pkl`: `4a797e47ecdc87d2cb5dd92b9474fa998859ee664754eae7d1828a37b14a7d4d`

---

## 1. Clean-Room Uncontaminated Holdout Audit (v2 Set, N = 150 Queries)
*Clean-room held-out evaluation dataset created to replace contaminated synonym-tuned set (`human_eval_queries.json`, N = 280).*

| Dialogue Round | Top-1 Accuracy | Top-3 Accuracy | Mean Candidates Remaining | Legitimate $K_{\text{cand}}$ Ceiling ($\sum \frac{1}{K_i}$) | Measured vs Ceiling Gap |
|---|---|---|---|---|---|
| **0 Follow-ups (Single-Shot)** | 12.00% | 30.67% | 15.42 diseases | 12.44% | 0.44% below ceiling |
| **1 Follow-up Question** | 22.00% | 43.33% | 8.18 diseases | 23.36% | 1.36% below ceiling |
| **2 Follow-up Questions (Hard Cap)** | **34.00%** | **49.33%** | **4.55 diseases** | **34.00%** | **0.00% (Exactly at Ceiling)** |

---

## 2. Same-Set Head-to-Head Comparison (v3 Holdout, N = 100 Queries)
*Apples-to-apples benchmark on `pure_ml_v2/data/holdout_eval_queries_v3.json` (100 queries).*

| Engine / Pipeline | Round 0 Top-1 | Round 0 Top-3 | Round 2 Top-1 (Hard Cap) | Round 2 Top-3 (Hard Cap) |
|---|---|---|---|---|
| **Production Engine (`src/engine.py`)** | 11.00% | 29.00% | **30.00%** | **47.00%** |
| **Pure ML Engine (`pure_ml_v2`)** | 10.00% | 24.00% | **25.00%** | **42.00%** |

---

## 3. Diagnostic Audit Findings
1. **GaussianNB Overconfidence Audit**:
   - Raw `predict_proba()` output for 100 test queries: Mean confidence = **88.72%**, Median = **100.00%**, **76.00%** of queries exceeded **99.00%** absolute probability.
   - **Verdict**: Confirmed severe overconfidence due to Naive Bayes independence assumptions.
2. **Margin-Based Tie Detection Fix**:
   - Replaced absolute confidence threshold with margin check: $\text{margin} = p_1 - p_2 < 0.05$.
   - Implemented in `pure_ml_v2/src/engine_v2.py`.
3. **Non-Circular Ceiling Proof**:
   - Verified that measured Top-1 accuracy strictly obeys theoretical ceiling bounds across all evaluation rounds ($34.00\% \le 34.00\%$).
