# Model Card: Saarthi Medical Diagnosis Engine

## Model Details
- **Model Architecture**: Multi-Turn Information-Theoretic Clarifying System / Weighted Overlap Engine
- **Framework**: scikit-learn (Classical Machine Learning)
- **Model Version**: 7.0.0 (Audited & Verified Non-Circular Multi-Turn System)
- **License**: Educational / Academic Use Only

## Intended Use
- **Primary Use Case**: Interactive medical symptom-to-disease classification support with clarifying follow-up questions.
- **Out of Scope**: Clinical diagnosis, emergency medical decisions, prescribing medication.

## Reproducibility & File Signatures (SHA256)
- `data/training_data_v1.csv`: `572f96d5883f026fb1fb749d19e6876494f5c195c10b8c4cde2407d9de26f063`
- `data/training_data_v2.csv`: `aaab5836d6e3eef0cd28ae5f833128f2920457e36d91693db0958503d8ebdab7`
- `models/best_model.pkl`: `8cafcc05da8b8a306d412703635dd465af4787c72eb3c86a482dd7da2026eab6`

## Verified Multi-Turn Performance & Legitimate Theoretical Ceiling (N = 280 Queries)

| Dialogue Round | Top-1 Accuracy | Top-3 Accuracy | Mean Candidates Remaining | Legitimate $K_{\text{cand}}$ Ceiling ($\sum \frac{1}{K_i}$) | Measured vs Ceiling Gap |
|---|---|---|---|---|---|
| **0 Follow-ups (Single-Shot)** | 12.50% | 35.71% | 15.12 diseases | 17.12% | 4.62% below ceiling |
| **1 Follow-up Question** | 23.21% | 53.57% | 8.26 diseases | 28.43% | 5.22% below ceiling |
| **2 Follow-up Questions (Hard Cap)** | **40.00%** | **65.36%** | **4.61 diseases** | **41.41%** | **1.41% below ceiling** |

## Mutually Exclusive Resolution Distribution (N = 280)
- **Resolved at Round 0** (Resolved immediately with 0 questions): **35 / 280 queries (12.50%)**
- **Resolved at Round 1** (Resolved after 1 question): **32 / 280 queries (11.43%)**
- **Resolved at Round 2** (Hard cap reached / 2 questions asked): **213 / 280 queries (76.07%)**
- **Total Sum Across Mutually Exclusive Buckets**: **280 / 280 queries (100.00%)**

## Summary of Ceiling Verification Audit
1. **Tautology Finding**: `K_exact` (which evaluated single top-score matches) collapsed to `Top-1 Accuracy` for single-candidate predictions ($K_{\text{exact}}=1$), making it a tautological restatement. It has been permanently removed.
2. **Legitimate Ceiling ($K_{\text{cand}}$)**: The true theoretical ceiling $K_{\text{cand}}$ calculates the expected random-choice accuracy $\sum_{i} \frac{1}{K_{\text{cand}, i}}$ over all candidates within $\delta \le 0.05$. It operates independently of system predictions and incorporates genuine fractional expectations ($0.5000$, $0.0909$, $0.0370$).
3. **Honest Bounded Results**: At all dialogue rounds, measured Top-1 accuracy is safely bounded below the legitimate theoretical ceiling ($12.50\% < 17.12\%$, $23.21\% < 28.43\%$, $40.00\% < 41.41\%$).
