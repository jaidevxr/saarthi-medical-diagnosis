# 🩺 Post-Deployment Model Monitoring & Alerting Plan

**Platform:** Saarthi AI Medical Diagnosis Assistant  
**Document Version:** 1.0  
**Status:** Active Clinical Decision Support Protocol  

---

## 📌 Overview

This document defines the post-deployment monitoring, query logging, and alerting strategy for the Saarthi Medical Diagnosis platform. Medical AI systems require continuous observation to detect data drift, phrasal shifts in user symptom descriptions, and confidence degradation over time.

---

## 1. Production Query & Prediction Logging

All inference requests executed through `DiseasePredictor` (`models/predictor.py`) and the Streamlit interface (`app/app.py`) record anonymized telemetry records in `reports/prediction_history.json`:

```json
{
  "timestamp": "2026-07-28 23:30:00",
  "symptoms_input": ["breathlessness", "chest_pain", "cough", "high_fever"],
  "top_prediction": "Pneumonia",
  "confidence_score": 88.42,
  "top_3_differential": [
    {"disease": "Pneumonia", "confidence": 88.42},
    {"disease": "Tuberculosis", "confidence": 7.15},
    {"disease": "Whooping Cough", "confidence": 2.10}
  ],
  "flagged_high_severity": true
}
```

### Privacy & Anonymization
- **Zero PII:** No patient names, IP addresses, email addresses, or phone numbers are recorded.
- **Text Normalization:** Free-text descriptions are stripped of proper nouns before logging.

---

## 2. Real-Time Telemetry & Alerting Thresholds

The following metrics are tracked continuously across production prediction traffic:

| Telemetry Metric | Normal Baseline Range | Alert Trigger Threshold | Action Required |
|---|---|---|---|
| **Average Prediction Confidence** | 65% – 92% | `< 50%` or `> 99.5%` | Investigate uncalibrated probabilities or out-of-distribution symptom inputs. |
| **Uncertainty Rate (Top-1 Conf < 40%)** | 5% – 12% | `> 25%` | Review new colloquial symptom phrases not recognized by parser. |
| **High-Severity Class Frequency** | 15% – 25% | `> 50%` spike | Audit potential symptom keyword over-triggering for emergency conditions. |
| **Unmatched Token Ratio** | `< 10%` | `> 25%` | Update `utils/helpers.py` Hinglish and fuzzy misspelling dictionary. |

---

## 3. Recurring Evaluation Cadence

1. **Quarterly Audit Execution:**
   - Execute `python scripts/evaluate_all.py` against the version-controlled `data/human_eval_queries.json` benchmark set.
   - Requirement: Top-1 Accuracy $\ge 60\%$, Top-3 Accuracy $\ge 85\%$, High-Severity FNR $\le 5.0\%$, Brier Score $\le 0.0500$.

2. **Continuous Human Eval Dataset Growth:**
   - Sample 50 anonymized production queries monthly (reviewed by clinical domain leads).
   - Add new verified queries to `data/human_eval_queries.json` to prevent benchmark stagnation.

3. **Model Retraining Trigger:**
   - Retrain model pipeline (`scripts/train_optimized_pipeline.py`) whenever dataset taxonomy is updated or quarterly audit metrics fall below threshold.

---

## 4. Governance & Ownership

- **Lead Maintainers:** AI/ML Systems Engineering & Clinical Validation Lead (@KhushiSharma006 & @jaidevxr)
- **Emergency Rollback:** Revert `models/best_model.pkl` to `models/naive_bayes.pkl` baseline via version control tag.
