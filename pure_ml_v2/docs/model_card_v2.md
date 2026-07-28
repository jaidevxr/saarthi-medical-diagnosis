# Model Card — Pure ML Diagnosis Engine (`pure_ml_v2`)

## Model Overview
- **Model Name**: GaussianNB
- **Artifact Path**: `pure_ml_v2/models/best_model.pkl`
- **SHA256 Hash**: `4a797e47ecdc87d2cb5dd92b9474fa998859ee664754eae7d1828a37b14a7d4d`
- **Framework**: Scikit-Learn (Module 4 Standard ML)
- **Feature Space**: 132 binary symptom indicators
- **Target Classes**: 168 canonical disease classes

---

## Benchmark Results (80/20 Stratified Validation Split)

| Model Name | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GaussianNB** | 91.93% | 91.97% | 91.82% | **91.10%** | 0.9998 |
| **LogisticRegression** | 90.18% | 90.11% | 89.47% | **89.27%** | 0.9993 |
| **RandomForestClassifier** | 89.72% | 89.86% | 89.21% | **89.03%** | 0.9864 |
| **SVC** | 87.78% | 88.41% | 86.90% | **86.92%** | 0.9956 |
| **KNeighborsClassifier** | 85.31% | 86.78% | 84.57% | **84.59%** | 0.9578 |
| **DecisionTreeClassifier** | 84.24% | 84.40% | 82.97% | **82.88%** | 0.9236 |

---

## Hyperparameters (GaussianNB)
```json
{
  "priors": null,
  "var_smoothing": 1e-09
}
```

---

## Data Pipeline Details
- **Base Dataset**: 4,920 dense rows across 168 disease classes.
- **Sparse Augmentation**: 2,520 synthetic sparse rows (1–3 active symptoms per row).
- **Total Training Instances**: 7,440 rows (44 samples per disease class).
