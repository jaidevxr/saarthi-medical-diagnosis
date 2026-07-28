"""
Comprehensive 168-Disease Benchmark Test Suite
===============================================
Generates test cases across ALL 168 diseases in the dataset (10 samples per disease = 1,680 total test cases).
Evaluates complete prediction pipeline across the full 168 disease taxonomy.
"""

import os, sys, random, joblib, time
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.prepare_dataset import DISEASE_SYMPTOMS
from scripts.test_clinical_interaction_ensemble import rule_model, CLINICAL_EQUIVALENCE
from utils.helpers import parse_symptoms_with_metadata

MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))

all_diseases = list(DISEASE_SYMPTOMS.keys())

def generate_all_disease_test_cases(samples_per_disease=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    test_cases = []
    tc_id = 1

    for disease, symptoms in DISEASE_SYMPTOMS.items():
        clean_symptoms = [s.replace("_", " ") for s in symptoms]
        if not clean_symptoms:
            continue

        for i in range(samples_per_disease):
            # Select 3 to 7 random symptoms for this disease
            k = random.randint(min(3, len(clean_symptoms)), min(7, len(clean_symptoms)))
            selected = random.sample(clean_symptoms, k)

            # Add occasional typos / layman terms
            phrases = list(selected)
            if random.random() < 0.2:
                phrases.append(random.choice(["bukhar", "sir dard", "pet dard", "fever", "headache"]))

            random.shuffle(phrases)
            symptom_text = ", ".join(phrases)

            test_cases.append({
                "id": tc_id,
                "expected_disease": disease,
                "symptoms": selected,
                "symptom_text": symptom_text
            })
            tc_id += 1

    return test_cases

def evaluate_all_diseases():
    test_cases = generate_all_disease_test_cases(samples_per_disease=10, seed=42)
    total = len(test_cases)

    top1_c, top2_c, top3_c, top5_c = 0, 0, 0, 0
    times = []

    y_true = []
    y_pred_top1 = []

    for tc in test_cases:
        t0 = time.perf_counter()
        top_d, top_p, detected = rule_model.predict_single_enhanced(tc["symptom_text"])
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        exp = tc["expected_disease"]
        exp_canonical = CLINICAL_EQUIVALENCE.get(exp, exp)
        y_true.append(exp_canonical)

        p1 = top_d[0] if top_d else "NONE"
        y_pred_top1.append(p1)

        t1 = (exp_canonical == p1) or (p1 in [exp, exp_canonical])
        t2 = (exp_canonical in top_d[:2]) or (exp in top_d[:2])
        t3 = (exp_canonical in top_d[:3]) or (exp in top_d[:3])
        t5 = (exp_canonical in top_d[:5]) or (exp in top_d[:5])

        if t1: top1_c += 1
        if t2: top2_c += 1
        if t3: top3_c += 1
        if t5: top5_c += 1

    t1_acc = top1_c / total * 100
    t2_acc = top2_c / total * 100
    t3_acc = top3_c / total * 100
    t5_acc = top5_c / total * 100
    avg_time = np.mean(times)

    print("=" * 80)
    print(f"FULL 168-DISEASE EVALUATION RESULTS ({total} TEST CASES)")
    print("=" * 80)
    print(f"  Total Diseases Tested: {len(all_diseases)}")
    print(f"  Total Test Cases:      {total}")
    print(f"  Top-1 Accuracy:        {t1_acc:.2f}% ({top1_c}/{total})")
    print(f"  Top-2 Accuracy:        {t2_acc:.2f}% ({top2_c}/{total})  <-- Target: >= 95%")
    print(f"  Top-3 Accuracy:        {t3_acc:.2f}% ({top3_c}/{total})")
    print(f"  Top-5 Accuracy:        {t5_acc:.2f}% ({top5_c}/{total})")
    print(f"  Avg Prediction Time:   {avg_time:.2f} ms")
    print("=" * 80)

    # Save 168-disease benchmark summary
    res_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_diseases": len(all_diseases),
        "total_test_cases": total,
        "top1_accuracy": t1_acc,
        "top2_accuracy": t2_acc,
        "top3_accuracy": t3_acc,
        "top5_accuracy": t5_acc,
        "avg_prediction_time_ms": float(avg_time)
    }

    out_file = os.path.join(REPORTS_DIR, "full_168_disease_benchmark.json")
    with open(out_file, "w") as f:
        import json
        json.dump(res_summary, f, indent=2)

    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    evaluate_all_diseases()
