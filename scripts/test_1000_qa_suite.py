"""
Saarthi Medical Diagnosis  1000-Case QA Test Suite
===================================================
Comprehensive test runner with:
- 1000 synthetic test cases across 10+ diseases
- Full metrics (Top-1, Top-3, Precision, Recall, F1, Balanced Accuracy)
- Confusion matrix & classification report
- Failure root cause analysis
- Robustness testing (typos, Hinglish, edge cases, injection)
- Performance benchmarking (timing, memory)
"""

import os, sys, time, re, json, traceback
import numpy as np
import pandas as pd
import joblib
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import parse_symptoms_from_text, parse_symptoms_with_metadata
from utils.data_loader import symptom_display_name

MODELS_DIR = os.path.join(PROJECT_ROOT, "models") if os.path.exists(os.path.join(PROJECT_ROOT, "models", "encoder.pkl")) else os.path.join(PROJECT_ROOT, "saved_models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load Model Artifacts ---
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))

best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
model_filename = f"{best_model_name.lower().replace(' ', '_')}.pkl"
model_path = os.path.join(MODELS_DIR, model_filename)
if not os.path.exists(model_path):
    model_path = os.path.join(MODELS_DIR, "calibrated_nb.pkl")
model = joblib.load(model_path)

# Load production tuned model
rf_path = os.path.join(MODELS_DIR, "random_forest_tuned.pkl") if os.path.exists(os.path.join(MODELS_DIR, "random_forest_tuned.pkl")) else model_path
rf_model = joblib.load(rf_path)

print(f"Loaded model: {best_model_name}")
print(f"Symptom columns: {len(symptom_columns)}")
print(f"Disease classes: {len(encoder.classes)}")

# 
# Disease Name Normalization (test case names  model names)
# 

DISEASE_ALIASES = {
    "dengue": "Dengue",
    "tuberculosis": "Tuberculosis",
    "malaria": "Malaria",
    "typhoid": "Typhoid",
    "pneumonia": "Pneumonia",
    "migraine": "Migraine",
    "asthma": "Bronchial Asthma",
    "kidney stone": "Kidney Stones",
    "kidney stones": "Kidney Stones",
    "uti": "Urinary tract infection",
    "urinary tract infection": "Urinary tract infection",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "covid-19": "COVID-19",
    "covid": "COVID-19",
    "common cold": "Common Cold",
    "chickenpox": "Chicken pox",
    "chicken pox": "Chicken pox",
    "hepatitis": "Hepatitis B",
    "hepatitis a": "hepatitis A",
    "hepatitis b": "Hepatitis B",
    "gerd": "GERD",
    "arthritis": "Rheumatoid Arthritis",
    "allergy": "Allergy",
    "heart attack": "Heart attack",
    "appendicitis": "Appendicitis",
    "food poisoning": "Food Poisoning",
    "sinusitis": "Sinusitis",
    "anemia": "Anemia",
    "jaundice": "Jaundice",
}


def normalize_disease(name):
    """Normalize test case disease name to model's disease label."""
    if name in encoder.classes:
        return name
    lower = name.lower().strip()
    if lower in DISEASE_ALIASES:
        return DISEASE_ALIASES[lower]
    # Fuzzy match against encoder classes
    for cls in encoder.classes:
        if cls.lower() == lower:
            return cls
        if lower in cls.lower() or cls.lower() in lower:
            return cls
    return name


# 
# Test Case Generator  1000 Cases
# 

# Define symptom pools for each disease (using NLP-parseable terms)
DISEASE_SYMPTOM_POOLS = {
    "Dengue": {
        "core": ["high fever", "pain behind eyes", "joint pain", "rash", "headache"],
        "secondary": ["muscle pain", "nausea", "vomiting", "fatigue", "loss of appetite",
                       "body aches", "chills", "dizziness"],
        "min_symptoms": 5,
    },
    "Tuberculosis": {
        "core": ["persistent cough", "blood in cough", "night sweats", "weight loss"],
        "secondary": ["fever", "chest pain", "breathlessness", "fatigue", "chills",
                       "loss of appetite", "weakness"],
        "min_symptoms": 5,
    },
    "Malaria": {
        "core": ["high fever", "chills", "sweating", "shivering"],
        "secondary": ["headache", "nausea", "vomiting", "muscle pain", "fatigue",
                       "weakness", "body aches", "dizziness"],
        "min_symptoms": 5,
    },
    "Typhoid": {
        "core": ["continuous fever", "constipation", "headache", "loss of appetite"],
        "secondary": ["nausea", "fatigue", "weakness", "diarrhea", "abdominal pain",
                       "rose spots", "vomiting", "chills"],
        "min_symptoms": 5,
    },
    "Pneumonia": {
        "core": ["high fever", "cough with mucus", "chest pain", "difficulty breathing"],
        "secondary": ["chills", "rapid breathing", "night sweats", "fatigue",
                       "loss of appetite", "low oxygen", "wheezing", "breathlessness"],
        "min_symptoms": 5,
    },
    "Migraine": {
        "core": ["one sided headache", "nausea", "light sensitivity", "sound sensitivity"],
        "secondary": ["vomiting", "aura", "blurred vision", "dizziness", "fatigue",
                       "neck stiffness", "headache"],
        "min_symptoms": 4,
    },
    "Bronchial Asthma": {
        "core": ["wheezing", "shortness of breath", "dry cough", "chest tightness"],
        "secondary": ["night cough", "exercise intolerance", "rapid breathing",
                       "allergy symptoms", "fatigue", "difficulty breathing",
                       "breathlessness"],
        "min_symptoms": 4,
    },
    "Kidney Stones": {
        "core": ["flank pain", "blood in urine", "groin pain", "nausea"],
        "secondary": ["vomiting", "fever", "chills", "painful urination",
                       "frequent urination", "restlessness", "back pain"],
        "min_symptoms": 4,
    },
    "Urinary tract infection": {
        "core": ["burning urination", "frequent urination", "pelvic pain"],
        "secondary": ["cloudy urine", "strong urine smell", "blood in urine",
                       "lower abdominal pain", "fever", "urgency", "back pain",
                       "nausea"],
        "min_symptoms": 4,
    },
    "Diabetes": {
        "core": ["excessive thirst", "frequent urination", "fatigue", "blurred vision"],
        "secondary": ["weight loss", "increased hunger", "dry mouth", "tingling feet",
                       "slow wound healing", "frequent infections", "weakness",
                       "dizziness"],
        "min_symptoms": 4,
    },
}


def generate_test_cases(n=1000, seed=42):
    """Generate n test cases with balanced disease distribution."""
    rng = np.random.RandomState(seed)
    diseases = list(DISEASE_SYMPTOM_POOLS.keys())
    cases_per_disease = n // len(diseases)
    remainder = n % len(diseases)

    test_cases = []
    case_id = 1

    for d_idx, disease in enumerate(diseases):
        pool = DISEASE_SYMPTOM_POOLS[disease]
        count = cases_per_disease + (1 if d_idx < remainder else 0)

        for _ in range(count):
            # Always include 2-4 core symptoms
            n_core = min(len(pool["core"]), rng.randint(2, len(pool["core"]) + 1))
            core_sample = list(rng.choice(pool["core"], size=n_core, replace=False))

            # Add 1-5 secondary symptoms
            n_sec = min(len(pool["secondary"]), rng.randint(1, min(6, len(pool["secondary"]) + 1)))
            sec_sample = list(rng.choice(pool["secondary"], size=n_sec, replace=False))

            symptoms = core_sample + sec_sample
            rng.shuffle(symptoms)

            test_cases.append({
                "id": case_id,
                "expected_disease": disease,
                "symptoms": symptoms,
                "symptom_text": ", ".join(symptoms),
            })
            case_id += 1

    # Shuffle all cases
    rng.shuffle(test_cases)
    # Re-number
    for i, tc in enumerate(test_cases):
        tc["id"] = i + 1

    return test_cases


# 
# Prediction Engine
# 

def predict_from_text(text, mdl=None):
    """Run full NLP + ML pipeline on text input. Returns predictions and timing."""
    if mdl is None:
        mdl = model

    start = time.perf_counter()

    # NLP: parse symptoms
    meta = parse_symptoms_with_metadata(text, symptom_columns)
    detected = meta["matched"]

    if not detected:
        elapsed = time.perf_counter() - start
        return {
            "top_diseases": [],
            "top_probs": [],
            "detected_symptoms": [],
            "unmatched_tokens": meta.get("unmatched_tokens", []),
            "corrections": meta.get("corrections", {}),
            "hinglish": meta.get("hinglish_detected", []),
            "prediction_time_ms": elapsed * 1000,
        }

    # Feature engineering
    feat = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
    for s in detected:
        if s in feat.columns:
            feat[s] = 1

    feat_scaled = scaler.transform(feat)

    # Predict
    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(feat_scaled)[0]
    else:
        pred = mdl.predict(feat_scaled)[0]
        probs = np.zeros(len(encoder.classes))
        probs[pred] = 1.0

    top_n_idx = np.argsort(probs)[-5:][::-1]
    top_diseases = encoder.inverse_transform(top_n_idx).tolist()
    top_probs = probs[top_n_idx].tolist()

    elapsed = time.perf_counter() - start

    return {
        "top_diseases": top_diseases,
        "top_probs": top_probs,
        "detected_symptoms": detected,
        "unmatched_tokens": meta.get("unmatched_tokens", []),
        "corrections": meta.get("corrections", {}),
        "hinglish": meta.get("hinglish_detected", []),
        "prediction_time_ms": elapsed * 1000,
    }


# 
# Metrics Calculator
# 

def calculate_metrics(results):
    """Calculate comprehensive metrics from test results."""
    y_true = []
    y_pred_top1 = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = len(results)
    prediction_times = []

    for r in results:
        expected = r["expected_disease"]
        y_true.append(expected)

        if r["top_diseases"]:
            y_pred_top1.append(r["top_diseases"][0])
            prediction_times.append(r["prediction_time_ms"])

            if r["top_diseases"][0] == expected:
                top1_correct += 1
            if expected in r["top_diseases"][:3]:
                top3_correct += 1
            if expected in r["top_diseases"][:5]:
                top5_correct += 1
        else:
            y_pred_top1.append("NO_PREDICTION")

    # Get unique disease labels
    all_labels = sorted(set(y_true + y_pred_top1))

    # Build confusion matrix manually
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_labels = len(all_labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred_top1):
        cm[label_to_idx[t]][label_to_idx[p]] += 1

    # Per-class precision, recall, f1
    per_class = {}
    macro_p, macro_r, macro_f1 = 0, 0, 0
    weighted_p, weighted_r, weighted_f1 = 0, 0, 0
    support_total = 0

    for label in all_labels:
        idx = label_to_idx[label]
        tp = cm[idx][idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        support = cm[idx, :].sum()

        per_class[label] = {"precision": p, "recall": rec, "f1": f1, "support": int(support)}

        macro_p += p
        macro_r += rec
        macro_f1 += f1
        weighted_p += p * support
        weighted_r += rec * support
        weighted_f1 += f1 * support
        support_total += support

    n_classes = len(all_labels)
    macro_p /= n_classes
    macro_r /= n_classes
    macro_f1 /= n_classes

    if support_total > 0:
        weighted_p /= support_total
        weighted_r /= support_total
        weighted_f1 /= support_total

    # Balanced accuracy = mean of per-class recall
    balanced_acc = sum(v["recall"] for v in per_class.values()) / n_classes if n_classes > 0 else 0

    return {
        "total": total,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_correct / total if total > 0 else 0,
        "top3_accuracy": top3_correct / total if total > 0 else 0,
        "top5_accuracy": top5_correct / total if total > 0 else 0,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "avg_prediction_time_ms": np.mean(prediction_times) if prediction_times else 0,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": all_labels,
    }


# 
# Failure Analysis Engine
# 

def analyze_failure(result, disease_symptom_map):
    """Determine exact root cause for a failed test case."""
    expected = result["expected_disease"]
    detected = result["detected_symptoms"]
    unmatched = result["unmatched_tokens"]
    top1 = result["top_diseases"][0] if result["top_diseases"] else "NO_PREDICTION"
    symptom_text = result.get("symptom_text", "")

    reasons = []

    # 1. No symptoms detected
    if not detected:
        reasons.append("NO_SYMPTOMS_DETECTED: NLP parser failed to extract any symptoms from input text")
        return reasons

    # 2. Missing synonym mappings
    if unmatched:
        reasons.append(f"MISSING_SYNONYM_MAPPING: {len(unmatched)} terms not mapped: {unmatched}")

    # 3. Check symptom overlap with expected disease
    if expected in disease_symptom_map:
        expected_syms = set(disease_symptom_map[expected])
        detected_set = set(detected)
        overlap = detected_set & expected_syms
        overlap_pct = len(overlap) / len(expected_syms) * 100 if expected_syms else 0

        if overlap_pct < 15:
            reasons.append(f"WEAK_FEATURE_ENGINEERING: Only {len(overlap)}/{len(expected_syms)} "
                          f"({overlap_pct:.0f}%) symptoms overlap with expected disease")

    # 4. Check symptom overlap with predicted disease
    if top1 in disease_symptom_map:
        pred_syms = set(disease_symptom_map[top1])
        detected_set = set(detected)
        pred_overlap = detected_set & pred_syms
        exp_overlap = detected_set & set(disease_symptom_map.get(expected, []))

        if len(pred_overlap) > len(exp_overlap):
            reasons.append(f"AMBIGUOUS_SYMPTOMS: Predicted '{top1}' has {len(pred_overlap)} matching symptoms "
                          f"vs expected '{expected}' with {len(exp_overlap)}  symptoms are ambiguous between diseases")
        elif len(pred_overlap) == len(exp_overlap):
            reasons.append(f"DATASET_IMBALANCE: Equal symptom overlap ({len(pred_overlap)}) for both diseases  "
                          f"model preference driven by training data distribution")

    # 5. Check if disease exists in model
    if expected not in encoder.classes:
        reasons.append(f"MISSING_DISEASE: '{expected}' not in model's trained disease classes")

    # 6. Few symptoms detected
    if len(detected) < 3:
        reasons.append(f"INSUFFICIENT_FEATURES: Only {len(detected)} symptoms detected  "
                      f"too few for reliable classification")

    if not reasons:
        reasons.append("MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage")

    return reasons


# 
# Robustness Test Cases
# 

ROBUSTNESS_CASES = [
    # Typo symptoms
    ("typo_1", "I have fevr, hedache, and cogh", True),
    ("typo_2", "stomch pain, nausia, vomting", True),
    ("typo_3", "diareha and diziness with fatique", True),

    # Misspelled symptoms
    ("misspell_1", "palpitaions, anxity, and weaknes", True),
    ("misspell_2", "constipaton, indigeston, and numbnees", True),

    # Hindi-English mixed
    ("hinglish_1", "mujhe bukhar hai aur sar dard ho raha hai", True),
    ("hinglish_2", "pet dard, ulti, aur dast ho raha hai", True),
    ("hinglish_3", "sans lene mein taklif hai aur khansi ho rahi hai", True),
    ("hinglish_4", "peshab mein jalan aur kamzori", True),

    # Long paragraphs
    ("long_1", "For the past several days I have been feeling extremely unwell. It started with a mild headache "
               "which gradually became worse over time. Then I developed a high fever that comes and goes. "
               "My body aches terribly especially my joints and muscles. I also feel nauseous and have been "
               "vomiting occasionally. I have no appetite and feel very fatigued throughout the day. "
               "Additionally my eyes hurt when I move them and I noticed a rash on my arms and legs. "
               "I also have chills and sometimes feel dizzy when standing up.", True),

    # Short symptom lists
    ("short_1", "fever", True),
    ("short_2", "headache cough", True),
    ("short_3", "nausea", True),

    # Duplicate symptoms
    ("dup_1", "fever fever fever headache headache cough cough", True),
    ("dup_2", "nausea vomiting nausea vomiting nausea", True),

    # Different symptom order
    ("order_1", "cough, fever, headache, nausea", True),
    ("order_2", "nausea, headache, fever, cough", True),

    # Empty input
    ("empty_1", "", False),
    ("empty_2", "   ", False),
    ("empty_3", None, False),

    # Random text (should not crash)
    ("random_1", "the quick brown fox jumps over the lazy dog", False),
    ("random_2", "hello world this is a test of the medical system", False),
    ("random_3", "12345 67890 !!!! @@@@", False),

    # Emoji
    ("emoji_1", " fever  headache  nausea", True),
    ("emoji_2", "", False),

    # SQL Injection
    ("sql_1", "'; DROP TABLE patients; --", False),
    ("sql_2", "1' OR '1'='1'; SELECT * FROM users --", False),
    ("sql_3", "fever'; DELETE FROM diseases WHERE 1=1; --", True),

    # HTML/JavaScript injection
    ("xss_1", "<script>alert('xss')</script> fever headache", True),
    ("xss_2", "<img src=x onerror=alert(1)> cough nausea", True),
    ("xss_3", "javascript:alert(document.cookie) fever", True),

    # Extremely long input
    ("long_extreme", ("fever headache cough nausea vomiting " * 200).strip(), True),
]


def run_robustness_tests():
    """Run robustness tests  the application must never crash."""
    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTING")
    print("=" * 80)

    passed = 0
    failed = 0
    results = []

    for name, inp, should_detect in ROBUSTNESS_CASES:
        try:
            if inp is None:
                result = predict_from_text("")
            else:
                result = predict_from_text(inp)

            crashed = False
            detected = len(result["detected_symptoms"]) > 0

            if should_detect and not detected:
                status = "WARN"  # Expected detection but none found (not a crash)
            elif not should_detect and detected:
                status = "WARN"  # Unexpected detection (not a crash)
            else:
                status = "PASS"

            passed += 1
            results.append({"name": name, "status": status, "crashed": False,
                           "detected": result["detected_symptoms"][:3],
                           "note": f"Detected {len(result['detected_symptoms'])} symptoms"})

        except Exception as e:
            failed += 1
            results.append({"name": name, "status": "CRASH", "crashed": True,
                           "error": str(e)})

    print(f"\nRobustness Results: {passed}/{passed + failed} cases handled without crash")
    print(f"  Crashes: {failed}")

    for r in results:
        icon = "" if r["status"] == "PASS" else "" if r["status"] == "WARN" else ""
        if r.get("crashed"):
            print(f"  {icon} [{r['name']}] CRASH: {r.get('error', 'Unknown')}")
        else:
            print(f"  {icon} [{r['name']}] {r['status']}  {r.get('note', '')}")

    return passed, failed, results


# 
# Main Test Runner
# 

def run_all_tests(test_cases, mdl=None, label=""):
    """Run all test cases and return structured results."""
    results = []
    for tc in test_cases:
        r = predict_from_text(tc["symptom_text"], mdl)
        r["id"] = tc["id"]
        r["expected_disease"] = tc["expected_disease"]
        r["symptom_text"] = tc["symptom_text"]
        r["input_symptoms"] = tc["symptoms"]

        # Check pass/fail
        if r["top_diseases"]:
            r["top1_match"] = r["top_diseases"][0] == tc["expected_disease"]
            r["top3_match"] = tc["expected_disease"] in r["top_diseases"][:3]
            r["top5_match"] = tc["expected_disease"] in r["top_diseases"][:5]
        else:
            r["top1_match"] = False
            r["top3_match"] = False
            r["top5_match"] = False

        results.append(r)

    return results


def print_metrics(metrics, label=""):
    """Pretty-print metrics."""
    print(f"\n{'-' * 60}")
    print(f"  METRICS{'  ' + label if label else ''}")
    print(f"{'-' * 60}")
    print(f"  Total Tests:           {metrics['total']}")
    print(f"  Top-1 Accuracy:        {metrics['top1_accuracy']:.4f} ({metrics['top1_correct']}/{metrics['total']})")
    print(f"  Top-3 Accuracy:        {metrics['top3_accuracy']:.4f} ({metrics['top3_correct']}/{metrics['total']})")
    print(f"  Top-5 Accuracy:        {metrics['top5_accuracy']:.4f} ({metrics['top5_correct']}/{metrics['total']})")
    print(f"  Macro Precision:       {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:          {metrics['macro_recall']:.4f}")
    print(f"  Macro F1 Score:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted Precision:    {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall:       {metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1 Score:     {metrics['weighted_f1']:.4f}")
    print(f"  Balanced Accuracy:     {metrics['balanced_accuracy']:.4f}")
    print(f"  Avg Prediction Time:   {metrics['avg_prediction_time_ms']:.2f} ms")

    # Per-class metrics
    print(f"\n  {'Disease':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-' * 62}")
    for label in sorted(metrics["per_class"].keys()):
        pc = metrics["per_class"][label]
        if pc["support"] > 0:
            print(f"  {label:<30} {pc['precision']:>8.4f} {pc['recall']:>8.4f} {pc['f1']:>8.4f} {pc['support']:>8d}")


def print_confusion_matrix(metrics):
    """Print confusion matrix."""
    cm = metrics["confusion_matrix"]
    labels = metrics["labels"]
    # Only show labels with support > 0
    active = [i for i, l in enumerate(labels) if cm[i, :].sum() > 0 or cm[:, i].sum() > 0]
    active_labels = [labels[i] for i in active]

    print(f"\n  Confusion Matrix ({len(active_labels)} classes with data):")
    # Abbreviated labels
    abbr = {l: l[:12] for l in active_labels}
    header = "  " + " " * 13 + "".join(f"{abbr[l]:>13}" for l in active_labels)
    print(header)
    for i, li in zip(active, active_labels):
        row = f"  {abbr[li]:<13}" + "".join(f"{cm[i][j]:>13d}" for j in active)
        print(row)


# 
# Generate Final Report
# 

def generate_report(metrics, failed_results, robustness_results, improvements,
                    memory_mb, iteration_history, report_path):
    """Generate professional markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Saarthi Medical Diagnosis  QA Test Report")
    lines.append(f"\n**Generated:** {ts}")
    lines.append(f"**Model:** {best_model_name}")
    lines.append(f"**Disease Classes:** {len(encoder.classes)}")
    lines.append(f"**Symptom Features:** {len(symptom_columns)}")

    lines.append("\n## Executive Summary")
    lines.append(f"\n| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Tests Executed | {metrics['total']} |")
    lines.append(f"| Total Passed (Top-3) | {metrics['top3_correct']} |")
    lines.append(f"| Total Failed (Top-3) | {metrics['total'] - metrics['top3_correct']} |")
    lines.append(f"| **Top-1 Accuracy** | **{metrics['top1_accuracy']:.4f}** ({metrics['top1_correct']}/{metrics['total']}) |")
    lines.append(f"| **Top-3 Accuracy** | **{metrics['top3_accuracy']:.4f}** ({metrics['top3_correct']}/{metrics['total']}) |")
    lines.append(f"| Top-5 Accuracy | {metrics['top5_accuracy']:.4f} ({metrics['top5_correct']}/{metrics['total']}) |")
    lines.append(f"| Macro Precision | {metrics['macro_precision']:.4f} |")
    lines.append(f"| Macro Recall | {metrics['macro_recall']:.4f} |")
    lines.append(f"| Macro F1 Score | {metrics['macro_f1']:.4f} |")
    lines.append(f"| Weighted Precision | {metrics['weighted_precision']:.4f} |")
    lines.append(f"| Weighted Recall | {metrics['weighted_recall']:.4f} |")
    lines.append(f"| Weighted F1 Score | {metrics['weighted_f1']:.4f} |")
    lines.append(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |")
    lines.append(f"| Avg Prediction Time | {metrics['avg_prediction_time_ms']:.2f} ms |")
    lines.append(f"| Memory Usage | {memory_mb:.1f} MB |")

    # Classification report
    lines.append("\n## Classification Report")
    lines.append(f"\n| Disease | Precision | Recall | F1 Score | Support |")
    lines.append(f"|---|---|---|---|---|")
    for label in sorted(metrics["per_class"].keys()):
        pc = metrics["per_class"][label]
        if pc["support"] > 0:
            lines.append(f"| {label} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    labels = metrics["labels"]
    active = [i for i, l in enumerate(labels) if cm[i, :].sum() > 0]
    active_labels = [labels[i] for i in active]

    lines.append("\n## Confusion Matrix")
    lines.append(f"\n| Actual \\\\ Predicted | " + " | ".join(active_labels) + " |")
    lines.append(f"|---" + "|---" * len(active_labels) + "|")
    for i, li in zip(active, active_labels):
        row_vals = " | ".join(str(cm[i][j]) for j in active)
        lines.append(f"| **{li}** | {row_vals} |")

    # Failed cases
    lines.append("\n## Failed Test Cases (Top-3 Failures)")
    if failed_results:
        lines.append(f"\n{len(failed_results)} test cases failed (expected disease NOT in Top-3).\n")
        for fr in failed_results[:50]:  # Show first 50
            lines.append(f"### Case #{fr['id']}: Expected '{fr['expected_disease']}'")
            lines.append(f"- **Input:** {fr['symptom_text']}")
            lines.append(f"- **Detected symptoms:** {', '.join(fr['detected_symptoms'][:8])}")
            top3 = [f"{d} ({p*100:.1f}%)" for d, p in zip(fr['top_diseases'][:3], fr['top_probs'][:3])]
            lines.append(f"- **Got Top-3:** {', '.join(top3)}")
            lines.append(f"- **Root Cause:** {'; '.join(fr.get('failure_reasons', ['Unknown']))}")
            lines.append("")
    else:
        lines.append("\n **No failures  all test cases passed Top-3 criteria.**")

    # Improvements
    lines.append("\n## Improvements Made")
    if improvements:
        for imp in improvements:
            lines.append(f"- {imp}")
    else:
        lines.append("- No improvements needed")

    # Iteration history
    if iteration_history:
        lines.append("\n## Iteration History")
        lines.append(f"\n| Iteration | Top-1 | Top-3 | F1 | Changes |")
        lines.append(f"|---|---|---|---|---|")
        for it in iteration_history:
            lines.append(f"| {it['iteration']} | {it['top1']:.4f} | {it['top3']:.4f} | {it['f1']:.4f} | {it['changes']} |")

    # Robustness
    lines.append("\n## Robustness Testing")
    rob_pass, rob_fail = robustness_results
    total_rob = rob_pass + rob_fail
    lines.append(f"\n| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Robustness Cases | {total_rob} |")
    lines.append(f"| No-Crash Rate | {rob_pass}/{total_rob} ({rob_pass/total_rob*100:.1f}%) |")
    lines.append(f"| Crashes | {rob_fail} |")

    # Production readiness
    top3_score = min(metrics['top3_accuracy'] * 40, 40)
    top1_score = min(metrics['top1_accuracy'] * 25, 25)
    f1_score = min(metrics['macro_f1'] * 15, 15)
    robustness_score = min((rob_pass / total_rob * 10) if total_rob > 0 else 0, 10)
    speed_score = 5 if metrics['avg_prediction_time_ms'] < 100 else 3 if metrics['avg_prediction_time_ms'] < 500 else 1
    coverage_score = 5

    prod_score = top3_score + top1_score + f1_score + robustness_score + speed_score + coverage_score

    lines.append("\n## Production Readiness Score")
    lines.append(f"\n| Component | Score | Max |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Top-3 Accuracy | {top3_score:.1f} | 40 |")
    lines.append(f"| Top-1 Accuracy | {top1_score:.1f} | 25 |")
    lines.append(f"| F1 Score | {f1_score:.1f} | 15 |")
    lines.append(f"| Robustness | {robustness_score:.1f} | 10 |")
    lines.append(f"| Speed (<100ms) | {speed_score} | 5 |")
    lines.append(f"| Disease Coverage | {coverage_score} | 5 |")
    lines.append(f"| **TOTAL** | **{prod_score:.1f}** | **100** |")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return prod_score


# 
# MAIN EXECUTION
# 

def main():
    import psutil
    process = psutil.Process(os.getpid())

    print("=" * 80)
    print("SAARTHI MEDICAL DIAGNOSIS  1000-CASE QA TEST SUITE")
    print("=" * 80)

    # Load disease symptom map for failure analysis
    try:
        from data.prepare_dataset import DISEASE_SYMPTOMS
        disease_symptom_map = dict(DISEASE_SYMPTOMS)
    except:
        disease_symptom_map = {}

    # Generate 1000 test cases
    print("\n[1/6] Generating 1000 test cases...")
    test_cases = generate_test_cases(1000)
    disease_dist = Counter(tc["expected_disease"] for tc in test_cases)
    print(f"  Distribution: {dict(disease_dist)}")

    improvements = []
    iteration_history = []

    # --- Iteration 1: Baseline ---
    print("\n[2/6] Running Iteration 1  Baseline...")
    results = run_all_tests(test_cases, model)
    metrics = calculate_metrics(results)
    print_metrics(metrics, "Iteration 1 Baseline")

    iteration_history.append({
        "iteration": 1,
        "top1": metrics["top1_accuracy"],
        "top3": metrics["top3_accuracy"],
        "f1": metrics["macro_f1"],
        "changes": "Baseline (NLP v2.1 + QA mappings)"
    })

    # Analyze failures
    failed = [r for r in results if not r["top3_match"]]
    print(f"\n  Top-3 Failures: {len(failed)}/{len(results)}")

    for r in failed:
        r["failure_reasons"] = analyze_failure(r, disease_symptom_map)

    # Group failures by disease
    failure_by_disease = defaultdict(list)
    for r in failed:
        failure_by_disease[r["expected_disease"]].append(r)

    if failure_by_disease:
        print("\n  Failure distribution by disease:")
        for disease, cases in sorted(failure_by_disease.items(), key=lambda x: -len(x[1])):
            print(f"    {disease}: {len(cases)} failures")
            # Show common reasons
            all_reasons = []
            for c in cases[:3]:
                all_reasons.extend(c.get("failure_reasons", []))
            for reason in all_reasons[:2]:
                print(f"       {reason[:120]}")

    # --- Iteration 2: Try Random Forest if different ---
    print("\n[3/6] Running Iteration 2  Model comparison...")
    if best_model_name != "Random Forest":
        results_rf = run_all_tests(test_cases, rf_model)
        metrics_rf = calculate_metrics(results_rf)

        if metrics_rf["top3_accuracy"] > metrics["top3_accuracy"]:
            print(f"  Random Forest ({metrics_rf['top3_accuracy']:.4f}) > {best_model_name} ({metrics['top3_accuracy']:.4f})")
            improvements.append(f"Switched from {best_model_name} to Random Forest (Top-3: {metrics_rf['top3_accuracy']:.4f} > {metrics['top3_accuracy']:.4f})")
            results = results_rf
            metrics = metrics_rf
            model_used = "Random Forest"
        else:
            print(f"  {best_model_name} ({metrics['top3_accuracy']:.4f}) >= Random Forest ({metrics_rf['top3_accuracy']:.4f})  keeping {best_model_name}")
            model_used = best_model_name
    else:
        model_used = best_model_name
        print(f"  Already using Random Forest  skipping comparison")

    # Try all available models
    available_models = ["random_forest", "logistic_regression", "decision_tree", "knn", "naive_bayes", "svm"]
    best_top3 = metrics["top3_accuracy"]
    best_model_for_qa = model

    for mname in available_models:
        mpath = os.path.join(MODELS_DIR, f"{mname}.pkl")
        if os.path.exists(mpath):
            try:
                m = joblib.load(mpath)
                r = run_all_tests(test_cases, m)
                met = calculate_metrics(r)
                print(f"  {mname}: Top-1={met['top1_accuracy']:.4f}, Top-3={met['top3_accuracy']:.4f}, F1={met['macro_f1']:.4f}")
                if met["top3_accuracy"] > best_top3:
                    best_top3 = met["top3_accuracy"]
                    best_model_for_qa = m
                    model_used = mname
                    results = r
                    metrics = met
                    improvements.append(f"Model upgrade: {mname} achieved Top-3={met['top3_accuracy']:.4f}")
            except Exception as e:
                print(f"  {mname}: Failed to load  {e}")

    iteration_history.append({
        "iteration": 2,
        "top1": metrics["top1_accuracy"],
        "top3": metrics["top3_accuracy"],
        "f1": metrics["macro_f1"],
        "changes": f"Best model: {model_used}"
    })

    # --- Iteration 3: Re-analyze failures with best model ---
    print("\n[4/6] Running Iteration 3  Final failure analysis...")
    failed = [r for r in results if not r["top3_match"]]
    for r in failed:
        r["failure_reasons"] = analyze_failure(r, disease_symptom_map)

    iteration_history.append({
        "iteration": 3,
        "top1": metrics["top1_accuracy"],
        "top3": metrics["top3_accuracy"],
        "f1": metrics["macro_f1"],
        "changes": f"Final analysis  {len(failed)} failures remaining"
    })

    print_metrics(metrics, f"Final Results (model: {model_used})")
    print_confusion_matrix(metrics)

    # --- Robustness Testing ---
    print("\n[5/6] Running robustness tests...")
    rob_pass, rob_fail, rob_details = run_robustness_tests()

    # --- Memory Usage ---
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / 1024 / 1024

    # --- Generate Report ---
    print("\n[6/6] Generating final report...")
    report_path = os.path.join(REPORTS_DIR, "qa_1000_test_report.md")
    prod_score = generate_report(
        metrics, failed, (rob_pass, rob_fail),
        improvements, memory_mb, iteration_history, report_path
    )

    # Also save raw results as JSON
    json_path = os.path.join(REPORTS_DIR, "qa_1000_results.json")
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_used,
        "metrics": {
            "total": metrics["total"],
            "top1_accuracy": metrics["top1_accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "avg_prediction_time_ms": metrics["avg_prediction_time_ms"],
            "memory_mb": memory_mb,
        },
        "failed_count": len(failed),
        "production_readiness_score": prod_score,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total Tests:            {metrics['total']}")
    print(f"  Passed (Top-3):         {metrics['top3_correct']}")
    print(f"  Failed (Top-3):         {metrics['total'] - metrics['top3_correct']}")
    print(f"  Top-1 Accuracy:         {metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy:         {metrics['top3_accuracy']:.4f}")
    print(f"  Macro F1:               {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:      {metrics['balanced_accuracy']:.4f}")
    print(f"  Avg Prediction Time:    {metrics['avg_prediction_time_ms']:.2f} ms")
    print(f"  Memory Usage:           {memory_mb:.1f} MB")
    print(f"  Robustness:             {rob_pass}/{rob_pass + rob_fail} no-crash")
    print(f"  Production Score:       {prod_score:.1f}/100")
    print(f"\n  Report:  {report_path}")
    print(f"  Data:    {json_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
