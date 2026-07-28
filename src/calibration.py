"""
Calibration & Uncertainty Module (src/calibration.py)
======================================================
1. Manually implements Expected Calibration Error (ECE).
2. Computes Brier score loss.
3. Optimizes confidence threshold for uncertainty refusal path using VALIDATION DATA ONLY.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calculate_ece(probs, y_true_onehot, n_bins=10):
    """
    Manually implements Expected Calibration Error (ECE).
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Returns:
        ece_score: float (0.0 = perfect calibration)
        bin_accuracies: list of float
        bin_confidences: list of float
        bin_counts: list of int
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_true_onehot, axis=1)
    accuracies = (predictions == true_labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(accuracy_in_bin - confidence_in_bin)

            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(confidence_in_bin))
            bin_counts.append(int(bin_size))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    return float(ece), bin_accs, bin_confs, bin_counts


def optimize_confidence_threshold(val_probs, val_y_true, thresholds=None):
    """
    Optimizes confidence threshold tau for refusal path using VALIDATION DATA ONLY.
    
    Max F1 score between:
    - Confidently Correct: confidence >= tau and prediction == true_label
    - Refused / Abstains: confidence < tau
    
    Returns:
        best_tau: float
        best_f1: float
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    confidences = np.max(val_probs, axis=1)
    predictions = np.argmax(val_probs, axis=1)
    correct = (predictions == val_y_true)

    best_tau = 0.5
    best_score = -1.0

    for tau in thresholds:
        accepted = (confidences >= tau)
        refused = (confidences < tau)

        # TP: accepted and correct
        tp = np.sum(accepted & correct)
        # FP: accepted but incorrect
        fp = np.sum(accepted & ~correct)
        # FN: refused but would have been correct
        fn = np.sum(refused & correct)

        if (2 * tp + fp + fn) > 0:
            f1 = (2 * tp) / (2 * tp + fp + fn)
        else:
            f1 = 0.0

        if f1 > best_score:
            best_score = f1
            best_tau = float(tau)

    return best_tau, float(best_score)
