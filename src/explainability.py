"""
Explainability & Output Formatting Module (src/explainability.py)
================================================================
Generates structured JSON explanations tied to prediction symptoms,
typical disease symptoms, confidence tiers, alternative diagnoses, and uncertainty threshold refusal.
"""

import numpy as np

def generate_prediction_explanation(
    primary_disease,
    confidence,
    top_predictions,
    matched_symptoms,
    typical_symptoms_map,
    confidence_threshold=0.35
):
    """
    Generates structured, programmatically tied prediction explanation.
    
    Args:
        primary_disease (str): Name of primary predicted disease
        confidence (float): Probability score (0.0 to 1.0)
        top_predictions (list of dict): list of {"disease": name, "confidence": float}
        matched_symptoms (list of str): List of user-matched symptoms
        typical_symptoms_map (dict): Mapping of disease -> list of typical symptoms
        confidence_threshold (float): Threshold below which system refuses diagnosis
        
    Returns:
        dict matching structured schema specified in Part 5.1
    """
    # Determine confidence tier
    if confidence >= 0.70:
        conf_tier = "High"
    elif confidence >= 0.40:
        conf_tier = "Moderate"
    else:
        conf_tier = "Low"

    # Determine typical vs missing symptoms
    typical = typical_symptoms_map.get(primary_disease, [])
    matched_set = set(matched_symptoms)
    typical_set = set(typical)
    
    missing_typical = sorted(list(typical_set - matched_set))
    matched_in_typical = sorted(list(matched_set.intersection(typical_set)))

    # Refusal handling if below threshold (Part 5.2)
    refused = confidence < confidence_threshold

    if refused:
        explanation_msg = (
            f"Confidence score ({confidence*100:.1f}%) is below the uncertainty threshold "
            f"({confidence_threshold*100:.1f}%). System abstains from committing to a primary diagnosis."
        )
        recommendation_msg = (
            "Please describe additional symptoms or consult a medical professional for clinical diagnosis."
        )
    else:
        if missing_typical:
            explanation_msg = (
                f"Matched {len(matched_in_typical)} symptoms for {primary_disease}. "
                f"Confidence reduced because {len(missing_typical)} typical symptoms "
                f"({', '.join(missing_typical[:3])}) were not mentioned."
            )
        else:
            explanation_msg = (
                f"High match with typical presentation for {primary_disease}."
            )
        recommendation_msg = "Consider mentioning any additional symptoms to refine the differential diagnosis."

    # Alternatives (excluding top 1)
    alternatives = [
        {"disease": item["disease"], "confidence": round(float(item["confidence"]), 4)}
        for item in top_predictions[1:4]
    ]

    return {
        "primary_diagnosis": "Uncertain / Additional Info Required" if refused else primary_disease,
        "confidence": round(float(confidence), 4),
        "confidence_tier": conf_tier,
        "refused": refused,
        "matched_symptoms": matched_symptoms,
        "missing_typical_symptoms": missing_typical,
        "explanation": explanation_msg,
        "alternative_diagnoses": alternatives,
        "recommendation": recommendation_msg
    }
