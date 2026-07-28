# Saarthi Medical Diagnosis  QA Test Report

**Generated:** 2026-07-28 10:44:47
**Model:** Naive Bayes
**Disease Classes:** 168
**Symptom Features:** 397

## Executive Summary

| Metric | Value |
|---|---|
| Total Tests Executed | 1000 |
| Total Passed (Top-3) | 804 |
| Total Failed (Top-3) | 196 |
| **Top-1 Accuracy** | **0.5120** (512/1000) |
| **Top-3 Accuracy** | **0.8040** (804/1000) |
| Top-5 Accuracy | 0.8510 (851/1000) |
| Macro Precision | 0.1483 |
| Macro Recall | 0.0898 |
| Macro F1 Score | 0.1061 |
| Weighted Precision | 0.8454 |
| Weighted Recall | 0.5120 |
| Weighted F1 Score | 0.6047 |
| Balanced Accuracy | 0.0898 |
| Avg Prediction Time | 67.12 ms |
| Memory Usage | 613.6 MB |

## Classification Report

| Disease | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Bronchial Asthma | 0.4651 | 0.4000 | 0.4301 | 100 |
| Dengue | 1.0000 | 0.7600 | 0.8636 | 100 |
| Diabetes | 1.0000 | 0.5600 | 0.7179 | 100 |
| Kidney Stones | 1.0000 | 0.5700 | 0.7261 | 100 |
| Malaria | 0.9889 | 0.8900 | 0.9368 | 100 |
| Migraine | 1.0000 | 0.9000 | 0.9474 | 100 |
| Pneumonia | 0.0000 | 0.0000 | 0.0000 | 100 |
| Tuberculosis | 1.0000 | 0.2100 | 0.3471 | 100 |
| Typhoid | 1.0000 | 0.6700 | 0.8024 | 100 |
| Urinary tract infection | 1.0000 | 0.1600 | 0.2759 | 100 |

## Confusion Matrix

| Actual \\ Predicted | Bronchial Asthma | Dengue | Diabetes | Kidney Stones | Malaria | Migraine | Pneumonia | Tuberculosis | Typhoid | Urinary tract infection |
|---|---|---|---|---|---|---|---|---|---|---|
| **Bronchial Asthma** | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Dengue** | 0 | 76 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Diabetes** | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Kidney Stones** | 0 | 0 | 0 | 57 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Malaria** | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 0 |
| **Migraine** | 0 | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 0 | 0 |
| **Pneumonia** | 46 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Tuberculosis** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 |
| **Typhoid** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 0 |
| **Urinary tract infection** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 |

## Failed Test Cases (Top-3 Failures)

196 test cases failed (expected disease NOT in Top-3).

### Case #8: Expected 'Pneumonia'
- **Input:** loss of appetite, chills, night sweats, chest pain, difficulty breathing, fatigue
- **Detected symptoms:** breathlessness, chest_pain, chills, fatigue, loss_of_appetite, night_sweats
- **Got Top-3:** Lung Abscess (35.0%), Tuberculosis (10.0%), Nephrotic Syndrome (9.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (5) for both diseases  model preference driven by training data distribution

### Case #9: Expected 'Urinary tract infection'
- **Input:** lower abdominal pain, frequent urination, burning urination, fever, pelvic pain, back pain, cloudy urine
- **Detected symptoms:** back_pain, burning_micturition, cloudy_urine, high_fever, pelvic_pain, polyuria
- **Got Top-3:** Pyelonephritis (42.0%), Bladder Infection (12.0%), Prostatitis (7.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Pyelonephritis' has 4 matching symptoms vs expected 'Urinary tract infection' with 3  symptoms are ambiguous between diseases

### Case #16: Expected 'Pneumonia'
- **Input:** cough with mucus, chest pain, wheezing, high fever, night sweats
- **Detected symptoms:** chest_pain, high_fever, mucoid_sputum, night_sweats, wheezing
- **Got Top-3:** Bronchial Asthma (19.0%), Bronchiectasis (14.0%), Lung Abscess (9.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #17: Expected 'Pneumonia'
- **Input:** cough with mucus, difficulty breathing, chills, fatigue, chest pain, wheezing
- **Detected symptoms:** breathlessness, chest_pain, chills, fatigue, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (29.0%), Lung Abscess (24.0%), Pleurisy (5.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #19: Expected 'Tuberculosis'
- **Input:** chills, persistent cough, night sweats, weakness, fever
- **Detected symptoms:** chills, cough, fatigue, high_fever, night_sweats
- **Got Top-3:** Lung Abscess (18.0%), Endocarditis (13.0%), Mononucleosis (6.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #20: Expected 'Bronchial Asthma'
- **Input:** fatigue, chest tightness, shortness of breath, allergy symptoms, dry cough, rapid breathing, breathlessness
- **Detected symptoms:** breathlessness, chest_tightness, continuous_sneezing, dry_cough, fast_heart_rate, fatigue
- **Got Top-3:** Pericarditis (17.3%), Allergic Rhinitis (15.0%), Ankylosing Spondylitis (13.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Pericarditis' has 4 matching symptoms vs expected 'Bronchial Asthma' with 3  symptoms are ambiguous between diseases

### Case #22: Expected 'Urinary tract infection'
- **Input:** burning urination, pelvic pain, nausea, fever, frequent urination
- **Detected symptoms:** burning_micturition, high_fever, nausea, pelvic_pain, polyuria
- **Got Top-3:** Drug Reaction (26.0%), Prostatitis (17.0%), Hyperparathyroidism (9.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Drug Reaction' has 3 matching symptoms vs expected 'Urinary tract infection' with 2  symptoms are ambiguous between diseases

### Case #27: Expected 'Pneumonia'
- **Input:** chills, breathlessness, low oxygen, difficulty breathing, wheezing, cough with mucus, high fever, chest pain
- **Detected symptoms:** breathlessness, chest_pain, chills, high_fever, mucoid_sputum, wheezing
- **Got Top-3:** Lung Abscess (31.0%), Bronchial Asthma (23.0%), Acute Bronchitis (5.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #32: Expected 'Pneumonia'
- **Input:** chills, night sweats, rapid breathing, chest pain, wheezing, loss of appetite, cough with mucus
- **Detected symptoms:** chest_pain, chills, fast_heart_rate, loss_of_appetite, mucoid_sputum, night_sweats, wheezing
- **Got Top-3:** Bronchial Asthma (14.0%), Lung Abscess (10.0%), Nephrotic Syndrome (8.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #35: Expected 'Bronchial Asthma'
- **Input:** dry cough, difficulty breathing, chest tightness
- **Detected symptoms:** breathlessness, chest_tightness, dry_cough
- **Got Top-3:** Ankylosing Spondylitis (21.0%), Pleurisy (17.2%), COVID-19 (14.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #39: Expected 'Diabetes'
- **Input:** tingling feet, dizziness, fatigue, dry mouth, weakness, frequent urination, weight loss, blurred vision, excessive thirst
- **Detected symptoms:** blurred_and_distorted_vision, dizziness, dry_mouth, excessive_thirst, fatigue, polyuria, tingling, weight_loss
- **Got Top-3:** Dehydration Syndrome (18.0%), Type 1 Diabetes (14.0%), Sjogren Syndrome (12.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #40: Expected 'Pneumonia'
- **Input:** rapid breathing, loss of appetite, wheezing, chest pain, chills, difficulty breathing, night sweats
- **Detected symptoms:** breathlessness, chest_pain, chills, fast_heart_rate, loss_of_appetite, night_sweats, wheezing
- **Got Top-3:** Lung Abscess (14.0%), Bronchiectasis (9.0%), Tuberculosis (8.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #46: Expected 'Urinary tract infection'
- **Input:** frequent urination, lower abdominal pain, pelvic pain, cloudy urine, fever
- **Detected symptoms:** cloudy_urine, high_fever, pelvic_pain, polyuria
- **Got Top-3:** Bladder Infection (18.0%), Pyelonephritis (16.0%), Hyperparathyroidism (8.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #59: Expected 'Pneumonia'
- **Input:** cough with mucus, difficulty breathing, chest pain, low oxygen
- **Detected symptoms:** breathlessness, chest_pain, mucoid_sputum
- **Got Top-3:** Bronchial Asthma (19.0%), Pleurisy (11.0%), Aortic Aneurysm (11.0%)
- **Root Cause:** WEAK_FEATURE_ENGINEERING: Only 2/15 (13%) symptoms overlap with expected disease; DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #61: Expected 'Kidney Stones'
- **Input:** flank pain, fever, frequent urination, painful urination, back pain, nausea
- **Detected symptoms:** back_pain, burning_micturition, high_fever, kidney_pain, nausea, polyuria
- **Got Top-3:** Pyelonephritis (47.0%), Hyperparathyroidism (17.0%), Drug Reaction (9.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Pyelonephritis' has 5 matching symptoms vs expected 'Kidney Stones' with 3  symptoms are ambiguous between diseases

### Case #62: Expected 'Tuberculosis'
- **Input:** weight loss, fatigue, fever, breathlessness, night sweats, chills
- **Detected symptoms:** breathlessness, chills, fatigue, high_fever, night_sweats, weight_loss
- **Got Top-3:** Lung Abscess (61.0%), Endocarditis (13.0%), Cellulitis (3.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (6) for both diseases  model preference driven by training data distribution

### Case #67: Expected 'Tuberculosis'
- **Input:** blood in cough, chest pain, night sweats
- **Detected symptoms:** blood_in_sputum, chest_pain, night_sweats
- **Got Top-3:** Lung Abscess (40.0%), Bronchiectasis (20.0%), Pulmonary Embolism (14.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #70: Expected 'Tuberculosis'
- **Input:** fever, night sweats, weakness, weight loss
- **Detected symptoms:** fatigue, high_fever, night_sweats, weight_loss
- **Got Top-3:** Lung Abscess (21.0%), Endocarditis (15.0%), Crohn Disease (12.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #73: Expected 'Pneumonia'
- **Input:** low oxygen, cough with mucus, high fever, night sweats
- **Detected symptoms:** breathlessness, high_fever, mucoid_sputum, night_sweats
- **Got Top-3:** Bronchial Asthma (18.0%), Lung Abscess (10.0%), Mononucleosis (7.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #81: Expected 'Pneumonia'
- **Input:** chest pain, cough with mucus, low oxygen, difficulty breathing, high fever
- **Detected symptoms:** breathlessness, chest_pain, high_fever, mucoid_sputum
- **Got Top-3:** Bronchial Asthma (16.0%), Lung Abscess (13.0%), Pleurisy (7.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #83: Expected 'Pneumonia'
- **Input:** rapid breathing, difficulty breathing, fatigue, high fever, chest pain
- **Detected symptoms:** breathlessness, chest_pain, fast_heart_rate, fatigue, high_fever
- **Got Top-3:** Myocarditis (26.0%), Pericarditis (14.0%), Lung Abscess (9.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (5) for both diseases  model preference driven by training data distribution

### Case #84: Expected 'Typhoid'
- **Input:** nausea, loss of appetite, headache
- **Detected symptoms:** headache, loss_of_appetite, nausea
- **Got Top-3:** Norovirus (14.0%), (vertigo) Paroxymal Positional Vertigo (9.0%), Migraine (5.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #98: Expected 'Pneumonia'
- **Input:** breathlessness, night sweats, loss of appetite, difficulty breathing, chest pain, wheezing, high fever
- **Detected symptoms:** breathlessness, chest_pain, high_fever, loss_of_appetite, night_sweats, wheezing
- **Got Top-3:** Bronchiectasis (14.0%), Mononucleosis (9.0%), Tuberculosis (8.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #101: Expected 'Pneumonia'
- **Input:** chest pain, high fever, low oxygen, difficulty breathing
- **Detected symptoms:** breathlessness, chest_pain, high_fever
- **Got Top-3:** Lung Abscess (15.0%), Pleurisy (8.0%), Hypertension (7.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #104: Expected 'Urinary tract infection'
- **Input:** cloudy urine, pelvic pain, frequent urination, fever
- **Detected symptoms:** cloudy_urine, high_fever, pelvic_pain, polyuria
- **Got Top-3:** Bladder Infection (18.0%), Pyelonephritis (16.0%), Hyperparathyroidism (8.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #108: Expected 'Dengue'
- **Input:** headache, joint pain, rash, nausea, high fever
- **Detected symptoms:** headache, high_fever, joint_pain, nausea, skin_rash
- **Got Top-3:** Chikungunya (57.0%), Drug Reaction (8.0%), Rubella (4.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (5) for both diseases  model preference driven by training data distribution

### Case #116: Expected 'Tuberculosis'
- **Input:** persistent cough, breathlessness, blood in cough, weight loss
- **Detected symptoms:** blood_in_sputum, breathlessness, cough, weight_loss
- **Got Top-3:** Lung Abscess (24.0%), Pulmonary Embolism (21.0%), COPD (13.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #131: Expected 'Typhoid'
- **Input:** diarrhea, headache, continuous fever
- **Detected symptoms:** diarrhoea, headache, high_fever
- **Got Top-3:** Food Poisoning (9.0%), AIDS (8.0%), Appendicitis (7.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #135: Expected 'Pneumonia'
- **Input:** wheezing, chest pain, cough with mucus, high fever, loss of appetite, difficulty breathing, rapid breathing, fatigue
- **Detected symptoms:** breathlessness, chest_pain, fast_heart_rate, fatigue, high_fever, loss_of_appetite, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (21.0%), Myocarditis (12.0%), Tuberculosis (6.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #136: Expected 'Typhoid'
- **Input:** weakness, rose spots, nausea, continuous fever, headache, loss of appetite
- **Detected symptoms:** fatigue, headache, high_fever, loss_of_appetite, nausea, red_spots_over_body
- **Got Top-3:** Meningitis (44.0%), Chicken pox (8.0%), Dengue (7.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Meningitis' has 6 matching symptoms vs expected 'Typhoid' with 5  symptoms are ambiguous between diseases

### Case #144: Expected 'Urinary tract infection'
- **Input:** burning urination, pelvic pain, back pain, lower abdominal pain, frequent urination
- **Detected symptoms:** back_pain, burning_micturition, pelvic_pain, polyuria
- **Got Top-3:** Pyelonephritis (14.0%), Aortic Aneurysm (12.0%), Prostatitis (11.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #147: Expected 'Bronchial Asthma'
- **Input:** shortness of breath, dry cough, chest tightness, fatigue
- **Detected symptoms:** breathlessness, chest_tightness, dry_cough, fatigue
- **Got Top-3:** Pleurisy (25.2%), Ankylosing Spondylitis (23.0%), COVID-19 (15.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #169: Expected 'Dengue'
- **Input:** joint pain, chills, fatigue, rash
- **Detected symptoms:** chills, fatigue, joint_pain, skin_rash
- **Got Top-3:** Sjogren Syndrome (17.0%), Rubella (13.0%), Endocarditis (11.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #170: Expected 'Pneumonia'
- **Input:** rapid breathing, fatigue, chest pain, wheezing, difficulty breathing, cough with mucus
- **Detected symptoms:** breathlessness, chest_pain, fast_heart_rate, fatigue, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (27.0%), Pericarditis (11.0%), Myocarditis (10.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #177: Expected 'Bronchial Asthma'
- **Input:** fatigue, shortness of breath, dry cough
- **Detected symptoms:** breathlessness, dry_cough, fatigue
- **Got Top-3:** Pleurisy (38.0%), Pericarditis (8.0%), Ankylosing Spondylitis (6.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Pleurisy' has 3 matching symptoms vs expected 'Bronchial Asthma' with 2  symptoms are ambiguous between diseases

### Case #180: Expected 'Kidney Stones'
- **Input:** fever, flank pain, back pain, chills, nausea, frequent urination
- **Detected symptoms:** back_pain, chills, high_fever, kidney_pain, nausea, polyuria
- **Got Top-3:** Pyelonephritis (49.0%), Hyperparathyroidism (20.0%), Pancreatitis (4.0%)
- **Root Cause:** AMBIGUOUS_SYMPTOMS: Predicted 'Pyelonephritis' has 5 matching symptoms vs expected 'Kidney Stones' with 2  symptoms are ambiguous between diseases

### Case #190: Expected 'Diabetes'
- **Input:** dry mouth, tingling feet, frequent infections, frequent urination, excessive thirst
- **Detected symptoms:** dry_mouth, excessive_thirst, frequent_infections, polyuria, tingling
- **Got Top-3:** Cholera (17.0%), Sjogren Syndrome (16.0%), Dehydration Syndrome (12.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #191: Expected 'Pneumonia'
- **Input:** high fever, rapid breathing, cough with mucus, chills, difficulty breathing, wheezing, fatigue
- **Detected symptoms:** breathlessness, chills, fast_heart_rate, fatigue, high_fever, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (24.0%), Lung Abscess (11.0%), Endocarditis (10.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #194: Expected 'Pneumonia'
- **Input:** cough with mucus, chest pain, difficulty breathing, wheezing, high fever
- **Detected symptoms:** breathlessness, chest_pain, high_fever, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (26.0%), Lung Abscess (11.0%), Bronchiectasis (11.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #196: Expected 'Bronchial Asthma'
- **Input:** dry cough, shortness of breath, allergy symptoms, night cough
- **Detected symptoms:** breathlessness, continuous_sneezing, coughing_at_night, dry_cough
- **Got Top-3:** Allergy (17.0%), Pleurisy (16.0%), Croup (12.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #198: Expected 'Pneumonia'
- **Input:** low oxygen, chills, cough with mucus, night sweats, high fever, chest pain, loss of appetite, breathlessness, difficulty breathing
- **Detected symptoms:** breathlessness, chest_pain, chills, high_fever, loss_of_appetite, mucoid_sputum, night_sweats
- **Got Top-3:** Lung Abscess (34.0%), Tuberculosis (12.0%), Bronchial Asthma (12.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (5) for both diseases  model preference driven by training data distribution

### Case #199: Expected 'Pneumonia'
- **Input:** chest pain, loss of appetite, high fever, rapid breathing, wheezing, difficulty breathing, cough with mucus, breathlessness, low oxygen
- **Detected symptoms:** breathlessness, chest_pain, fast_heart_rate, high_fever, loss_of_appetite, mucoid_sputum, wheezing
- **Got Top-3:** Bronchial Asthma (17.0%), Myocarditis (9.0%), Bronchiectasis (8.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #201: Expected 'Typhoid'
- **Input:** continuous fever, diarrhea, headache
- **Detected symptoms:** diarrhoea, headache, high_fever
- **Got Top-3:** Food Poisoning (9.0%), AIDS (8.0%), Appendicitis (7.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution

### Case #204: Expected 'Tuberculosis'
- **Input:** fatigue, fever, weight loss, chills, breathlessness, night sweats
- **Detected symptoms:** breathlessness, chills, fatigue, high_fever, night_sweats, weight_loss
- **Got Top-3:** Lung Abscess (61.0%), Endocarditis (13.0%), Cellulitis (3.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (6) for both diseases  model preference driven by training data distribution

### Case #207: Expected 'Tuberculosis'
- **Input:** night sweats, weight loss, chest pain, persistent cough
- **Detected symptoms:** chest_pain, cough, night_sweats, weight_loss
- **Got Top-3:** Lung Abscess (17.0%), Bronchiectasis (12.0%), AIDS (8.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #210: Expected 'Pneumonia'
- **Input:** difficulty breathing, cough with mucus, chills
- **Detected symptoms:** breathlessness, chills, mucoid_sputum
- **Got Top-3:** Bronchial Asthma (19.0%), Lung Abscess (13.0%), Aortic Aneurysm (6.0%)
- **Root Cause:** WEAK_FEATURE_ENGINEERING: Only 2/15 (13%) symptoms overlap with expected disease; DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #214: Expected 'Bronchial Asthma'
- **Input:** shortness of breath, fatigue, chest tightness, allergy symptoms, wheezing, dry cough, difficulty breathing
- **Detected symptoms:** breathlessness, chest_tightness, continuous_sneezing, dry_cough, fatigue, wheezing
- **Got Top-3:** COPD (20.0%), Allergic Rhinitis (15.0%), Pleurisy (14.3%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (4) for both diseases  model preference driven by training data distribution

### Case #228: Expected 'Dengue'
- **Input:** rash, chills, headache, high fever
- **Detected symptoms:** chills, headache, high_fever, skin_rash
- **Got Top-3:** Cellulitis (12.0%), Chikungunya (9.0%), Viral Fever (6.0%)
- **Root Cause:** MODEL_LIMITATION: Model ranked a competing disease higher despite adequate symptom coverage

### Case #237: Expected 'Dengue'
- **Input:** dizziness, high fever, headache
- **Detected symptoms:** dizziness, headache, high_fever
- **Got Top-3:** Tinnitus (13.0%), Hypertension (12.0%), Vestibular Neuritis (7.0%)
- **Root Cause:** WEAK_FEATURE_ENGINEERING: Only 2/15 (13%) symptoms overlap with expected disease; DATASET_IMBALANCE: Equal symptom overlap (2) for both diseases  model preference driven by training data distribution

### Case #239: Expected 'Dengue'
- **Input:** rash, nausea, joint pain
- **Detected symptoms:** joint_pain, nausea, skin_rash
- **Got Top-3:** Chikungunya (22.0%), Celiac Disease (13.0%), Sjogren Syndrome (11.0%)
- **Root Cause:** DATASET_IMBALANCE: Equal symptom overlap (3) for both diseases  model preference driven by training data distribution


## Improvements Made
- Switched from Naive Bayes to Random Forest (Top-3: 0.8040 > 0.7810)

## Iteration History

| Iteration | Top-1 | Top-3 | F1 | Changes |
|---|---|---|---|---|
| 1 | 0.6270 | 0.7810 | 0.1678 | Baseline (NLP v2.1 + QA mappings) |
| 2 | 0.5120 | 0.8040 | 0.1061 | Best model: Random Forest |
| 3 | 0.5120 | 0.8040 | 0.1061 | Final analysis  196 failures remaining |

## Robustness Testing

| Metric | Value |
|---|---|
| Total Robustness Cases | 32 |
| No-Crash Rate | 32/32 (100.0%) |
| Crashes | 0 |

## Production Readiness Score

| Component | Score | Max |
|---|---|---|
| Top-3 Accuracy | 32.2 | 40 |
| Top-1 Accuracy | 12.8 | 25 |
| F1 Score | 1.6 | 15 |
| Robustness | 10.0 | 10 |
| Speed (<100ms) | 5 | 5 |
| Disease Coverage | 5 | 5 |
| **TOTAL** | **66.6** | **100** |