"""
Saarthi Diagnostic Platform — Disease Knowledge Base Registry (v2)
================================================================
Compiles clinical disease metadata with disease-specific curated content
for ~167 diseases. Includes severity, emergency signs, recovery time,
specialist, prevention, contagious status, and self-care guidance.
"""

import os
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# Disease-Specific Metadata — Curated for High-Priority Diseases
# ═══════════════════════════════════════════════════════════════════

CURATED_DISEASES = {
    # ── RESPIRATORY ──────────────────────────────────────────────
    "Common Cold": {
        "category": "Respiratory",
        "specialist": "General Physician",
        "severity": "Low",
        "description": "A viral upper respiratory tract infection causing nasal congestion, sore throat, and mild fever. Usually self-limiting within 7-10 days.",
        "causes": "Rhinoviruses (most common), coronaviruses, RSV, and other respiratory viruses spread through droplets and contact.",
        "risk_factors": "Weakened immunity, close contact with infected individuals, cold weather, poor hand hygiene, stress, lack of sleep.",
        "prevention": "Frequent hand washing, avoid touching face, maintain distance from sick individuals, adequate sleep, vitamin C intake.",
        "recovery_time": "7-10 days",
        "emergency_signs": "High fever >103°F, severe headache with stiff neck, difficulty breathing, persistent chest pain.",
        "contagious": "Yes (droplets, 1-2 weeks)",
        "self_care": "Yes — rest, hydration, OTC decongestants, warm fluids, steam inhalation.",
    },
    "Influenza": {
        "category": "Respiratory",
        "specialist": "General Physician / Pulmonologist",
        "severity": "Moderate",
        "description": "A contagious respiratory illness caused by influenza viruses, characterized by sudden onset of high fever, body aches, and respiratory symptoms.",
        "causes": "Influenza A and B viruses, transmitted through respiratory droplets and contaminated surfaces.",
        "risk_factors": "Age >65 or <5, pregnancy, chronic diseases, immunocompromised, healthcare workers.",
        "prevention": "Annual flu vaccination, hand hygiene, respiratory etiquette, avoid crowded places during flu season.",
        "recovery_time": "1-2 weeks; fatigue may persist longer",
        "emergency_signs": "Difficulty breathing, persistent chest pain, confusion, severe vomiting, flu symptoms that improve then return with worsening cough.",
        "contagious": "Yes (airborne droplets, 5-7 days)",
        "self_care": "Yes initially — rest, fluids, OTC fever reducers. Antiviral medication (oseltamivir) within 48 hours of onset.",
    },
    "Pneumonia": {
        "category": "Respiratory",
        "specialist": "Pulmonologist / Chest Specialist",
        "severity": "High",
        "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid. Can range from mild to life-threatening.",
        "causes": "Bacteria (Streptococcus pneumoniae most common), viruses, fungi. Hospital-acquired forms often involve resistant bacteria.",
        "risk_factors": "Age >65 or <2, chronic lung disease, smoking, weakened immune system, hospitalization, recent surgery.",
        "prevention": "Pneumococcal vaccination, flu vaccination, good hygiene, avoid smoking, manage chronic conditions.",
        "recovery_time": "1-3 weeks for mild; 6-8 weeks for severe",
        "emergency_signs": "Severe difficulty breathing, chest pain, high fever >104°F, confusion, bluish lips/fingertips, coughing up blood.",
        "contagious": "Varies — bacterial pneumonia less so; viral pneumonia can be contagious",
        "self_care": "No — requires medical evaluation. Mild cases may be treated at home with prescribed antibiotics.",
    },
    "Bronchial Asthma": {
        "category": "Respiratory",
        "specialist": "Pulmonologist / Allergist",
        "severity": "Moderate",
        "description": "A chronic inflammatory airway disease causing recurrent episodes of wheezing, breathlessness, chest tightness, and coughing.",
        "causes": "Combination of genetic predisposition and environmental triggers including allergens, air pollution, exercise, cold air.",
        "risk_factors": "Family history of asthma/allergies, childhood respiratory infections, obesity, smoking exposure, occupational exposures.",
        "prevention": "Identify and avoid triggers, take controller medications regularly, use air purifiers, maintain healthy weight.",
        "recovery_time": "Chronic condition — episodes resolve in hours to days with treatment",
        "emergency_signs": "Severe breathlessness even at rest, inability to speak full sentences, blue lips, no relief from rescue inhaler, peak flow <50% of personal best.",
        "contagious": "No",
        "self_care": "Partial — rescue inhaler for mild episodes. Action plan from doctor. Severe attacks need emergency care.",
    },
    "Tuberculosis": {
        "category": "Respiratory",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "severity": "High",
        "description": "A serious bacterial infection that mainly affects the lungs, caused by Mycobacterium tuberculosis. Can spread to other organs.",
        "causes": "Mycobacterium tuberculosis, spread through airborne droplets when an infected person coughs, sneezes, or speaks.",
        "risk_factors": "HIV/AIDS, weakened immunity, close contact with TB patients, malnutrition, diabetes, smoking, crowded living conditions.",
        "prevention": "BCG vaccination, identify and treat latent TB, avoid close contact with active TB patients, good ventilation.",
        "recovery_time": "6-9 months with full course of anti-TB drugs",
        "emergency_signs": "Coughing up large amounts of blood, severe chest pain, high persistent fever, extreme weight loss, confusion.",
        "contagious": "Yes (airborne, until 2-3 weeks into treatment)",
        "self_care": "No — requires supervised medical treatment (DOTS). Never stop medications early.",
    },
    "COVID-19": {
        "category": "Respiratory",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "severity": "Moderate to High",
        "description": "A respiratory illness caused by SARS-CoV-2 virus, ranging from mild cold-like symptoms to severe pneumonia. Can cause loss of smell/taste and long-term complications.",
        "causes": "SARS-CoV-2 virus, transmitted through respiratory droplets and aerosols, with possible fomite transmission.",
        "risk_factors": "Age >60, obesity, diabetes, heart disease, lung disease, immunocompromised, unvaccinated status.",
        "prevention": "Vaccination, mask-wearing in high-risk settings, hand hygiene, adequate ventilation, social distancing during outbreaks.",
        "recovery_time": "Mild: 1-2 weeks. Severe: 3-6 weeks. Long COVID symptoms may persist months.",
        "emergency_signs": "Persistent chest pain, difficulty breathing at rest, confusion, bluish lips/face, inability to stay awake, oxygen saturation <94%.",
        "contagious": "Yes (airborne, 5-10 days typically)",
        "self_care": "Mild cases: yes — rest, fluids, fever management, pulse oximeter monitoring. Seek care if breathing worsens.",
    },
    "Sinusitis": {
        "category": "Respiratory",
        "specialist": "ENT Specialist (Otolaryngologist)",
        "severity": "Low to Moderate",
        "description": "Inflammation of the paranasal sinuses causing facial pain, nasal congestion, thick nasal discharge, and reduced smell.",
        "causes": "Viral infection (most common), bacterial infection, allergies, nasal polyps, deviated septum.",
        "risk_factors": "Allergic rhinitis, nasal polyps, deviated septum, smoking, dental infections, immunodeficiency.",
        "prevention": "Treat allergies promptly, use humidifier, nasal saline irrigation, avoid smoking, hand hygiene.",
        "recovery_time": "Acute: 7-10 days. Chronic: may require weeks of treatment.",
        "emergency_signs": "High fever >102°F, severe headache, visual changes, swelling around eyes, stiff neck.",
        "contagious": "Viral sinusitis: yes. Bacterial/allergic: no.",
        "self_care": "Yes — saline rinse, steam, decongestants. See doctor if symptoms persist >10 days.",
    },

    # ── CARDIOVASCULAR ───────────────────────────────────────────
    "Heart attack": {
        "category": "Cardiovascular",
        "specialist": "Cardiologist",
        "severity": "Critical",
        "description": "A medical emergency where blood flow to part of the heart muscle is blocked, causing tissue damage. Requires immediate treatment.",
        "causes": "Coronary artery blockage from atherosclerotic plaque rupture, blood clot formation.",
        "risk_factors": "High cholesterol, hypertension, smoking, diabetes, obesity, family history, sedentary lifestyle, stress.",
        "prevention": "Regular exercise, heart-healthy diet, manage BP/cholesterol/diabetes, quit smoking, stress management.",
        "recovery_time": "Hospital: 3-5 days. Full recovery: 3-6 months. Cardiac rehab recommended.",
        "emergency_signs": "CALL EMERGENCY IMMEDIATELY. Crushing chest pain, pain radiating to arm/jaw, cold sweat, severe shortness of breath, nausea.",
        "contagious": "No",
        "self_care": "NO — this is a medical emergency. Call ambulance immediately. Chew aspirin if not allergic.",
    },
    "Hypertension": {
        "category": "Cardiovascular",
        "specialist": "Cardiologist / Internal Medicine",
        "severity": "Moderate to High",
        "description": "Persistently elevated blood pressure (≥140/90 mmHg) that increases risk of heart disease, stroke, and kidney damage. Often called the 'silent killer'.",
        "causes": "Primary: unknown (95% of cases). Secondary: kidney disease, hormonal disorders, medications, sleep apnea.",
        "risk_factors": "Family history, age >45, obesity, high salt diet, sedentary lifestyle, stress, excessive alcohol, smoking.",
        "prevention": "Low-sodium diet, regular exercise, maintain healthy weight, limit alcohol, manage stress, regular BP monitoring.",
        "recovery_time": "Chronic condition — requires lifelong management",
        "emergency_signs": "BP >180/120, severe headache, chest pain, vision changes, difficulty speaking, blood in urine, seizures.",
        "contagious": "No",
        "self_care": "Partial — lifestyle modifications help. Most patients need prescribed medications. Regular monitoring essential.",
    },
    "Congestive Heart Failure": {
        "category": "Cardiovascular",
        "specialist": "Cardiologist",
        "severity": "High to Critical",
        "description": "A chronic condition where the heart cannot pump blood efficiently enough to meet the body's needs, causing fluid buildup in lungs and extremities.",
        "causes": "Coronary artery disease, prior heart attack, hypertension, valve disease, cardiomyopathy, myocarditis.",
        "risk_factors": "Prior heart attack, coronary artery disease, hypertension, diabetes, obesity, valve disease, sleep apnea.",
        "prevention": "Control blood pressure and cholesterol, manage diabetes, maintain healthy weight, exercise regularly, avoid excessive alcohol.",
        "recovery_time": "Chronic condition — managed long-term with medication and lifestyle changes",
        "emergency_signs": "Sudden severe shortness of breath, rapid weight gain (>3 lbs/day), chest pain, fainting, coughing up pink frothy sputum.",
        "contagious": "No",
        "self_care": "Partial — daily weight monitoring, fluid/salt restriction, medication adherence. Requires regular cardiology follow-up.",
    },

    # ── NEUROLOGICAL ─────────────────────────────────────────────
    "Migraine": {
        "category": "Neurological",
        "specialist": "Neurologist",
        "severity": "Moderate",
        "description": "A neurological condition causing intense, throbbing headaches usually on one side of the head, often accompanied by nausea and sensitivity to light/sound.",
        "causes": "Complex neurovascular disorder involving brain chemical imbalances, trigeminal nerve activation, and blood vessel changes.",
        "risk_factors": "Family history, female sex, hormonal changes, stress, certain foods (aged cheese, alcohol, MSG), sleep disruption.",
        "prevention": "Identify and avoid triggers, regular sleep schedule, stress management, prophylactic medications for frequent migraines.",
        "recovery_time": "Individual episodes: 4-72 hours. Chronic migraines require ongoing management.",
        "emergency_signs": "Sudden worst headache of your life, headache with fever and stiff neck, confusion, seizures, vision loss, weakness on one side.",
        "contagious": "No",
        "self_care": "Yes for mild attacks — dark quiet room, cold compress, OTC pain relievers. Frequent migraines need prescription treatment.",
    },
    "Parkinson Disease": {
        "category": "Neurological",
        "specialist": "Neurologist / Movement Disorder Specialist",
        "severity": "High",
        "description": "A progressive neurodegenerative disorder affecting movement, caused by loss of dopamine-producing brain cells. Characterized by tremor, rigidity, and slowness.",
        "causes": "Loss of dopamine-producing neurons in the substantia nigra. Exact cause unknown; involves genetic and environmental factors.",
        "risk_factors": "Age >60, male sex, family history, pesticide/herbicide exposure, head trauma, rural living.",
        "prevention": "No proven prevention. Regular exercise, caffeine consumption, and avoiding pesticide exposure may reduce risk.",
        "recovery_time": "Chronic progressive condition — no cure. Symptoms managed with medication and therapy long-term.",
        "emergency_signs": "Sudden inability to move (freezing episodes), severe confusion, high fever with rigidity (neuroleptic malignant syndrome), falls with head injury.",
        "contagious": "No",
        "self_care": "Partial — exercise programs, physical therapy, support groups. Requires neurologist-prescribed medications (levodopa).",
    },
    "Alzheimer Disease": {
        "category": "Neurological",
        "specialist": "Neurologist / Geriatric Psychiatrist",
        "severity": "High",
        "description": "A progressive neurodegenerative disease causing memory loss, cognitive decline, and behavioral changes. The most common form of dementia.",
        "causes": "Abnormal protein deposits (amyloid plaques and tau tangles) in the brain causing neuron death. Complex interaction of genetics, lifestyle, and environment.",
        "risk_factors": "Age >65, family history, APOE4 gene, Down syndrome, head trauma, cardiovascular risk factors, social isolation.",
        "prevention": "Regular physical and mental exercise, social engagement, heart-healthy diet, manage cardiovascular risk factors, adequate sleep.",
        "recovery_time": "Chronic progressive condition — no cure. Disease progression spans 4-20 years.",
        "emergency_signs": "Sudden confusion or agitation, wandering/getting lost, falls, inability to eat/drink, signs of infection (fever in elderly).",
        "contagious": "No",
        "self_care": "No — requires medical management, caregiver support, and safety planning. Medications may slow progression.",
    },
    "Epilepsy": {
        "category": "Neurological",
        "specialist": "Neurologist / Epileptologist",
        "severity": "Moderate to High",
        "description": "A neurological disorder characterized by recurrent, unprovoked seizures due to abnormal electrical activity in the brain.",
        "causes": "Brain injury, stroke, genetic factors, infections (meningitis), developmental disorders. Often cause is unknown.",
        "risk_factors": "Family history, head trauma, stroke, brain infections, premature birth, abnormal brain development.",
        "prevention": "Wear seat belts and helmets, prevent head injuries, manage stroke risk factors, prenatal care to reduce birth complications.",
        "recovery_time": "Chronic condition — 60-70% of patients achieve seizure control with medication. Some may eventually discontinue treatment.",
        "emergency_signs": "Seizure lasting >5 minutes, repeated seizures without recovery, breathing difficulties after seizure, seizure in water, first-time seizure.",
        "contagious": "No",
        "self_care": "Partial — medication adherence, adequate sleep, avoid seizure triggers (alcohol, flashing lights). Emergency plan needed.",
    },

    # ── DERMATOLOGY ──────────────────────────────────────────────
    "Fungal infection": {
        "category": "Skin & Dermatology",
        "specialist": "Dermatologist",
        "severity": "Low to Moderate",
        "description": "Skin infections caused by fungi (dermatophytes), presenting as itchy, red, ring-shaped or patchy lesions on skin, nails, or scalp.",
        "causes": "Dermatophyte fungi (Trichophyton, Microsporum, Epidermophyton) spread through direct contact or contaminated surfaces.",
        "risk_factors": "Warm humid climate, excessive sweating, tight clothing, shared personal items, weakened immunity, diabetes.",
        "prevention": "Keep skin dry, wear breathable fabrics, avoid sharing towels/clothing, treat athlete's foot to prevent spread, antifungal powder in shoes.",
        "recovery_time": "Skin: 2-4 weeks. Nails: 3-6 months. Scalp: 4-8 weeks.",
        "emergency_signs": "Rapidly spreading redness with fever, pus drainage, severe pain (may indicate bacterial superinfection).",
        "contagious": "Yes (direct contact and fomites)",
        "self_care": "Yes for mild cases — OTC antifungal cream/powder, keep area dry. See doctor if not improving in 2 weeks.",
    },
    "Psoriasis": {
        "category": "Skin & Dermatology",
        "specialist": "Dermatologist",
        "severity": "Moderate",
        "description": "A chronic autoimmune skin condition causing rapid skin cell buildup, resulting in thick, silvery scales and itchy, dry, red patches.",
        "causes": "Autoimmune — T-cells mistakenly attack healthy skin cells, accelerating skin cell production to 3-4 days instead of normal 3-4 weeks.",
        "risk_factors": "Family history, stress, infections (streptococcal), medications (lithium, beta-blockers), obesity, smoking, alcohol.",
        "prevention": "Manage stress, moisturize regularly, avoid skin trauma, limit alcohol, maintain healthy weight, avoid known triggers.",
        "recovery_time": "Chronic condition with flares and remissions. Flares may last weeks to months.",
        "emergency_signs": "Generalized redness covering most of the body (erythrodermic psoriasis), fever with widespread pustules, joint swelling with skin symptoms.",
        "contagious": "No",
        "self_care": "Partial — moisturizers, coal tar products, limited sun exposure. Moderate-severe cases need prescription treatment.",
    },

    # ── GASTROINTESTINAL ─────────────────────────────────────────
    "GERD": {
        "category": "Gastroenterology",
        "specialist": "Gastroenterologist",
        "severity": "Low to Moderate",
        "description": "A chronic digestive condition where stomach acid frequently flows back into the esophagus, causing heartburn and acid regurgitation.",
        "causes": "Weakened lower esophageal sphincter allowing acid reflux. Contributing factors include obesity, hiatal hernia, pregnancy, delayed gastric emptying.",
        "risk_factors": "Obesity, pregnancy, hiatal hernia, smoking, certain foods (spicy, fatty, citrus), eating late at night, certain medications.",
        "prevention": "Maintain healthy weight, eat smaller meals, avoid lying down after eating, elevate head of bed, avoid trigger foods, quit smoking.",
        "recovery_time": "Symptoms improve in 2-4 weeks with treatment. Chronic condition requiring ongoing lifestyle management.",
        "emergency_signs": "Severe chest pain (rule out heart attack), difficulty swallowing, vomiting blood, black/tarry stools, unexplained weight loss.",
        "contagious": "No",
        "self_care": "Yes for mild symptoms — antacids, dietary changes, weight management. See doctor for frequent symptoms (>2x/week).",
    },
    "Appendicitis": {
        "category": "Gastroenterology",
        "specialist": "General Surgeon",
        "severity": "High",
        "description": "Inflammation of the appendix causing severe right lower abdominal pain. Requires urgent surgical treatment to prevent rupture.",
        "causes": "Blockage of the appendix opening by fecal matter, foreign body, infection, or lymphoid tissue swelling.",
        "risk_factors": "Age 10-30, family history, male sex, low-fiber diet.",
        "prevention": "No proven prevention. High-fiber diet may reduce risk.",
        "recovery_time": "Uncomplicated: 1-3 weeks post-surgery. Ruptured: 4-6 weeks.",
        "emergency_signs": "Sudden severe right lower abdominal pain, rigid abdomen, high fever, inability to pass gas, signs of shock (rapid pulse, dizziness).",
        "contagious": "No",
        "self_care": "NO — requires immediate medical evaluation. Do not take laxatives or pain medications before diagnosis.",
    },
    "Crohn Disease": {
        "category": "Gastroenterology",
        "specialist": "Gastroenterologist",
        "severity": "Moderate to High",
        "description": "A chronic inflammatory bowel disease (IBD) that can affect any part of the GI tract, causing abdominal pain, severe diarrhea, fatigue, and malnutrition.",
        "causes": "Exact cause unknown. Involves immune system dysfunction, genetics (NOD2 gene), and environmental triggers.",
        "risk_factors": "Family history, smoking (strongest modifiable risk), age 15-35, Ashkenazi Jewish heritage, Western diet, NSAIDs.",
        "prevention": "No proven prevention. Quit smoking, anti-inflammatory diet, stress management may reduce flares.",
        "recovery_time": "Chronic condition with flares and remissions. Flares may last weeks to months. Lifelong management needed.",
        "emergency_signs": "Severe abdominal pain with distension, high fever, bloody diarrhea with dizziness, signs of bowel obstruction (vomiting, inability to pass gas).",
        "contagious": "No",
        "self_care": "Partial — diet modifications, stress management during remission. Active flares require medical treatment.",
    },
    "Ulcerative Colitis": {
        "category": "Gastroenterology",
        "specialist": "Gastroenterologist",
        "severity": "Moderate to High",
        "description": "A chronic inflammatory bowel disease affecting the colon and rectum, causing continuous mucosal inflammation with bloody diarrhea and abdominal pain.",
        "causes": "Autoimmune — immune system attacks the colon lining. Genetic predisposition with environmental triggers.",
        "risk_factors": "Family history, age 15-30, Ashkenazi Jewish heritage, cessation of smoking (paradoxically), Western diet.",
        "prevention": "No proven prevention. Regular colonoscopy screening for early detection of complications.",
        "recovery_time": "Chronic condition — flares typically last days to weeks. Long-term medication needed to maintain remission.",
        "emergency_signs": "Severe bloody diarrhea (>6 episodes/day), high fever, rapid heart rate, severe abdominal pain/distension, signs of dehydration.",
        "contagious": "No",
        "self_care": "Partial — maintain remission with medication adherence, low-residue diet during flares. Active disease needs medical management.",
    },

    # ── KIDNEY / URINARY ─────────────────────────────────────────
    "Urinary tract infection": {
        "category": "Urology & Nephrology",
        "specialist": "Urologist / General Physician",
        "severity": "Low to Moderate",
        "description": "A bacterial infection in any part of the urinary system (bladder, urethra, ureters, kidneys). Most commonly affects the bladder (cystitis).",
        "causes": "Bacteria (E. coli most common) entering the urinary tract through the urethra. Can ascend to kidneys if untreated.",
        "risk_factors": "Female sex, sexual activity, certain contraceptives, menopause, urinary catheter, kidney stones, diabetes.",
        "prevention": "Drink plenty of water, urinate after intercourse, wipe front to back, avoid irritating feminine products, cranberry products may help.",
        "recovery_time": "Uncomplicated: 3-7 days with antibiotics. Kidney infection: 10-14 days.",
        "emergency_signs": "High fever with flank pain (kidney infection), blood in urine, severe pelvic pain, inability to urinate, nausea/vomiting.",
        "contagious": "No",
        "self_care": "Partial — increase water intake, urinary analgesics. Antibiotics needed for cure. See doctor for confirmed diagnosis.",
    },
    "Kidney Stones": {
        "category": "Urology & Nephrology",
        "specialist": "Urologist / Nephrologist",
        "severity": "Moderate to High",
        "description": "Hard mineral deposits that form in the kidneys, causing severe pain when passing through the urinary tract. One of the most painful conditions.",
        "causes": "Supersaturation of urine with calcium, oxalate, uric acid, or other minerals. Dehydration is the most common contributing factor.",
        "risk_factors": "Dehydration, high-protein/sodium diet, obesity, family history, recurrent UTIs, certain medications, gout.",
        "prevention": "Drink 2-3 liters of water daily, reduce sodium intake, limit animal protein, reduce oxalate-rich foods, maintain healthy weight.",
        "recovery_time": "Small stones: pass in 1-6 weeks. Large stones may need lithotripsy or surgery, recovery 1-2 weeks post-procedure.",
        "emergency_signs": "Severe pain with high fever/chills (infected stone — emergency!), inability to urinate, persistent vomiting, blood in urine with fever.",
        "contagious": "No",
        "self_care": "Yes for small stones — hydration, pain management, strain urine. Stones >6mm or with infection need medical intervention.",
    },

    # ── ENDOCRINE ────────────────────────────────────────────────
    "Diabetes": {
        "category": "Endocrinology",
        "specialist": "Endocrinologist / Diabetologist",
        "severity": "Moderate to High",
        "description": "A chronic metabolic disorder characterized by elevated blood glucose levels due to insufficient insulin production or insulin resistance.",
        "causes": "Type 2: insulin resistance + progressive beta-cell dysfunction. Genetic predisposition with lifestyle factors (obesity, inactivity, poor diet).",
        "risk_factors": "Obesity, sedentary lifestyle, family history, age >45, gestational diabetes history, PCOS, certain ethnicities.",
        "prevention": "Maintain healthy weight, regular exercise (150 min/week), balanced diet, regular blood sugar screening after age 45.",
        "recovery_time": "Chronic condition — requires lifelong management. Well-controlled diabetes allows normal quality of life.",
        "emergency_signs": "Blood sugar >300 mg/dL, diabetic ketoacidosis (fruity breath, vomiting, confusion), hypoglycemia (shakiness, confusion, loss of consciousness).",
        "contagious": "No",
        "self_care": "Partial — blood sugar monitoring, diet management, exercise. Most patients need prescribed medications. Regular check-ups essential.",
    },
    "Hypothyroidism": {
        "category": "Endocrinology",
        "specialist": "Endocrinologist",
        "severity": "Moderate",
        "description": "An underactive thyroid gland that doesn't produce enough thyroid hormones, causing fatigue, weight gain, cold intolerance, and cognitive slowing.",
        "causes": "Hashimoto's thyroiditis (autoimmune — most common), thyroid surgery, radiation therapy, certain medications, iodine deficiency.",
        "risk_factors": "Female sex, age >60, family history of thyroid disease, autoimmune disorders, previous thyroid surgery/radiation.",
        "prevention": "Adequate iodine intake, regular thyroid screening for high-risk individuals, manage autoimmune conditions.",
        "recovery_time": "Symptoms improve in 2-4 weeks after starting levothyroxine. Full stabilization in 6-8 weeks. Lifelong treatment needed.",
        "emergency_signs": "Myxedema coma (severe hypothermia, unconsciousness, slow breathing) — rare but life-threatening emergency.",
        "contagious": "No",
        "self_care": "No — requires prescription thyroid hormone replacement. Regular blood tests to monitor levels.",
    },

    # ── INFECTIOUS ───────────────────────────────────────────────
    "Malaria": {
        "category": "Infectious Diseases",
        "specialist": "Infectious Disease Specialist / General Physician",
        "severity": "High",
        "description": "A mosquito-borne parasitic disease causing cyclic fever, chills, sweating, and potentially fatal complications if caused by P. falciparum.",
        "causes": "Plasmodium parasites (P. falciparum, P. vivax, P. ovale, P. malariae) transmitted through infected Anopheles mosquito bites.",
        "risk_factors": "Travel to endemic areas, lack of mosquito protection, no prophylaxis, pregnancy, young children, immunocompromised.",
        "prevention": "Mosquito nets, insect repellent, antimalarial prophylaxis when traveling, eliminate standing water, wear long clothing.",
        "recovery_time": "Uncomplicated: 2-3 days with treatment. Severe: weeks of recovery. P. vivax/ovale may relapse.",
        "emergency_signs": "Very high fever >104°F, confusion/altered consciousness, severe anemia, respiratory distress, jaundice, seizures, dark urine.",
        "contagious": "Not person-to-person (mosquito vector only)",
        "self_care": "NO — requires immediate antimalarial drug treatment. Delay can be fatal with P. falciparum.",
    },
    "Dengue": {
        "category": "Infectious Diseases",
        "specialist": "Infectious Disease Specialist / General Physician",
        "severity": "Moderate to High",
        "description": "A mosquito-borne viral infection causing high fever, severe body pain, rash, and in severe cases, hemorrhagic complications.",
        "causes": "Dengue virus (4 serotypes) transmitted through Aedes aegypti mosquito bites. Second infection with different serotype increases severity risk.",
        "risk_factors": "Living in/traveling to tropical areas, prior dengue infection (different serotype), rainy season, stagnant water near residence.",
        "prevention": "Mosquito control, eliminate breeding sites (stagnant water), use repellent, wear protective clothing, dengue vaccination where available.",
        "recovery_time": "7-10 days. Post-recovery fatigue may last 2-4 weeks.",
        "emergency_signs": "Severe abdominal pain, persistent vomiting, bleeding gums/nose, blood in vomit/stool, rapid platelet drop, restlessness, cold clammy skin.",
        "contagious": "Not person-to-person (mosquito vector only)",
        "self_care": "Mild cases: rest, hydration, paracetamol (NO aspirin/ibuprofen). Monitor platelet count. Seek hospital care if warning signs appear.",
    },
    "Typhoid": {
        "category": "Infectious Diseases",
        "specialist": "Infectious Disease Specialist / General Physician",
        "severity": "Moderate to High",
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food and water, causing prolonged fever and GI symptoms.",
        "causes": "Salmonella typhi bacteria, transmitted through fecal-oral route via contaminated food, water, or close contact with infected person.",
        "risk_factors": "Travel to endemic areas, poor sanitation, contaminated water sources, street food consumption, lack of hand hygiene.",
        "prevention": "Typhoid vaccination before travel, drink safe water, eat thoroughly cooked food, hand hygiene, avoid street food in endemic areas.",
        "recovery_time": "2-4 weeks with antibiotics. Full recovery may take 4-6 weeks.",
        "emergency_signs": "Intestinal perforation (sudden severe abdominal pain), high fever >104°F for >2 weeks, intestinal bleeding, confusion, very slow heart rate.",
        "contagious": "Yes (fecal-oral route)",
        "self_care": "No — requires antibiotic treatment. Rest, hydration, and easily digestible food as supportive care.",
    },
    "Chicken pox": {
        "category": "Infectious Diseases",
        "specialist": "General Physician / Pediatrician",
        "severity": "Low to Moderate",
        "description": "A highly contagious viral infection caused by varicella-zoster virus, producing an itchy, blister-like rash all over the body with mild fever.",
        "causes": "Varicella-zoster virus (VZV), spread through direct contact with rash or airborne droplets from coughing/sneezing.",
        "risk_factors": "No prior vaccination or infection, pregnancy, immunocompromised, age (more severe in adults), household exposure.",
        "prevention": "Varicella vaccine (2 doses), avoid contact with infected individuals, post-exposure prophylaxis for high-risk persons.",
        "recovery_time": "1-2 weeks for lesions to crust over. Full recovery 2-3 weeks. Adults may take longer.",
        "emergency_signs": "High fever >102°F for >4 days, difficulty breathing, stiff neck with headache, rash spreading to eyes, bleeding into blisters, bacterial superinfection.",
        "contagious": "Highly contagious (1-2 days before rash until all lesions crust over)",
        "self_care": "Yes for children — calamine lotion, oatmeal baths, antihistamines for itch. Avoid aspirin. Adults and immunocompromised may need antivirals.",
    },

    # ── AUTOIMMUNE ───────────────────────────────────────────────
    "Rheumatoid Arthritis": {
        "category": "Rheumatology & Autoimmune",
        "specialist": "Rheumatologist",
        "severity": "Moderate to High",
        "description": "A chronic autoimmune disease where the immune system attacks joint linings, causing painful swelling, joint deformity, and systemic inflammation.",
        "causes": "Autoimmune — immune system attacks synovial membrane of joints. Exact trigger unknown; involves genetic and environmental factors.",
        "risk_factors": "Female sex, age 40-60, family history, smoking, obesity, environmental exposures (silica, asbestos).",
        "prevention": "No proven prevention. Quit smoking, maintain healthy weight, regular exercise. Early treatment prevents joint damage.",
        "recovery_time": "Chronic condition — early aggressive treatment can achieve remission. Flares may last weeks.",
        "emergency_signs": "Sudden loss of joint function, severe joint infection signs (red, hot, swollen single joint with fever), vasculitis symptoms, lung involvement.",
        "contagious": "No",
        "self_care": "Partial — gentle exercise, joint protection, heat/cold therapy. Requires disease-modifying drugs (DMARDs) prescribed by rheumatologist.",
    },

    # ── GENERAL ──────────────────────────────────────────────────
    "Allergy": {
        "category": "Immunology & Allergy",
        "specialist": "Allergist / Immunologist",
        "severity": "Low to Moderate",
        "description": "An exaggerated immune response to normally harmless substances (allergens) causing sneezing, itching, rash, or in severe cases, anaphylaxis.",
        "causes": "Immune system overreaction to allergens: pollen, dust mites, pet dander, certain foods, insect stings, medications.",
        "risk_factors": "Family history of allergies/asthma, childhood, existing allergic conditions, environmental exposures.",
        "prevention": "Identify and avoid known allergens, use air purifiers, keep windows closed during high pollen, shower after outdoor exposure.",
        "recovery_time": "Symptoms resolve within hours to days after allergen removal. Chronic allergies need ongoing management.",
        "emergency_signs": "Anaphylaxis: difficulty breathing, throat swelling, rapid pulse, dizziness, loss of consciousness — USE EPINEPHRINE AND CALL EMERGENCY.",
        "contagious": "No",
        "self_care": "Yes for mild symptoms — antihistamines, nasal sprays, avoid allergens. Severe allergies need allergist evaluation and may need immunotherapy.",
    },
    "Anemia": {
        "category": "Hematology",
        "specialist": "Hematologist / Internal Medicine",
        "severity": "Low to Moderate",
        "description": "A condition where the blood lacks enough healthy red blood cells or hemoglobin, reducing oxygen delivery to tissues. Causes fatigue, pallor, and weakness.",
        "causes": "Iron deficiency (most common), vitamin B12/folate deficiency, chronic disease, blood loss, bone marrow disorders, hemolysis.",
        "risk_factors": "Women of childbearing age, pregnancy, poor diet, chronic diseases, GI conditions, heavy menstrual periods, family history.",
        "prevention": "Iron-rich diet (red meat, spinach, lentils), vitamin C with iron foods, prenatal vitamins during pregnancy, regular health check-ups.",
        "recovery_time": "Iron-deficiency: hemoglobin improves in 2-4 weeks, iron stores replete in 3-6 months. Other types vary.",
        "emergency_signs": "Severe shortness of breath, chest pain, rapid heartbeat, fainting, hemoglobin <7 g/dL, acute blood loss.",
        "contagious": "No",
        "self_care": "Partial — dietary improvement, iron/vitamin supplements (after blood tests). Underlying cause must be identified and treated.",
    },
}

# Domain Specialty Mappings (fallback for uncurated diseases)
SPECIALTY_MAP = {
    "Respiratory": ("Pulmonologist / Chest Specialist", "High"),
    "Cardiovascular": ("Cardiologist", "Critical"),
    "Neurological": ("Neurologist", "High"),
    "Dermatology": ("Dermatologist", "Moderate"),
    "Gastrointestinal": ("Gastroenterologist", "Moderate"),
    "Kidney/Urinary": ("Urologist / Nephrologist", "Moderate"),
    "Endocrine": ("Endocrinologist", "Moderate"),
    "Infectious": ("Infectious Disease Specialist", "High"),
    "ENT": ("ENT Specialist (Otolaryngologist)", "Moderate"),
    "Eye": ("Ophthalmologist", "Moderate"),
    "Autoimmune": ("Rheumatologist", "High"),
    "General": ("General Physician", "Low"),
}


def _categorize_disease(disease_name):
    """Determine category, specialist, and severity from disease name keywords."""
    d_lower = disease_name.lower()
    cat = "General Medicine"
    spec = "General Physician"
    sev = "Moderate"

    if any(w in d_lower for w in ["asthma", "pneumonia", "bronchitis", "copd", "tuberculosis", "pleurisy", "cough", "croup", "covid", "bronchiectasis", "lung", "sleep apnea"]):
        cat, (spec, sev) = "Respiratory", SPECIALTY_MAP["Respiratory"]
    elif any(w in d_lower for w in ["heart", "hypertension", "angina", "thrombosis", "carditis", "aneurysm", "varicose", "fibrillation", "embolism", "raynaud", "congestive", "myocarditis"]):
        cat, (spec, sev) = "Cardiovascular", SPECIALTY_MAP["Cardiovascular"]
    elif any(w in d_lower for w in ["migraine", "epilepsy", "meningitis", "palsy", "headache", "tunnel", "sciatica", "neuralgia", "spondylosis", "vertigo", "paralysis", "parkinson", "alzheimer", "neuropathy", "concussion"]):
        cat, (spec, sev) = "Neurological", SPECIALTY_MAP["Neurological"]
    elif any(w in d_lower for w in ["fungal", "acne", "psoriasis", "impetigo", "eczema", "cellulitis", "urticaria", "shingles", "rosacea", "scabies", "dermatitis", "ringworm", "athletes", "vitiligo", "melanoma", "carcinoma", "folliculitis", "tinea", "seborrheic"]):
        cat, (spec, sev) = "Skin & Dermatology", SPECIALTY_MAP["Dermatology"]
    elif any(w in d_lower for w in ["gerd", "cholestasis", "ulcer", "gastroenteritis", "jaundice", "hepatitis", "bowel", "appendicitis", "gallstones", "pancreatitis", "celiac", "poisoning", "gastritis", "piles", "crohn", "colitis", "diverticulitis", "lactose", "hemorrhagic"]):
        cat, (spec, sev) = "Gastroenterology", SPECIALTY_MAP["Gastrointestinal"]
    elif any(w in d_lower for w in ["urinary", "kidney", "nephrotic", "bladder", "prostatitis", "pyelonephritis", "cystitis", "prostatic"]):
        cat, (spec, sev) = "Urology & Nephrology", SPECIALTY_MAP["Kidney/Urinary"]
    elif any(w in d_lower for w in ["diabetes", "thyroid", "hypoglycemia", "cushing", "addison", "pcos", "gout", "vitamin d", "type 1", "parathyroid"]):
        cat, (spec, sev) = "Endocrinology", SPECIALTY_MAP["Endocrine"]
    elif any(w in d_lower for w in ["malaria", "dengue", "typhoid", "aids", "chicken pox", "measles", "mumps", "rubella", "mononucleosis", "lyme", "tetanus", "cholera", "hand foot", "scarlet", "diphtheria", "chikungunya", "zika", "viral fever", "covid", "norovirus", "herpes"]):
        cat, (spec, sev) = "Infectious Diseases", SPECIALTY_MAP["Infectious"]
    elif any(w in d_lower for w in ["otitis", "tonsillitis", "meniere", "tinnitus", "polyps", "septum", "neuritis", "labyrinthitis", "epiglottitis", "abscess", "laryngitis", "pharyngitis"]):
        cat, (spec, sev) = "ENT (Ear, Nose, Throat)", SPECIALTY_MAP["ENT"]
    elif any(w in d_lower for w in ["conjunctivitis", "glaucoma", "stye", "dry eye", "uveitis", "blepharitis", "corneal", "orbital"]):
        cat, (spec, sev) = "Ophthalmology", SPECIALTY_MAP["Eye"]
    elif any(w in d_lower for w in ["arthritis", "lupus", "sclerosis", "sjogren", "spondylitis", "fibromyalgia"]):
        cat, (spec, sev) = "Rheumatology & Autoimmune", SPECIALTY_MAP["Autoimmune"]

    return cat, spec, sev


def build_disease_info():
    from data.prepare_dataset import DISEASE_SYMPTOMS

    records = []
    for disease in sorted(DISEASE_SYMPTOMS.keys()):
        # Check if we have curated data for this disease
        if disease in CURATED_DISEASES:
            info = CURATED_DISEASES[disease]
            records.append({
                "disease": disease,
                "category": info["category"],
                "severity": info["severity"],
                "specialist": info["specialist"],
                "description": info["description"],
                "causes": info["causes"],
                "risk_factors": info["risk_factors"],
                "prevention": info["prevention"],
                "recovery_time": info.get("recovery_time", "Varies — consult specialist"),
                "emergency_signs": info.get("emergency_signs", "High fever >102°F, severe pain, breathing difficulty, confusion."),
                "contagious": info.get("contagious", "Consult your doctor"),
                "self_care": info.get("self_care", "Consult your doctor"),
                "diet": info.get("diet", "Balanced nutritious diet, adequate water intake, avoid heavy/processed foods."),
                "aliases": f"{disease.lower()}, {disease.lower().replace(' ', '-')}",
            })
        else:
            # Fallback: auto-generate from category
            cat, spec, sev = _categorize_disease(disease)
            records.append({
                "disease": disease,
                "category": cat,
                "severity": sev,
                "specialist": spec,
                "description": f"{disease} is a medical condition affecting the {cat.lower()} system requiring proper evaluation and treatment by a {spec}.",
                "causes": "Multiple factors including genetic predisposition, infections, environmental factors, and lifestyle factors.",
                "risk_factors": "Weakened immune system, age, family history, chronic stress, poor diet, exposure to pathogens.",
                "prevention": f"Maintain hygiene, get vaccinated where applicable, undergo regular health checkups. Consult a {spec}.",
                "recovery_time": "Varies — consult specialist for accurate timeline.",
                "emergency_signs": "High persistent fever, severe chest pain, shortness of breath, sudden confusion, difficulty breathing.",
                "contagious": "Consult your doctor",
                "self_care": "Consult your doctor for personalized guidance.",
                "diet": "Hydration, balanced nutrient-dense diet, low sodium, avoid processed foods.",
                "aliases": f"{disease.lower()}, {disease.lower().replace(' ', '-')}",
            })

    return pd.DataFrame(records)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "disease_info.csv")
    df_info = build_disease_info()
    df_info.to_csv(out_path, index=False)
    print(f"Compiled {out_path} with {len(df_info)} disease entries.")
    print(f"  Curated: {sum(1 for d in df_info['disease'] if d in CURATED_DISEASES)} diseases")
    print(f"  Auto-generated: {sum(1 for d in df_info['disease'] if d not in CURATED_DISEASES)} diseases")


if __name__ == "__main__":
    main()
