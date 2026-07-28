# 🏗️ System Architecture & Design Documentation

## 📌 Architectural Layers

```
                       +-------------------------------+
                       |   Streamlit Web Interface     |
                       |  (app/app.py & app/pages/)    |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Conversational NLP Engine   |
                       | (utils/helpers.py & src/nlp/) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Feature Vector Transformer  |
                       |     (preprocessing/ & models/)|
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       | Calibrated Multi-Class Classifier|
                       |     (models/calibrated_nb.pkl)|
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |  Differential Diagnosis Engine|
                       |    (Top-3 Confidence Ranks)   |
                       +-------------------------------+
```

### 1. Presentation Layer (`app/`)
- Streamlit application entry point (`app/app.py`)
- Multi-page application architecture (`app/pages/`)
- Custom CSS & Favicon assets (`app/assets/`)

### 2. NLP & Processing Layer (`src/nlp/` & `utils/helpers.py`)
- Tokenizer & Stop-word removal
- Hinglish vocabulary mapping (~70 Hindi medical terms)
- Fuzzy spelling correction engine (~60 common misspellings)

### 3. Model Inference Layer (`models/` & `src/prediction/`)
- `DiseasePredictor` OOP inference wrapper
- Probability calibration (Sigmoid curves)
- Clinical subclass resolution rules (e.g. UTI ↔ Cystitis)
