"""
Symptom Normalization Layer (src/normalization.py)
=================================================
Translates natural language text (including typos, synonyms, and Hinglish)
into canonical symptom names using dictionary mapping and RapidFuzz fuzzy matching.
"""

import os
import re
import json
from rapidfuzz import process, fuzz

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

class SymptomNormalizer:
    """
    Normalizes free-text symptom descriptions into canonical symptom column names.
    Uses:
    1. Regex-based exact keyword & synonym lookup (dict-based)
    2. RapidFuzz ratio fuzzy matching fallback (Levenshtein-based similarity)
    """

    def __init__(self, synonym_file=os.path.join(CONFIG_DIR, "synonyms.json"), fuzzy_cutoff=85.0):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.synonyms = {}
        if os.path.exists(synonym_file):
            with open(synonym_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.synonyms = data.get("synonyms", {})
        
        # Sort synonyms by length descending to match multi-word phrases first
        self.sorted_synonym_keys = sorted(self.synonyms.keys(), key=len, reverse=True)
        self._cache = {}

    def normalize_phrase(self, phrase, valid_symptoms):
        """
        Maps a single string phrase to a canonical symptom name.
        Returns canonical_symptom name or None.
        """
        phrase_clean = phrase.lower().strip()
        phrase_clean = re.sub(r"[^\w\s]", " ", phrase_clean)
        phrase_clean = re.sub(r"\s+", " ", phrase_clean).strip()

        if not phrase_clean:
            return None

        # 1. Exact match in valid_symptoms
        if phrase_clean in valid_symptoms:
            return phrase_clean

        # 2. Match via synonym dictionary
        if phrase_clean in self.synonyms:
            sym = self.synonyms[phrase_clean]
            if sym in valid_symptoms:
                return sym

        # 3. Fuzzy match using RapidFuzz against valid_symptoms and synonyms
        candidates = list(valid_symptoms) + list(self.synonyms.keys())
        match = process.extractOne(phrase_clean, candidates, scorer=fuzz.WRatio)
        if match and match[1] >= self.fuzzy_cutoff:
            matched_term = match[0]
            if matched_term in valid_symptoms:
                return matched_term
            elif matched_term in self.synonyms:
                sym = self.synonyms[matched_term]
                if sym in valid_symptoms:
                    return sym

        return None

    def extract_symptoms(self, text, valid_symptoms):
        """
        Parses full free-text input and returns a dictionary with:
        - matched: list of canonical symptom column names
        - unmatched: list of unmapped tokens
        - details: mapping details
        """
        if not text or not text.strip():
            return {"matched": [], "unmatched": []}

        cache_key = (text, tuple(valid_symptoms))
        if cache_key in self._cache:
            return self._cache[cache_key]

        text_clean = text.lower().replace("'", "").replace("\u2019", "")
        text_clean = re.sub(r"[^\w\s]", " ", text_clean)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()

        matched = set()
        remaining_text = text_clean

        # Step 1: Match multi-word dictionary synonyms (longest first)
        for phrase in self.sorted_synonym_keys:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, remaining_text):
                canonical = self.synonyms[phrase]
                if canonical in valid_symptoms:
                    matched.add(canonical)
                remaining_text = re.sub(pattern, " ", remaining_text)

        # Step 2: Token-level matching & Fuzzy fallback
        tokens = remaining_text.split()
        unmatched = []
        stop_words = {
            "i", "have", "with", "and", "a", "an", "the", "for", "since", "days", "day",
            "my", "me", "feel", "feeling", "is", "are", "very", "severe", "bad", "good",
            "well", "unwell", "today", "also", "had", "got", "get", "sick", "ill", "fine",
            "okay", "dont", "do", "not", "problem", "issue", "something", "going", "on",
            "now", "week", "feels", "total", "totally", "keeps", "keep", "getting", "having",
            "has", "been", "was", "were", "this", "like", "when"
        }
        
        for token in tokens:
            if token in stop_words or len(token) < 3:
                continue

            # Try exact match in valid_symptoms
            if token in valid_symptoms:
                matched.add(token)
                continue

            # Try de-pluralization (stemming fallback)
            stem = token[:-1] if token.endswith("s") else token
            if stem in valid_symptoms:
                matched.add(stem)
                continue

            # RapidFuzz fallback: Use ratio scorer with >= 85.0 threshold
            match = process.extractOne(token, list(valid_symptoms), scorer=fuzz.ratio)
            if match and match[1] >= 85.0:
                matched.add(match[0])
            else:
                unmatched.append(token)

        res = {
            "matched": sorted(list(matched)),
            "unmatched": unmatched
        }
        self._cache[cache_key] = res
        return res
