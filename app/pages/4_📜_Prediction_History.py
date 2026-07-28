"""
Page 4 — Prediction History Logs
View and export past prediction sessions.
"""

import streamlit as st
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import styled_metric_card
from utils.helpers import load_prediction_history, clear_prediction_history

_CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📜 Prediction History Logs")
st.markdown("Review and export past automated diagnosis sessions.")

history = load_prediction_history()

if not history:
    st.info("No prediction history recorded yet. Run a prediction on the home page.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.markdown(styled_metric_card("Total Sessions Logged", str(len(history)), color="#2B6CB0"), unsafe_allow_html=True)
with c2:
    latest = history[0].get("timestamp", "N/A")[:16] if history else "N/A"
    st.markdown(styled_metric_card("Latest Session", latest, color="#38A169"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

for rec in history:
    ts = rec.get("timestamp", "N/A")
    syms = rec.get("symptoms", [])
    preds = rec.get("predictions", [])
    top = preds[0] if preds else {"disease": "N/A", "probability": 0}

    with st.expander(f"🗓️ {ts} — Result: {top['disease']} ({top['probability']}%)"):
        st.markdown(f"**Symptoms:** {', '.join(syms)}")
        for i, p in enumerate(preds[:5]):
            st.markdown(f"{i+1}. **{p['disease']}** ({p['probability']}%)")

st.divider()

col_dl, col_clear = st.columns([3, 1])
with col_dl:
    flat = []
    for r in history:
        preds = r.get("predictions", [])
        top_d = preds[0]["disease"] if preds else "N/A"
        top_p = preds[0]["probability"] if preds else 0
        flat.append({
            "Timestamp": r.get("timestamp"),
            "Symptoms": ", ".join(r.get("symptoms", [])),
            "Diagnosis": top_d,
            "Confidence (%)": top_p,
        })
    df_h = pd.DataFrame(flat)
    st.download_button("📥 Export History CSV", df_h.to_csv(index=False).encode('utf-8'), "prediction_history.csv", "text/csv")

with col_clear:
    if st.button("🗑️ Clear History"):
        clear_prediction_history()
        st.success("History cleared.")
        st.rerun()
