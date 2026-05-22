import os
import json
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_models():
    try:
        with open(os.path.join(BASE_DIR, "model.pkl"),   "rb") as f: model   = pickle.load(f)
        with open(os.path.join(BASE_DIR, "scaler.pkl"),  "rb") as f: scaler  = pickle.load(f)
        with open(os.path.join(BASE_DIR, "imputer.pkl"), "rb") as f: imputer = pickle.load(f)
        threshold = 0.5
        tp = os.path.join(BASE_DIR, "threshold.txt")
        if os.path.exists(tp):
            with open(tp) as f:
                threshold = float(f.read().strip())
        return model, scaler, imputer, threshold
    except Exception as e:
        st.warning(f"Model files not found: {e}") 
        return None, None, None, 0.5


@st.cache_data
def load_metrics() -> dict:
    path = os.path.join(BASE_DIR, "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"metrics.json not found at {path}: {e}")  # ← debug path
        return {
            "model_name": "Random Forest",
            "threshold":  0.50,   
            "accuracy":   76.62,
            "precision":  63.24,
            "recall":     79.63,
            "f1":         70.49,
            "roc_auc":    82.67,
            "cm":         [[75, 25], [11, 43]],
        }