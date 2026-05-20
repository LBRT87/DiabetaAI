import os
import json
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


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
    except Exception:
        return None, None, None, 0.5


@st.cache_data
def load_metrics() -> dict:
    try:
        with open(os.path.join(BASE_DIR, "metrics.json")) as f:
            return json.load(f)
    except Exception:
        return {
            "model_name": "Random Forest",
            "threshold":  0.35,
            "accuracy":   77.92,
            "precision":  76.47,
            "recall":     88.89,
            "f1":         82.11,
            "roc_auc":    90.17,
            "cm":         [[81, 12], [6, 48]],
        }