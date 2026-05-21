import streamlit as st
import subprocess
import os
import sys

st.set_page_config(
    page_title="DiabetaAI — Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_FILES = ["model.pkl", "scaler.pkl", "imputer.pkl", "metrics.json"]

if not all(os.path.exists(f) for f in MODEL_FILES):
    with st.spinner("⏳ Setting up model for first time... please wait (1-2 min)"):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "scikit-learn", "imbalanced-learn", "xgboost",
                 "pandas", "numpy"],
                capture_output=True, check=True
            )
            result = subprocess.run(
                [sys.executable, "model_training.py"],
                capture_output=True, text=True, check=True
            )
            st.success("✅ Model ready!")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Training failed: {e.stderr}")
            st.stop()

from services.css_loader import load_all_css
load_all_css()

from ui.header import render_header
from ui.footer import render_footer
from ui.tabs.prediction_tab import render_prediction_tab
from ui.tabs.performance_tab import render_performance_tab
from ui.tabs.visualization_tab import render_visualization_tab
from ui.tabs.feedback_tab import render_feedback_tab

render_header()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬  Risk Prediction",
    "📊  Model Performance",
    "📈  Data Insights",
    "📋  User Testing",
])

with tab1:
    render_prediction_tab()

with tab2:
    render_performance_tab()

with tab3:
    render_visualization_tab()

with tab4:
    render_feedback_tab()

render_footer()