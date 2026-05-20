import streamlit as st
from PIL import Image 

icon = Image.open("assets/Logo.png")
st.set_page_config(
    page_title="DiabetaAI — Diabetes Risk Prediction",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    "Risk Prediction",
    "Model Performance",
    "Data Insights",
    "User Testing",
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