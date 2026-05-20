import os
import base64
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "Logo.png")


def _img_to_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def render_header():
    b64 = _img_to_b64(LOGO_PATH)

    if b64:
        logo_html = f'<img src="data:image/png;base64,{b64}" class="app-logo" alt="DiabetaAI Logo">'
    else:
        logo_html = """
        <div style="
            width:72px; height:72px;
            background: linear-gradient(135deg, #3D6B35, #4E8A44);
            border-radius: 20px;
            display: flex; align-items: center; justify-content: center;
            font-size: 2rem;
            box-shadow: 0 8px 24px rgba(61,107,53,0.22);
            margin: 0 auto 1rem;
        ">🩺</div>
        """

    st.markdown(f"""
    <style>
    .app-logo {{
        width: 80px;
        height: 80px;
        object-fit: contain;
        border-radius: 20px;
        margin: 0 auto 1rem;
        display: block;
        box-shadow: 0 8px 24px rgba(61,107,53,0.18);
    }}
    .app-header {{
        text-align: center;
        padding: 3rem 1rem 2.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .app-wordmark {{
        font-family: 'DM Serif Display', serif !important;
        font-size: 3.6rem;
        font-style: italic;
        color: #111827;
        line-height: 1;
        margin-bottom: 0.5rem;
    }}
    .app-tagline {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #3D6B35;
        margin-bottom: 1rem;
    }}
    .app-desc {{
        max-width: 560px;
        margin: 0 auto;
        color: #6B7280;
        font-size: 0.96rem;
        line-height: 1.75;
        text-align: center;
    }}
    </style>
    <div class="app-header">
        {logo_html}
        <div class="app-wordmark">DiabetaAI</div>
        <div class="app-tagline">Early Diabetes Risk Prediction System</div>
        <p class="app-desc">
            A supervised machine learning system designed to support early-stage diabetes screening 
            DiabetaAI analyzes eight clinical biomarkers to predict risk probability bridging the gap between raw medical data and actionable health insights.
        </p>
    </div>
    """, unsafe_allow_html=True)