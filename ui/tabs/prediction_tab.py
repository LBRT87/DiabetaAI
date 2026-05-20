import streamlit as st
import plotly.graph_objects as go

from services.model_loader import load_models
from services.prediction_service import predict_diabetes


def render_prediction_tab():
    model, scaler, imputer, threshold = load_models()

    desc = "Enter the patient's medical parameters below to perform an early diabetes screening, Fields marked <strong>◆</strong> accept 0 if data is unavailable — the system will automatically impute using dataset medians."

    st.markdown(f'<div class="sec-head"><h2>Risk Prediction</h2><p>{desc}</p></div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<span class="form-section-title">General Information</span>', unsafe_allow_html=True)
            age            = st.number_input("Age (years)",              21,  120, 30,   help="Minimum age 21.")
            pregnancies    = st.number_input("Number of Pregnancies",     0,   20,  0,   help="0 is valid for nulliparous patients.")
            glucose        = st.number_input("Glucose Level ◆ (mg/dL)",  0,  300, 120,  help="Plasma glucose 2h post OGTT. Enter 0 to auto-impute.")
            blood_pressure = st.number_input("Blood Pressure ◆ (mm Hg)", 0,  200,  70,  help="Diastolic BP. Enter 0 to auto-impute.")

        with col2:
            st.markdown('<span class="form-section-title">Clinical Parameters</span>', unsafe_allow_html=True)
            bmi            = st.number_input("BMI ◆",                    0.0, 70.0, 25.0, step=0.1, help="Enter 0 to auto-impute.")
            insulin        = st.number_input("Insulin ◆ (IU/mL)",        0,  1000,    0,  help="2h serum insulin. Enter 0 to auto-impute.")
            skin_thickness = st.number_input("Skin Thickness ◆ (mm)",    0,   100,    0,  help="Triceps skinfold. Enter 0 to auto-impute.")
            dpf            = st.number_input("Diabetes Pedigree Function",0.0, 3.0, 0.47, step=0.01, help="Family history diabetes score.")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze Diabetes Risk →")

    if submitted:
        if model is None:
            st.error("Model not found. Please run `python train.py` first.")
            return

        input_data = dict(age=age, pregnancies=pregnancies, glucose=glucose,
                          blood_pressure=blood_pressure, bmi=bmi, insulin=insulin,
                          skin_thickness=skin_thickness, dpf=dpf)

        with st.spinner("Running prediction…"):
            prediction, probability, imputed = predict_diabetes(
                model=model, scaler=scaler, imputer=imputer,
                threshold=threshold, input_data=input_data,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#6B7280;margin-bottom:1rem">Analysis Result</p>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns([1.1, 1.4, 1], gap="medium")

        with rc1:
            if prediction == 1:
                st.markdown(f'<div class="result-card result-high"><div class="result-badge">Prediction</div><div class="result-label">High Risk</div><div class="result-prob">Probability: {probability*100:.1f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card result-low"><div class="result-badge">Prediction</div><div class="result-label">Low Risk</div><div class="result-prob">Probability: {probability*100:.1f}%</div></div>', unsafe_allow_html=True)

        with rc2:
            color = "#B84040" if prediction == 1 else "#3D6B35"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 1),
                number={"suffix": "%", "font": {"size": 32, "family": "DM Sans", "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9CA3AF",
                             "tickfont": {"color": "#9CA3AF", "size": 10}},
                    "bar":  {"color": color, "thickness": 0.2},
                    "bgcolor": "#F9FAFB",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, threshold*100],   "color": "#F0FDF4"},
                        {"range": [threshold*100, 100], "color": "#FEF2F2"},
                    ],
                    "threshold": {
                        "line": {"color": "#374151", "width": 2},
                        "thickness": 0.85,
                        "value": threshold * 100
                    }
                },
                title={"text": f"Diabetes Probability<br><span style='font-size:.78em;color:#9CA3AF'>Decision threshold: {threshold*100:.0f}%</span>",
                       "font": {"color": "#374151", "size": 12, "family": "DM Sans"}}
            ))
            fig.update_layout(
                height=240, margin=dict(t=55, b=0, l=16, r=16),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#111827"
            )
            st.plotly_chart(fig, use_container_width=True)

        with rc3:
            st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#3D6B35;margin-bottom:0.8rem">Input Summary</p>', unsafe_allow_html=True)
            rows = {
                "Age":            f"{age} yr",
                "Pregnancies":    f"{pregnancies}×",
                "Glucose":        f"{imputed['Glucose'].values[0]:.1f} mg/dL",
                "Blood Pressure": f"{imputed['BloodPressure'].values[0]:.1f} mmHg",
                "BMI":            f"{imputed['BMI'].values[0]:.1f}",
                "Insulin":        f"{imputed['Insulin'].values[0]:.1f} IU/mL",
                "DPF":            f"{dpf:.2f}",
            }
            for k, v in rows.items():
                st.markdown(
                    f'<div class="summary-row"><span class="summary-key">{k}</span><span class="summary-val">{v}</span></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.error("**Clinical Recommendation:** Please consult a specialist for formal blood glucose testing (FPG / 2hPG / HbA1c) to confirm this screening result.")
        else:
            st.success("**Lifestyle Recommendation:** Maintain a balanced diet, limit added sugar, and aim for ≥150 min/week of moderate physical activity. Schedule periodic blood glucose checks as a preventive measure.")

        st.caption("❕ This tool is intended for early screening only and does not replace a clinical diagnosis.")