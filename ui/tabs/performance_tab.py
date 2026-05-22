import streamlit as st
import plotly.graph_objects as go

from services.model_loader import load_metrics

_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1-Score",
    "roc_auc":   "ROC-AUC",
}


def render_performance_tab():
    m = load_metrics()

    st.markdown(f"""
    <div class="sec-head">
      <h2>Model Performance</h2>
      <p>Evaluation results of the best model after GridSearchCV tuning and threshold optimization.
         Selected model: <strong>{m['model_name']}</strong> &nbsp;·&nbsp;
         Decision threshold: <strong>{m['threshold']:.2f}</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#F4FAF2;border:1px solid #C6E0BC;border-radius:14px;
                padding:12px 18px;margin-bottom:1.5rem;font-size:0.85rem;color:#3D6B35;">
      <span>All figures below are loaded dynamically from the last training run.</span>
    </div>
    """, unsafe_allow_html=True)

    keys = list(_LABELS.keys())
    cols = st.columns(5)
    for col, key in zip(cols, keys):
        with col:
            st.metric(_LABELS[key], f"{m[key]:.2f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("#### Confusion Matrix")
    cm = m["cm"]

    color_matrix = [
        [ cm[0][0], -cm[0][1]],
        [-cm[1][0],  cm[1][1]],
    ]
    text = [
        [f"TN = {cm[0][0]}<br>Correct",    f"FP = {cm[0][1]}<br>False Alarm"],
        [f"FN = {cm[1][0]}<br>Missed!",    f"TP = {cm[1][1]}<br>Correct"],
    ]

    fig = go.Figure(go.Heatmap(
        z=color_matrix,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale=[
            [0.0,  "#DC2626"],
            [0.45, "#F87171"],
            [0.5,  "#ffffff"],
            [0.55, "#4E8A44"],
            [1.0,  "#2D5226"],
        ],
        showscale=False,
        xgap=6, ygap=6,
    ))
    fig.update_layout(
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=["Predicted Negative", "Predicted Positive"],
            title="Prediction",
            color="#6B7280",
        ),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["Actual Negative", "Actual Positive"],
            title="Actual",
            color="#6B7280",
            autorange="reversed",
        ),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#374151",
        font_family="DM Sans",
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Threshold: {m['threshold']:.2f} · "
        f"The model correctly identified {cm[1][1]} diabetic cases "
        f"and only missed {cm[1][0]}."
    )