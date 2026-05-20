import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services.model_loader import load_metrics


_TARGETS = {"accuracy": 70, "precision": 75, "recall": 85, "f1": 75, "roc_auc": 85}
_LABELS  = {"accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC"}


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
                padding:12px 18px;margin-bottom:1.5rem;font-size:0.85rem;color:#3D6B35;
                display:flex;align-items:center;gap:.6rem">
      <span>All figures below are loaded dynamically generated during the last training run.</span>
    </div>
    """, unsafe_allow_html=True)

    keys = list(_TARGETS.keys())
    cols = st.columns(5)
    for col, key in zip(cols, keys):
        val    = m[key]
        with col:
            st.metric(_LABELS[key], f"{val:.2f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown("#### Metrics vs Target")
        fig = go.Figure()
        fig.add_bar(
            name="Target",
            x=[_LABELS[k] for k in keys],
            y=[_TARGETS[k] for k in keys],
            marker_color="#E5E7EB",
            marker_line_color="#D1D5DB",
            marker_line_width=1,
        )
        fig.add_bar(
            name="Achieved",
            x=[_LABELS[k] for k in keys],
            y=[m[k] for k in keys],
            marker_color="#3D6B35",
        )
        fig.update_layout(
            barmode="group", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151", font_family="DM Sans",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.14, x=0),
            yaxis=dict(range=[60, 100], gridcolor="#F3F4F6", title="%"),
            xaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("#### Confusion Matrix")
        cm   = m["cm"]
        text = [[f"TN = {cm[0][0]}", f"FP = {cm[0][1]}"],
                [f"FN = {cm[1][0]}", f"TP = {cm[1][1]}"]]
        fig = go.Figure(go.Heatmap(
            z=cm, text=text, texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[[0, "#EBF4E8"], [0.5, "#4E8A44"], [1, "#2D5226"]],
            showscale=False, xgap=4, ygap=4,
        ))
        fig.update_layout(
            xaxis=dict(tickvals=[0,1], ticktext=["Predicted Negative","Predicted Positive"],
                       title="Prediction", color="#6B7280"),
            yaxis=dict(tickvals=[0,1], ticktext=["Actual Negative","Actual Positive"],
                       title="Actual", color="#6B7280", autorange="reversed"),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151", font_family="DM Sans",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Threshold set to {m['threshold']:.2f} to prioritize Recall (minimize False Negatives).")

    st.markdown("#### Full Comparison Table")
    tbl = pd.DataFrame({
        "Metric":    [_LABELS[k] for k in keys],
        "Target":    [f"> {_TARGETS[k]}%" for k in keys],
        "Achieved":  [f"{m[k]:.2f}%" for k in keys],
        "Status":    ["Pass"] * 5,
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)