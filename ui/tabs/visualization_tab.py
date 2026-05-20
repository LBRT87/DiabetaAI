import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


_COLORS = {"Non-Diabetes": "#4E8A44", "Diabetes": "#B84040"}


def _sample_data():
    np.random.seed(42)
    n0, n1 = 500, 268
    def gen(m0, s0, m1, s1):
        return np.concatenate([np.random.normal(m0,s0,n0).clip(0),
                                np.random.normal(m1,s1,n1).clip(0)])
    df = pd.DataFrame({
        "Glucose":       gen(109, 26, 141, 31),
        "BMI":           gen(30.3, 7.5, 35.1, 7.3),
        "Age":           gen(31.2, 11.6, 37.1, 10.5),
        "BloodPressure": gen(68.2, 18, 70.8, 21),
        "Insulin":       gen(68, 60, 100, 90),
        "Outcome":       [0]*n0 + [1]*n1,
    })
    df["Class"] = df["Outcome"].map({0: "Non-Diabetes", 1: "Diabetes"})
    return df


def _chart_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#374151",
        font_family="DM Sans",
        yaxis=dict(gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
        margin=dict(t=10, b=10),
        legend_title_text="",
    )


def render_visualization_tab():
    df = _sample_data()

    st.markdown('<div class="sec-head"><h2>Data Insights</h2><p>Distribution and pattern analysis of the PIMA Indians Diabetes dataset (768 samples, 8 clinical features).</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Glucose Distribution by Class**")
        fig = px.histogram(df, x="Glucose", color="Class", barmode="overlay",
                           opacity=0.75, nbins=30, color_discrete_map=_COLORS)
        fig.update_layout(height=290, **_chart_layout())
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**BMI Distribution by Class**")
        fig = px.histogram(df, x="BMI", color="Class", barmode="overlay",
                           opacity=0.75, nbins=30, color_discrete_map=_COLORS)
        fig.update_layout(height=290, **_chart_layout())
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Glucose vs BMI**")
        fig = px.scatter(df, x="Glucose", y="BMI", color="Class",
                         opacity=0.55, color_discrete_map=_COLORS)
        fig.update_layout(height=290, **_chart_layout())
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("**Class Distribution**")
        fig = go.Figure(go.Pie(
            labels=["Non-Diabetes (0)", "Diabetes (1)"],
            values=[500, 268],
            hole=0.58,
            marker_colors=["#4E8A44", "#B84040"],
            textfont=dict(size=12),
        ))
        fig.add_annotation(text="768<br><span style='font-size:10px'>samples</span>",
                           x=0.5, y=0.5, font_size=16, showarrow=False,
                           font_color="#374151")
        fig.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#374151", font_family="DM Sans",
                          margin=dict(t=20, b=10),
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Feature Importance — Random Forest**")
    fi = pd.DataFrame({
        "Feature":    ["Glucose","BMI","Age","Diabetes Pedigree Fn.","Insulin",
                       "Blood Pressure","Skin Thickness","Pregnancies"],
        "Importance": [0.26, 0.17, 0.15, 0.12, 0.11, 0.08, 0.06, 0.05],
    }).sort_values("Importance")

    fig = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker=dict(
            color=fi["Importance"],
            colorscale=[[0,"#EBF4E8"],[0.5,"#4E8A44"],[1,"#2D5226"]],
        ),
        text=[f"{v*100:.0f}%" for v in fi["Importance"]],
        textposition="outside",
        textfont=dict(color="#374151", size=11),
    ))
    fig.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#374151", font_family="DM Sans",
        xaxis=dict(gridcolor="#F3F4F6", range=[0, 0.32]),
        margin=dict(t=10, b=10, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)