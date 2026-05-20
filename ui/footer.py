import streamlit as st


def render_footer():
    st.markdown("""
    <div class="footer-wrapper">
        <strong style="color:#374151">@2026 DiabetaAI</strong> &nbsp;&nbsp;
        <br>
        Airell Brandon Kho &nbsp;·&nbsp; 
        Elbert Joan &nbsp;·&nbsp;
        Glenn Nielsen &nbsp;·&nbsp; 
        Marcell Kurniawan
        <br>
    </div>
    """, unsafe_allow_html=True)