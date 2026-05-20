import os
import streamlit as st

_CSS_FILES = [
    "styles/component.css",
    "styles/tabs.css",
    "styles/form.css",
]

@st.cache_data
def _get_css() -> str:
    css = ""
    base = os.path.dirname(os.path.dirname(__file__))
    for rel in _CSS_FILES:
        path = os.path.join(base, rel)
        try:
            with open(path, encoding="utf-8") as f:
                css += f.read() + "\n"
        except FileNotFoundError:
            pass
    return css


def load_all_css():
    st.markdown(f"<style>{_get_css()}</style>", unsafe_allow_html=True)