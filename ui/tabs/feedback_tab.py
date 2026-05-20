import streamlit as st

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdMbpGrq3RjS8QBViwdBt_b1HdSYSLNlj8MEHqzf1I0x5RaUw/viewform?usp=header"

_QUESTIONS = [
    ("01", "Ease of Use",
     "How easy was it to navigate and use the application without guidance?"),
    ("02", "Clarity of Results",
     "Were the prediction results and recommendations easy to understand?"),
    ("03", "Visual Design",
     "How would you rate the overall appearance and visual design?"),
    ("04", "Response Speed",
     "How fast did the application respond to your inputs?"),
    ("05", "Usefulness for Screening",
     "How useful do you find this tool for early diabetes screening?"),
    ("06", "Respondent Background",
     "What best describes your background?"),
    ("07", "Suggestions for Improvement",
     "What could be improved in the application?"),
    ("08", "Recommendation",
     "Would you recommend this application to others?"),
]


def render_feedback_tab():
    st.markdown('<div class="sec-head"><h2>User Feedback</h2></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="gform-card">
      <h3>Fill in the Questionnaire</h3>
      <p>Click the button below to open the questionnaire in Google Forms. All responses are automatically saved to our Google Sheets — no login required.<br><strong>Estimated time:</strong> 2–3 minutes.</p>
      <a class="gform-btn" href="{GOOGLE_FORM_URL}" target="_blank">Open Google Form &nbsp;↗</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#6B7280;margin:1.5rem 0 0.8rem">Question Preview</p>', unsafe_allow_html=True)

    for num, title, desc in _QUESTIONS:
        st.markdown(
            f'<div class="q-card"><div class="q-num">Question {num}</div>'
            f'<div class="q-title">{title}</div>'
            f'<div class="q-desc">{desc}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)