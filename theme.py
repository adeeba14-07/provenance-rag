import streamlit as st
from settings import load_settings as load_user_settings

def apply_theme():
    user_settings = load_user_settings()
    theme = user_settings.get("theme", "Soft Dark")

    if theme == "Cool Light":
        bg_color = "#F8FAFC"
        card_color = "#FFFFFF"
        sidebar_color = "#F1F5F9"
        text_primary = "#0F172A"
        text_secondary = "#475569"
        border_color = "#E2E8F0"
    elif theme == "Warm Sepia":
        bg_color = "#FAF5EF"
        card_color = "#FFFBF5"
        sidebar_color = "#F5EDE0"
        text_primary = "#3E2C1C"
        text_secondary = "#6B5B4A"
        border_color = "#E5D5C0"
    else:
        bg_color = "#0F172A"
        card_color = "#1E293B"
        sidebar_color = "#0B1120"
        text_primary = "#F1F5F9"
        text_secondary = "#94A3B8"
        border_color = "#334155"

    st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_primary};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_color};
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary};
    }}
    div[data-testid="stMetric"] {{
        background: {card_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 16px;
    }}
    .stButton>button {{
        background: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary};
        font-weight: 700;
    }}
    p, li, span, label {{
        color: {text_secondary};
    }}
</style>
""", unsafe_allow_html=True)