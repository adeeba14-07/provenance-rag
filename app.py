import streamlit as st
from tracker import (
    load_stats,
    get_average_retrieval_time,
    get_average_generation_time,
    get_average_trust_score
)
from settings import load_settings as load_user_settings

st.set_page_config(
    page_title="Provenance RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load theme settings
user_settings = load_user_settings()
theme = user_settings.get("theme", "Soft Dark")

if theme == "Cool Light":
    bg_color = "#F8FAFC"
    card_color = "#FFFFFF"
    sidebar_color = "#F1F5F9"
    text_primary = "#0F172A"
    text_secondary = "#475569"
    accent_blue = "#2563EB"
    accent_green = "#059669"
    accent_orange = "#D97706"
    border_color = "#E2E8F0"
elif theme == "Warm Sepia":
    bg_color = "#FAF5EF"
    card_color = "#FFFBF5"
    sidebar_color = "#F5EDE0"
    text_primary = "#3E2C1C"
    text_secondary = "#6B5B4A"
    accent_blue = "#8B5E3C"
    accent_green = "#5F7A3C"
    accent_orange = "#C07A3C"
    border_color = "#E5D5C0"
else:  # Soft Dark
    bg_color = "#0F172A"
    card_color = "#1E293B"
    sidebar_color = "#0B1120"
    text_primary = "#F1F5F9"
    text_secondary = "#94A3B8"
    accent_blue = "#3B82F6"
    accent_green = "#10B981"
    accent_orange = "#F59E0B"
    border_color = "#334155"

# Apply CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --bg-dark: {bg_color};
        --bg-card: {card_color};
        --bg-sidebar: {sidebar_color};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --accent-blue: {accent_blue};
        --accent-green: {accent_green};
        --accent-orange: {accent_orange};
        --border: {border_color};
    }}

    .stApp {{
        background-color: var(--bg-dark);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }}

    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-primary);
    }}

    .provenance-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }}

    div[data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
    }}

    .stButton>button {{
        background: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background: #2563EB;
        transform: translateY(-1px);
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
        font-weight: 700;
    }}

    p, li, span, label {{
        color: var(--text-secondary);
    }}
</style>
""", unsafe_allow_html=True)

# Load real stats
stats = load_stats()

# Sidebar
with st.sidebar:
    st.markdown("# 🧠 Provenance")
    st.markdown("### RAG Research Platform")
    st.markdown("---")
    st.markdown("**Status:** 🟢 System Online")
    st.markdown("**Model:** GPT-OSS-120B")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("---")
    st.markdown(f"**Last Updated:** {stats['last_updated'] or 'Never'}")

# Main Page
st.title("Provenance RAG")
st.markdown("### Research Platform")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents Uploaded", stats["documents_uploaded"])

with col2:
    st.metric("Total Chunks Created", stats["total_chunks"])

with col3:
    st.metric("Total Queries Asked", stats["total_queries"])

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg Retrieval Time", f"{get_average_retrieval_time()} ms")

with col2:
    st.metric("Avg Generation Time", f"{get_average_generation_time()} ms")

with col3:
    st.metric("Avg Trust Score", f"{get_average_trust_score()}/100")

st.markdown("---")

st.markdown("### Quick Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="provenance-card">📄 **Upload Documents**<br>Go to Ingestion page to upload your research corpus.</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="provenance-card">💬 **Start Chatting**<br>Go to Chat page to ask questions and get source-cited answers.</div>', unsafe_allow_html=True)