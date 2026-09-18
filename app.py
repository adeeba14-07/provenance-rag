import streamlit as st

st.set_page_config(
    page_title="Provenance RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Target ONLY the text inside Streamlit buttons */
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #ffffff !important;
    }

    /* Optional: Ensure the button background stays dark brown */
    div.stButton > button {
        background-color: #5c4033 !important;
        border: 1px solid #5c4033 !important;
    }

    /* Optional: Ensure text stays white when hovering */
    div.stButton > button:hover p,
    div.stButton > button:hover span,
    div.stButton > button:hover div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)
from tracker import (
    load_stats,
    get_average_retrieval_time,
    get_average_generation_time,
    get_average_trust_score
)
from theme import apply_theme, render_navigation, render_page_header, render_top_bar


apply_theme()

# Load real stats
stats = load_stats()

render_navigation("app.py")
with st.sidebar:
    st.markdown("**Status:** 🟢 System Online")
    st.markdown("**Model:** GPT-OSS-120B")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("---")
    st.markdown(f"**Last Updated:** {stats['last_updated'] or 'Never'}")

# Main Page
render_top_bar("Workspace")
render_page_header("Provenance RAG", "Workspace overview", "A grounded research environment for ingesting, questioning, and auditing documents.")

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
    st.markdown('<div class="provenance-panel">▣ <b>Upload documents</b><br>Prepare a research corpus in the ingestion workspace.</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="provenance-panel">◌ <b>Start research chat</b><br>Ask questions and inspect source-cited answers.</div>', unsafe_allow_html=True)