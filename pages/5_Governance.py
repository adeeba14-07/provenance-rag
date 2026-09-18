import streamlit as st
from tracker import load_stats
from conversations import load_conversations
from settings import load_settings, save_settings
from theme import apply_theme, render_navigation, render_page_header, render_top_bar
from auth import current_user, get_user_profile, save_user_profile

if not current_user():
    st.warning("Please log in first.")
    st.stop()

apply_theme()
render_navigation("pages/5_Governance.py")
render_top_bar("Workspace Governance")

st.set_page_config(page_title="Governance", page_icon="⌘", layout="wide")

render_page_header("Workspace governance", "Profile and controls", "Tune the research workspace while keeping the retrieval pipeline explicit and inspectable.")

# Load current settings
settings = load_settings()
stats = load_stats()
conversations = load_conversations()
user_profile = get_user_profile(current_user())

total_chats = 0
for doc, chats in conversations.items():
    total_chats += len(chats)

# ============================================
# PROFILE SECTION
# ============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Profile")
    st.caption(f"Logged in as `{current_user()}`")
    
    full_name = st.text_input("Full Name", value=user_profile.get("full_name", ""))
    title = st.text_input("Professional Title", value=user_profile.get("title", ""))
    email = st.text_input("Email", value=user_profile.get("email", ""))

with col2:
    st.markdown("### Real Usage Stats")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Docs Parsed", stats["documents_uploaded"])

    with m2:
        st.metric("Total Queries", stats["total_queries"])

    with m3:
        st.metric("Total Chats", total_chats)

    with m4:
        st.metric("Chunks Stored", stats["total_chunks"])

st.markdown("---")

# ============================================
# RAG PIPELINE CONFIGURATION
# ============================================
st.markdown("### RAG Pipeline Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    chunk_size = st.number_input(
        "Document Chunk Size (tokens)",
        min_value=100,
        max_value=5000,
        value=settings.get("chunk_size", 1000),
        step=100
    )

with col2:
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=settings.get("chunk_overlap", 200),
        step=50
    )

with col3:
    top_k = st.number_input(
        "Retrieval Top-K",
        min_value=1,
        max_value=20,
        value=settings.get("top_k", 8),
        step=1
    )

st.markdown("---")

# ============================================
# INTERFACE CUSTOMIZATION
# ============================================
st.markdown("### Interface Customization")

col1, col2 = st.columns(2)

with col1:
    theme = st.radio(
        "Workspace Theme",
        ["Soft Dark", "Warm Sepia"],
        index=["Soft Dark", "Warm Sepia"].index(settings.get("theme", "Soft Dark"))
    )

with col2:
    font_size = st.slider(
        "Base Font Size",
        min_value=10,
        max_value=20,
        value=settings.get("font_size", 14),
        step=1
    )

st.markdown("---")

# ============================================
# PREVIEW PANE
# ============================================
st.markdown("### Preview")

preview_text = "The appellate court's interpretation of Chevron deference in this context significantly limits agency discretion when determining statutory ambiguity."
st.markdown(f'<p style="font-size:{font_size}px; color:#94A3B8;">{preview_text}</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SAVE BUTTON
# ============================================
if st.button(" Save All Changes", use_container_width=True):
    # Profile: saved per-user (in session, never shared)
    save_user_profile(current_user(), full_name, email, title)

    # Pipeline settings: global (shared, non-personal)
    settings["chunk_size"] = chunk_size
    settings["chunk_overlap"] = chunk_overlap
    settings["top_k"] = top_k
    settings["theme"] = theme
    settings["font_size"] = font_size
    save_settings(settings)
    st.session_state.governance_saved = True

if st.session_state.get("governance_saved"):
    st.success("[OK] Settings saved successfully!")
    del st.session_state["governance_saved"]