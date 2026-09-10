import streamlit as st
from tracker import load_stats
from conversations import load_conversations
from settings import load_settings, save_settings
from theme import apply_theme

apply_theme()

st.set_page_config(page_title="Governance", page_icon="⚙️", layout="wide")

st.title("⚙️ Workspace Governance")
st.markdown("### Profile and pipeline configuration")

# Load current settings
settings = load_settings()
stats = load_stats()
conversations = load_conversations()

total_chats = 0
for doc, chats in conversations.items():
    total_chats += len(chats)

# ============================================
# PROFILE SECTION
# ============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 👤 Profile")
    
    full_name = st.text_input("Full Name", value=settings.get("full_name", ""))
    title = st.text_input("Professional Title", value=settings.get("title", ""))
    email = st.text_input("Email", value=settings.get("email", ""))

with col2:
    st.markdown("### 📊 Real Usage Stats")

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
st.markdown("### 🔧 RAG Pipeline Configuration")

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
st.markdown("### 🎨 Interface Customization")

col1, col2 = st.columns(2)

with col1:
    theme = st.radio(
        "Workspace Theme",
        ["Soft Dark", "Cool Light", "Warm Sepia"],
        index=["Soft Dark", "Cool Light", "Warm Sepia"].index(settings.get("theme", "Soft Dark"))
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
st.markdown("### 👁️ Preview")

preview_text = "The appellate court's interpretation of Chevron deference in this context significantly limits agency discretion when determining statutory ambiguity."
st.markdown(f'<p style="font-size:{font_size}px; color:#94A3B8;">{preview_text}</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SAVE BUTTON
# ============================================
if st.button("💾 Save All Changes", use_container_width=True):
    # Update settings
    settings["full_name"] = full_name
    settings["title"] = title
    settings["email"] = email
    settings["chunk_size"] = chunk_size
    settings["chunk_overlap"] = chunk_overlap
    settings["top_k"] = top_k
    settings["theme"] = theme
    settings["font_size"] = font_size

save_settings(settings)
st.success("✅ Settings saved successfully!")
st.rerun()