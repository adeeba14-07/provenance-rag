import streamlit as st
from settings import load_settings as load_user_settings


def apply_theme():
    user_settings = load_user_settings()
    theme = user_settings.get("theme", "Soft Dark")

    if theme == "Warm Sepia":
        colors = {
            "background": "#faf5ef",
            "surface": "#fffbf5",
            "surface_low": "#f5ede0",
            "text": "#3e2c1c",
            "muted": "#6b5b4a",
            "outline": "#e5d5c0",
            "primary": "#5f472f",
        }
    else:
        colors = {
            "background": "#fafaf4",
            "surface": "#ffffff",
            "surface_low": "#f4f4ee",
            "text": "#1a1c19",
            "muted": "#494551",
            "outline": "#c3c8bd",
            "primary": "#1a3717",
        }

    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{
        --background: {colors['background']};
        --surface: {colors['surface']};
        --surface-low: {colors['surface_low']};
        --text: {colors['text']};
        --muted: {colors['muted']};
        --outline: {colors['outline']};
        --primary: {colors['primary']};
        --secondary: #bdefbc;
    }}
    html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; font-size: 14px; }}
    .stApp {{ background: var(--background); color: var(--text); }}
    /* Keep header visually minimal without hiding the sidebar controls. */
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    #MainMenu {{ visibility: hidden !important; }}

    /* Ensure the sidebar expand/collapse button is always clickable */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        width: 2.25rem !important;
        height: 2.25rem !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        width: 2.25rem !important;
        height: 2.25rem !important;
        margin: 0 !important;
    }}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebar"] button[aria-label="Close sidebar"],
[data-testid="stHeader"] button[aria-label="Open sidebar"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}}
    footer {{ display: none; }}
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > .main > div {{ padding-top: 0 !important; margin-top: 0 !important; }}
    [data-testid="stMainBlockContainer"] {{ padding: 0 1.5rem 3rem !important; }}
    [data-testid="stSidebar"] {{ background: var(--surface-low); border-right: 1px solid var(--outline); width: 256px !important; min-width: 256px !important; }}
    [data-testid="stSidebar"] > div {{ width: 256px !important; }}
    [data-testid="stSidebar"] > div:first-child {{ padding: 1rem; }}
    [data-testid="stSidebar"] * {{ color: var(--text); }}
    [data-testid="stSidebarNav"] {{ display: none !important; }}
    [data-testid="stSidebarContent"] {{ padding-top: 0.5rem; }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--text) !important; font-weight: 800 !important; letter-spacing: -0.02em; }}
    p, li, label, [data-testid="stCaptionContainer"] {{ color: var(--muted); }}
    div[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--outline); border-radius: 8px; padding: 1rem; box-shadow: none; }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: var(--text) !important; }}
    div[data-testid="stMetric"] label {{ white-space: normal; line-height: 1.2; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ font-family: 'Manrope', sans-serif; font-size: 1.8rem; }}
    .stButton>button {{ background: var(--primary); border: 1px solid var(--primary); border-radius: 4px; min-height: 2.5rem; padding: 0.5rem 1rem; font-weight: 600; font-family: 'Manrope', sans-serif; transition: background 150ms ease, transform 150ms ease; }}
    .stButton>button p, .stButton>button span, .stButton>button div {{ color: white !important; }}
    .stButton>button:hover {{ background: #304e2c; border-color: #304e2c; transform: translateY(-1px); }}
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stFileUploader"] section, [data-testid="stSelectbox"] [data-baseweb="select"] > div {{ background: var(--surface); border-color: var(--outline); color: var(--text); border-radius: 4px; }}
    [data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--outline); border-radius: 6px; }}
    [data-testid="stPlotlyChart"] {{ background: var(--surface); border: 1px solid var(--outline); border-radius: 8px; padding: 0.4rem; }}
    .provenance-panel {{ background: var(--surface); border: 1px solid var(--outline); border-radius: 6px; padding: 1rem 1.25rem; margin: 0.75rem 0; }}
    .stitch-topbar {{ display:flex; align-items:center; justify-content:space-between; min-height:4rem; padding:0 1.5rem; margin:0 -1.5rem 1.25rem; background:rgba(255,255,255,0.86); border-bottom:1px solid var(--outline); }}
    .stitch-topbar-title {{ color:var(--text); font-size:1.05rem; font-weight:800; }}
    .stitch-topbar-tools {{ display:flex; align-items:center; gap:0.5rem; color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.04em; }}
    .stitch-tool {{ width:2.25rem; height:2.25rem; display:flex; align-items:center; justify-content:center; border:1px solid var(--outline); border-radius:4px; background:var(--surface); }}
    [data-testid="stChatMessage"] {{ max-width: 860px; padding: 0.25rem 0; }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{ font-size: 1rem; line-height: 1.65; }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{ margin-left: auto; }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{ background: #1a3717; color: #ffffff; border: 1px solid #304e2c; border-radius: 1rem 0.25rem 1rem 1rem; padding: 0.9rem 1.1rem; box-shadow: 0 4px 12px rgba(26,55,23,0.14); }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{ background: #e3e3dd; color: #1a1c19; border: 1px solid #c3c8bd; border-radius: 0.25rem 1rem 1rem 1rem; padding: 0.9rem 1.1rem; }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {{ color: #ffffff !important; }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p {{ color: #1a1c19 !important; }}
    .stitch-source-card {{ background:#ffffff; border:1px solid #c3c8bd; border-radius:0.6rem; padding:0.9rem 1rem; margin-top:0.65rem; }}
    .stitch-chip-row {{ display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin:0.6rem 0 0.25rem; }}
    .stitch-chip-label {{ color:#494551; font:700 0.68rem 'JetBrains Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; margin-right:0.3rem; }}
</style>
""", unsafe_allow_html=True)


def render_navigation(active_page):
    """Render the Stitch navigation while preserving Streamlit page routing."""
    pages = [
        ("app.py", "Workspace", "⌂ "),
        ("pages/1_Chat.py", "Chat", "◌ "),
        ("pages/2_Ingestion.py", "Ingestion", "＋ "),
        ("pages/3_History.py", "History", "◷ "),
        ("pages/4_Analytics.py", "Analytics", "▥ "),
        ("pages/5_Governance.py", "Governance", "⌘ "),
        ("pages/6_Evaluation.py", "Evaluation", "⌁ "),
    ]
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.65rem;margin-bottom:1.5rem;">'
            '<div style="width:2rem;height:2rem;border-radius:0.35rem;background:#1a3717;'
            'color:white;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">◈</div>'
            '<div><div style="font-size:1rem;font-weight:800;color:#1a3717;line-height:1.1;">Provenance RAG</div></div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        for page_path, label, icon in pages:
            if page_path == active_page:
                st.markdown(
                    f'<div style="background:#bdefbc;color:#1a3717;padding:0.65rem 0.75rem;'
                    f'border-radius:4px;font-weight:700;margin:0.2rem 0;">{icon}&nbsp;&nbsp;{label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(page_path, label=f"{icon}  {label}")
        st.divider()
        st.caption("SYSTEM ONLINE")


def render_page_header(title, eyebrow, description):
    st.markdown(
        f'''<div class="provenance-panel" style="padding:1.35rem 1.5rem;">
        <div style="font-size:0.7rem;letter-spacing:0.14em;text-transform:uppercase;color:#494551;font-weight:700;">{eyebrow}</div>
        <h1 style="margin:0.25rem 0;font-size:2rem;">{title}</h1>
        <div style="color:#494551;font-size:0.9rem;">{description}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def render_top_bar(title):
    st.markdown(
        f'''<div class="stitch-topbar">
        <div class="stitch-topbar-title">{title}</div>
        <div class="stitch-topbar-tools"><span class="stitch-tool">◐</span><span class="stitch-tool">◌</span><span class="stitch-tool">⌁</span><span>● ONLINE</span></div>
        </div>''',
        unsafe_allow_html=True,
    )
