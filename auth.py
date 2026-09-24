import streamlit as st
import sqlite3
import bcrypt
import os
from datetime import datetime

DB_PATH = "users.db"


def init_db():
    """Create the users table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, password_hash):
    """Check a plain password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(username, email, password):
    """Register a new user. Returns (success, message)."""
    username = username.strip().lower()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email:
        return False, "Please enter a valid email."

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists."

    c.execute(
        "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (username, email, hash_password(password), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return True, "Account created. Please log in."


# Hardcoded fallback users — survive Streamlit Cloud redeploys
FALLBACK_USERS = {
    "bob": {"password": "password123", "email": "bob123@gmail.com"},
    "admin": {"password": "admin123", "email": "admin@provenance.internal"},
    "demo": {"password": "demo123", "email": "demo@provenance.internal"},
}


def authenticate(username, password):
    """Check username and password. Returns (success, email)."""
    username = username.strip().lower()

    # 1. Try the hardcoded fallback first — always works
    if username in FALLBACK_USERS:
        if password == FALLBACK_USERS[username]["password"]:
            return True, FALLBACK_USERS[username]["email"]
        else:
            return False, None

    # 2. Fall back to SQLite for user-created accounts
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash, email FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        return False, None
    if verify_password(password, row[0]):
        return True, row[1]
    return False, None


def get_user_profile(username):
    """Return the user's stored profile (name, email, title)."""
    if "user_profiles" not in st.session_state:
        st.session_state.user_profiles = {}

    if username not in st.session_state.user_profiles:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        st.session_state.user_profiles[username] = {
            "full_name": username,
            "email": row[0] if row else "",
            "title": ""
        }
    return st.session_state.user_profiles[username]


def save_user_profile(username, full_name, email, title):
    """Save profile updates in session (not shared across users)."""
    if "user_profiles" not in st.session_state:
        st.session_state.user_profiles = {}
    st.session_state.user_profiles[username] = {
        "full_name": full_name,
        "email": email,
        "title": title
    }


def is_logged_in():
    return st.session_state.get("auth_user") is not None


def current_user():
    return st.session_state.get("auth_user")


def login_user(username, email):
    st.session_state.auth_user = username
    st.session_state.auth_email = email


def logout_user():
    st.session_state.auth_user = None
    st.session_state.auth_email = None
    st.session_state.messages = []
    st.session_state.current_chat_id = None


def render_login_page():
    """Render the login and signup page. Returns True if user is authenticated."""
    init_db()

    if is_logged_in():
        return True

    st.markdown("""
    <style>
        .auth-card {
            background: #ffffff;
            border: 1px solid #e0e0d6;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            max-width: 420px;
            margin: 40px auto;
        }
        .auth-title {
            font-size: 28px;
            font-weight: 800;
            color: #1a1c19;
            text-align: center;
            margin-bottom: 8px;
        }
        .auth-subtitle {
            font-size: 14px;
            color: #4a4d47;
            text-align: center;
            margin-bottom: 24px;
        }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])

    with col_c:
        st.markdown('<div class="auth-title">🧠 Provenance</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Source-cited RAG for trustworthy document analysis</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Log In", use_container_width=True)

            if submitted:
                ok, email = authenticate(u, p)
                if ok:
                    login_user(u.strip().lower(), email)
                    st.success("Logged in.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                nu = st.text_input("Choose a Username", key="signup_user")
                ne = st.text_input("Email", key="signup_email")
                np = st.text_input("Password (min 6 chars)", type="password", key="signup_pass")
                np2 = st.text_input("Confirm Password", type="password", key="signup_pass2")
                submitted2 = st.form_submit_button("Create Account", use_container_width=True)

            if submitted2:
                if np != np2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(nu, ne, np)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    return False