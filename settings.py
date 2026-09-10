import json
import os

SETTINGS_FILE = "settings.json"

def load_settings():
    """Load user settings from JSON."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "full_name": "Your Name",
            "title": "AI Research Analyst",
            "email": "you@provenance.internal",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 8,
            "theme": "Soft Dark",
            "font_size": 14
        }

def save_settings(settings):
    """Save user settings to JSON."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)