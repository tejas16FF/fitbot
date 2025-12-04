# gamification.py — redesigned for FitBot UI
# Provides functions: initialize_gamification, save_all_state, update_daily_login,
# reward_for_chat, update_challenge_progress, render_progress_sidebar_full,
# render_weekly_challenge_section, show_challenge_popup, get_gamification_state

import os
import json
import time
import streamlit as st
from datetime import date, datetime

PROGRESS_FILE = "user_progress.json"

# default structure
DEFAULT = {
    "xp": 0,
    "level": 1,
    "badges": [],
    "streak": 0,
    "last_login": None,
    "challenges": {},
    "history": []
}

def _read_storage():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT.copy()
    return DEFAULT.copy()

def _write_storage(data):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# make gamification state available on st.session_state.gamification
def initialize_gamification():
    data = _read_storage()
    if "gamification" not in st.session_state:
        st.session_state.gamification = data
    else:
        # merge/load values if needed
        gam = st.session_state.gamification
        for k, v in data.items():
            if k not in gam:
                gam[k] = v
    # init toast queue if not present
    if "toast_queue" not in st.session_state:
        st.session_state.toast_queue = []

def save_all_state():
    # update file from session state
    data = st.session_state.get("gamification", _read_storage())
    _write_storage(data)

def get_gamification_state():
    return st.session_state.get("gamification", _read_storage())

def update_daily_login(silent=False):
    gam = get_gamification_state()
    last = gam.get("last_login")
    today = date.today().isoformat()
    if last != today:
        # update streak
        # if yesterday was last login -> streak +1 else reset to 1
        try:
            last_date = datetime.fromisoformat(last).date() if last else None
        except Exception:
            last_date = None
        if last_date and (date.today() - last_date).days == 1:
            gam["streak"] = gam.get("streak", 0) + 1
        else:
            gam["streak"] = 1
        gam["last_login"] = today
        gam["xp"] = gam.get("xp", 0) + 10  # daily login XP
        gam["level"] = 1 + gam["xp"] // 200
        save_all_state()
        if not silent:
            show_challenge_popup("Daily check-in +10 XP", kind="success")

def reward_for_chat():
    gam = get_gamification_state()
    gam["xp"] = gam.get("xp", 0) + 10
    gam["level"] = 1 + gam["xp"] // 200
    save_all_state()
    # show popup
    show_challenge_popup("+10 XP for chatting", kind="xp")

def update_challenge_progress(action="chat"):
    # simple demo challenge logic
    gam = get_gamification_state()
    challenges = gam.get("challenges", {})
    # increment simple counter per action
    key = f"cnt_{action}"
    challenges[key] = challenges.get(key, 0) + 1
    gam["challenges"] = challenges
    # example: if user chats 5 times -> award badge
    if action == "chat" and challenges[key] == 5:
        badges = gam.get("badges", [])
        if "Chatterbox" not in badges:
            badges.append("Chatterbox")
            gam["badges"] = badges
            show_challenge_popup("Badge unlocked: Chatterbox 🎉", kind="badge")
            gam["xp"] = gam.get("xp",0) + 30
            gam["level"] = 1 + gam["xp"] // 200
    save_all_state()

def render_progress_sidebar_full():
    gam = get_gamification_state()
    xp = int(gam.get("xp",0)); lvl = gam.get("level",1); streak = gam.get("streak",0)
    # simple render using streamlit fragments (handled in app.py layout)
    st.markdown(f"**Level:** {lvl}")
    st.progress(min(1, (xp % 200) / 200))
    st.markdown(f"**XP:** {xp}  •  **Streak:** {streak} days")
    bad = gam.get("badges", [])
    if bad:
        st.markdown("**Badges:** " + ", ".join(bad))
    else:
        st.markdown("**Badges:** —")

def render_weekly_challenge_section():
    st.markdown("### Weekly challenge")
    gam = get_gamification_state()
    ch = gam.get("challenges", {})
    st.markdown(f"- Chats: {ch.get('cnt_chat',0)}")
    st.markdown(f"- Manual logs: {ch.get('cnt_manual',0)}")

def show_challenge_popup(message: str, kind: str = "info"):
    """
    Add a popup to Streamlit session toast queue.
    kind: 'info', 'xp', 'success', 'badge', 'warn'
    """
    # Ensure session_state toast_queue exists
    if "toast_queue" not in st.session_state:
        st.session_state.toast_queue = []
    st.session_state.toast_queue.append({"msg": message, "kind": kind})

# EOF
