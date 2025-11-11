# gamification.py — Minimal API compatible with your final app.py
# Includes: XP + Level system, simple weekly challenges, persistence,
# and a small popup helper you can call later if needed.

import json
import os
import random
import time
import streamlit as st
from datetime import date, datetime, timedelta

_STATE_FILE = "gamification_state.json"

# --------------------------
# Internal state management
# --------------------------
def _default_state():
    return {
        "xp": 0,
        "completed_challenges": [],   # list of challenge ids
        "weekly": {
            "id": None,
            "title": None,
            "description": None,
            "xp": 0,
            "started_on": None,      # iso date
        },
        "last_login_date": None,
        "streak": 0,
    }

def _load_state():
    if not os.path.exists(_STATE_FILE):
        return _default_state()
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # merge with defaults to be safe
        base = _default_state()
        base.update(data)
        # ensure nested weekly defaults
        if "weekly" not in base or not isinstance(base["weekly"], dict):
            base["weekly"] = _default_state()["weekly"]
        return base
    except Exception:
        return _default_state()

def _save_state(state):
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

def _ensure_session():
    if "gam_state" not in st.session_state:
        st.session_state.gam_state = _load_state()

# --------------------------
# Level math
# --------------------------
def _level_from_xp(xp: int) -> int:
    # Level every 200 XP: 0-199 -> L1, 200-399 -> L2, ...
    return 1 + xp // 200

# --------------------------
# Public API (used by app.py)
# --------------------------
def get_user_xp() -> int:
    _ensure_session()
    return int(st.session_state.gam_state.get("xp", 0))

def get_current_level() -> int:
    return _level_from_xp(get_user_xp())

def add_xp(amount: int):
    """Add XP and persist."""
    _ensure_session()
    st.session_state.gam_state["xp"] = int(st.session_state.gam_state.get("xp", 0)) + int(amount)
    _save_state(st.session_state.gam_state)

# ---- Weekly challenges model ----
# Provide a stable list; app reads with get_weekly_challenges()
_CHALLENGE_POOL = {
    "water_7":      {"title": "Drink water 7 times",         "description": "Stay hydrated through the day.", "xp": 60},
    "steps_5000":   {"title": "Walk 5000 steps",              "description": "Light cardio and step target.",  "xp": 80},
    "stretch_10":   {"title": "10-min stretching",            "description": "Mobilize to prevent injuries.",  "xp": 50},
    "sleep_8":      {"title": "Sleep 8 hours",                "description": "Prioritize deep recovery.",      "xp": 70},
    "cardio_20":    {"title": "20-min cardio",                "description": "Elevate your heart rate.",       "xp": 90},
    "no_sugar":     {"title": "No sugar today",               "description": "Cut refined sugars for a day.",  "xp": 75},
    "protein_goal": {"title": "Hit protein goal",             "description": "Stay on track with nutrition.",  "xp": 85},
    "pushup_50":    {"title": "50 push-ups total",            "description": "Accumulate across sets.",        "xp": 110},
    "meditate_10":  {"title": "10-min meditation",            "description": "Reset and focus.",               "xp": 50},
    "plank_2min":   {"title": "Hold plank 2 minutes total",   "description": "Core strength builder.",         "xp": 100},
}

def _ensure_weekly_selected():
    _ensure_session()
    wk = st.session_state.gam_state["weekly"]
    today = date.today()
    if not wk["id"] or not wk["started_on"]:
        cid, meta = random.choice(list(_CHALLENGE_POOL.items()))
        st.session_state.gam_state["weekly"] = {
            "id": cid,
            "title": meta["title"],
            "description": meta["description"],
            "xp": meta["xp"],
            "started_on": today.isoformat()
        }
        _save_state(st.session_state.gam_state)
        return

    # rotate every 7 days
    try:
        started = datetime.fromisoformat(wk["started_on"]).date()
        if (today - started).days >= 7:
            cid, meta = random.choice(list(_CHALLENGE_POOL.items()))
            st.session_state.gam_state["weekly"] = {
                "id": cid,
                "title": meta["title"],
                "description": meta["description"],
                "xp": meta["xp"],
                "started_on": today.isoformat()
            }
            _save_state(st.session_state.gam_state)
    except Exception:
        # if bad date, reset
        cid, meta = random.choice(list(_CHALLENGE_POOL.items()))
        st.session_state.gam_state["weekly"] = {
            "id": cid,
            "title": meta["title"],
            "description": meta["description"],
            "xp": meta["xp"],
            "started_on": today.isoformat()
        }
        _save_state(st.session_state.gam_state)

def get_weekly_challenges() -> dict:
    """
    Return a dict of available challenges (pool),
    and ensure a current weekly selection exists.
    """
    _ensure_weekly_selected()
    return _CHALLENGE_POOL

def get_completed_challenges() -> list:
    _ensure_session()
    return list(st.session_state.gam_state.get("completed_challenges", []))

def complete_challenge(challenge_id: str):
    """
    Mark a challenge completed, award XP once.
    """
    _ensure_session()
    if challenge_id in st.session_state.gam_state["completed_challenges"]:
        return  # already completed

    # Add to completed and award XP
    st.session_state.gam_state["completed_challenges"].append(challenge_id)
    meta = _CHALLENGE_POOL.get(challenge_id)
    if meta:
        add_xp(int(meta["xp"]))
    _save_state(st.session_state.gam_state)

# --------------------------
# (Optional) Toast popup
# --------------------------
def show_toast(message: str, duration_ms: int = 2600):
    """
    Small slide-in toast (top-right). Use in app when you want feedback.
    """
    uid = random.randint(100000, 999999)
    safe = str(message).replace("'", "\\'")
    html = f"""
    <div id="toast_{uid}" style="
        position: fixed; top: 18px; right: -420px; z-index: 9999;
        background: #0FB38B; color: #fff; padding: 12px 16px;
        border-radius: 10px; box-shadow: 0 8px 22px rgba(0,0,0,.25);
        font-weight: 700; min-width: 240px; transition: right .35s ease, opacity .35s ease;">
      {safe}
    </div>
    <script>
      const el = document.getElementById('toast_{uid}');
      el.style.right = '18px';
      setTimeout(() => {{
        el.style.opacity = 0;
        el.style.right = '-420px';
      }}, {duration_ms});
      setTimeout(() => el.remove(), {duration_ms} + 500);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)
    time.sleep(0.05)
