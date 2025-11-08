# gamification.py
"""
Gamification utilities for FitBot.

Features:
- Persistent store in user_progress.json
- Weekly challenge generation (multiple types)
- XP, level, badges, streaks
- show_challenge_popup(html) to show a fade-in/out message on completion
"""

import json
import os
import random
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional

PROGRESS_FILE = "user_progress.json"
AUTO_SAVE = True

# -----------------------------
# Persistence helpers
# -----------------------------
def _read_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"(Gamify) read error: {e}")
    return {}

def _write_json(path: str, obj: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return True
    except Exception as e:
        st.warning(f"(Gamify) write error: {e}")
        return False

def load_persistent() -> Dict[str, Any]:
    s = _read_json(PROGRESS_FILE)
    # normalize values
    return s

def save_persistent(gam: Dict[str, Any], profile: Dict[str, Any], history: List[Dict[str, Any]]):
    gam_copy = dict(gam)
    # last_login -> iso
    if isinstance(gam_copy.get("last_login"), (date, datetime)):
        gam_copy["last_login"] = gam_copy["last_login"].isoformat()
    data = {"gamification": gam_copy, "profile": profile, "history": history}
    _write_json(PROGRESS_FILE, data)

# -----------------------------
# Defaults & init
# -----------------------------
DEFAULT_GAM = {
    "xp": 0,
    "level": 1,
    "streak": 0,
    "last_login": None,
    "badges": [],
    "weekly_challenge": None,
    "challenge_progress": 0,
    "challenge_completed": False,
    "challenge_start_date": None,
}

NEW_CHALLENGES = [
    {"type": "chat", "target": 5, "reward": 100, "desc": "Ask 5 fitness questions this week."},
    {"type": "login", "target": 3, "reward": 120, "desc": "Log in on 3 different days this week."},
    {"type": "streak", "target": 3, "reward": 150, "desc": "Maintain a 3-day login streak."},
    {"type": "xp", "target": 200, "reward": 80, "desc": "Earn 200 XP this week."},
    {"type": "manual", "target": 2, "reward": 100, "desc": "Log 2 completed workouts (manual) this week."},
    {"type": "badge", "target": 1, "reward": 100, "desc": "Unlock 1 badge this week."},
]

def initialize_gamification():
    """Load persisted info into session state without spamming UI."""
    persisted = load_persistent()
    gam = persisted.get("gamification") or {}
    merged = DEFAULT_GAM.copy()
    merged.update(gam)
    # convert last_login
    if isinstance(merged.get("last_login"), str):
        try:
            merged["last_login"] = datetime.fromisoformat(merged["last_login"]).date()
        except Exception:
            merged["last_login"] = None
    if "gamification" not in st.session_state:
        st.session_state.gamification = merged
    if "profile" not in st.session_state:
        st.session_state.profile = persisted.get("profile", {})
    if "history" not in st.session_state:
        st.session_state.history = persisted.get("history", [])
    # ensure challenge exists
    _ensure_weekly_challenge(save=False)

# -----------------------------
# XP & badges
# -----------------------------
def _level_from_xp(xp: int) -> int:
    return 1 + xp // 200

def add_xp(amount: int, show_msg: bool = False, reason: str = ""):
    gam = st.session_state.gamification
    gam["xp"] = int(gam.get("xp", 0)) + int(amount)
    old_level = int(gam.get("level", 1))
    gam["level"] = _level_from_xp(gam["xp"])
    if show_msg:
        st.success(f"💪 +{amount} XP! {reason}")
    if gam["level"] > old_level and show_msg:
        st.balloons()
        st.success(f"🎉 Level up! You are now Level {gam['level']}")
    _check_badges()
    if AUTO_SAVE:
        save_persistent(gam, st.session_state.profile, st.session_state.history)

def _check_badges():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    badges = gam.get("badges", [])
    milestones = [(100, "Active Learner"), (300, "Fitness Champ"), (600, "Consistency King")]
    for thr, name in milestones:
        if xp >= thr and name not in badges:
            badges.append(name)
            # small notice (only show if on non-profile pages)
            try:
                st.info(f"🏅 Badge unlocked: {name}")
            except Exception:
                pass
    gam["badges"] = badges

def reward_for_chat(show_msg: bool = False):
    add_xp(10, show_msg=show_msg, reason="Asked a question")
    _increase_xp_goal_counter(10)

def _increase_xp_goal_counter(amount: int):
    # If current weekly challenge is xp type, increment progress
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge")
    if ch and ch.get("type") == "xp" and not gam.get("challenge_completed"):
        gam["challenge_progress"] = int(gam.get("challenge_progress", 0)) + amount
        _maybe_complete_challenge()

# -----------------------------
# Weekly challenge helpers
# -----------------------------
def generate_weekly_challenge() -> Dict[str, Any]:
    return random.choice(NEW_CHALLENGES)

def _ensure_weekly_challenge(save: bool = True):
    gam = st.session_state.gamification
    today = datetime.now().date()
    start = gam.get("challenge_start_date")
    if isinstance(start, str):
        try:
            start = datetime.fromisoformat(start).date()
        except Exception:
            start = None
    if gam.get("weekly_challenge") is None or start is None:
        gam["weekly_challenge"] = generate_weekly_challenge()
        gam["challenge_start_date"] = today
        gam["challenge_progress"] = 0
        gam["challenge_completed"] = False
    else:
        if (today - start).days >= 7:
            gam["weekly_challenge"] = generate_weekly_challenge()
            gam["challenge_start_date"] = today
            gam["challenge_progress"] = 0
            gam["challenge_completed"] = False
    if save and AUTO_SAVE:
        save_persistent(gam, st.session_state.profile, st.session_state.history)

def update_challenge_progress(action_type: str, increment: int = 1):
    """Call this when user performs an action that might advance the challenge."""
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch or gam.get("challenge_completed"):
        return
    # if action_type matches challenge type or is a general manual log
    if ch.get("type") == action_type or (ch.get("type") == "manual" and action_type == "manual"):
        gam["challenge_progress"] = int(gam.get("challenge_progress", 0)) + increment
    # special: login increments also update streaks in update_daily_login (separate)
    _maybe_complete_challenge()

def _maybe_complete_challenge():
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch:
        return
    prog = gam.get("challenge_progress", 0)
    target = ch.get("target", 9999)
    # For XP-type, progress can be >= target (we increment by XP amounts)
    if prog >= target:
        if not gam.get("challenge_completed"):
            gam["challenge_completed"] = True
            # reward XP once
            add_xp(ch.get("reward", 0), show_msg=False, reason=f"Completed: {ch.get('desc')}")
            # show popup
            try:
                show_challenge_popup(f"🎉 Challenge completed — {ch.get('desc')} +{ch.get('reward',0)} XP!")
            except Exception:
                try:
                    st.success(f"Challenge completed! +{ch.get('reward',0)} XP")
                except Exception:
                    pass
            if AUTO_SAVE:
                save_persistent(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# Daily login + streak
# -----------------------------
def update_daily_login(silent: bool = True):
    gam = st.session_state.gamification
    today = datetime.now().date()
    last = gam.get("last_login")
    if isinstance(last, str):
        try:
            last = datetime.fromisoformat(last).date()
        except Exception:
            last = None
    if last is None or today > last:
        if last and (today - last).days == 1:
            gam["streak"] = int(gam.get("streak", 0)) + 1
        else:
            gam["streak"] = 1
        gam["last_login"] = today
        add_xp(50, show_msg=not silent, reason="Daily login bonus")
        update_challenge_progress("login")
        if AUTO_SAVE:
            save_persistent(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# UI: progress rendering (only used on Challenges page)
# -----------------------------
def render_progress_sidebar_compact():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    level = int(gam.get("level", 1))
    streak = int(gam.get("streak", 0))
    st.markdown("---")
    st.subheader("🏆 Progress (summary)")
    prog = (xp % 200) / 200
    st.progress(prog)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 **Streak:** {streak} days")

def render_progress_sidebar_full():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    level = int(gam.get("level", 1))
    streak = int(gam.get("streak", 0))
    st.subheader("🏆 Progress")
    st.progress((xp % 200) / 200)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 **Streak:** {streak} days")
    if gam.get("badges"):
        st.markdown("**Badges:** " + ", ".join(gam.get("badges", [])))

def render_weekly_challenge_section():
    _ensure_weekly_challenge(save=False)
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch:
        st.info("No active weekly challenge.")
        return
    st.markdown("---")
    st.subheader("🎯 Weekly Challenge")
    st.markdown(f"**{ch.get('desc','-')}**")
    prog = int(gam.get("challenge_progress", 0))
    target = int(ch.get("target", 1))
    st.progress(min(prog / target, 1.0))
    if gam.get("challenge_completed"):
        st.success("✅ Completed — reward claimed")

# -----------------------------
# Popup: fade in/out using HTML+JS
# -----------------------------
def show_challenge_popup(message: str, duration_ms: int = 3000):
    """Show a small centered popup that fades out after duration_ms."""
    uid = random.randint(100000, 999999)
    safe = str(message).replace("'", "\\'")
    html = f"""
    <div id="popup_{uid}" style="
        position: fixed;
        top: 18%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: rgba(15,179,139,0.95);
        color: white;
        padding: 18px 22px;
        border-radius: 12px;
        font-weight:700;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        opacity: 0;
        transition: opacity 0.6s ease-in-out;
    ">
      {safe}
    </div>
    <script>
      const el = document.getElementById('popup_{uid}');
      setTimeout(()=>{{ el.style.opacity = 1; }}, 50);
      setTimeout(()=>{{ el.style.opacity = 0; }}, {duration_ms});
      setTimeout(()=>{{ el.remove(); }}, {duration_ms} + 700);
    </script>
    """
    try:
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        # fallback: plain Streamlit message
        try:
            st.success(message)
        except Exception:
            pass

# -----------------------------
# Utilities
# -----------------------------
def save_all_state():
    save_persistent(st.session_state.gamification, st.session_state.profile, st.session_state.history)

def reset_progress_file():
    if os.path.exists(PROGRESS_FILE):
        try:
            os.remove(PROGRESS_FILE)
            st.success("Progress file removed.")
        except Exception as e:
            st.error(f"Remove failed: {e}")
    else:
        st.info("No progress file present.")
