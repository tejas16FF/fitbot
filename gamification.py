# gamification.py
import json
import os
import random
import streamlit as st
from datetime import datetime, date
from typing import Dict, Any, Optional, List

PROGRESS_FILE = "user_progress.json"
AUTO_SAVE = True

# -----------------------------
# Persistent storage helpers
# -----------------------------
def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not read progress file: {e}")
    return None

def _write_json(path: str, obj: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return True
    except Exception as e:
        st.warning(f"Could not save progress file: {e}")
        return False

def load_persistent_state() -> Dict[str, Any]:
    state = _read_json(PROGRESS_FILE) or {}
    gam = state.get("gamification", {})
    profile = state.get("profile", {})
    history = state.get("history", [])
    # convert last_login to ISO date object if present
    if isinstance(gam.get("last_login"), str):
        try:
            gam["last_login"] = datetime.fromisoformat(gam["last_login"]).date()
        except Exception:
            gam["last_login"] = None
    return {"gamification": gam, "profile": profile, "history": history}

def save_persistent_state(gamification: Dict[str, Any], profile: Dict[str, Any], history: List[Dict[str, Any]]):
    gam_copy = dict(gamification)
    if isinstance(gam_copy.get("last_login"), (datetime, date)):
        gam_copy["last_login"] = gam_copy["last_login"].isoformat()
    data = {"gamification": gam_copy, "profile": profile, "history": history}
    _write_json(PROGRESS_FILE, data)

# -----------------------------
# Initialization & defaults
# -----------------------------
def initialize_gamification():
    """Load persisted gamification into session state without showing messages."""
    persisted = load_persistent_state()
    gam = persisted.get("gamification", {})
    defaults = {
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
    if "gamification" not in st.session_state:
        merged = defaults.copy()
        merged.update(gam)
        st.session_state.gamification = merged
    # restore profile/history if not present
    if "profile" not in st.session_state:
        st.session_state.profile = persisted.get("profile", {})
    if "history" not in st.session_state:
        st.session_state.history = persisted.get("history", [])
    # normalize last_login
    if isinstance(st.session_state.gamification.get("last_login"), str):
        try:
            st.session_state.gamification["last_login"] = datetime.fromisoformat(st.session_state.gamification["last_login"]).date()
        except Exception:
            st.session_state.gamification["last_login"] = None
    # ensure a weekly challenge exists
    check_and_reset_weekly_challenge(save=False)
    if AUTO_SAVE:
        save_persistent_state(st.session_state.gamification, st.session_state.profile, st.session_state.history)

# -----------------------------
# XP, Levels & Badges
# -----------------------------
def _level_from_xp(xp: int) -> int:
    # simple linear progression: every 200 XP => level up
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
    check_badges()
    if AUTO_SAVE:
        save_persistent_state(gam, st.session_state.profile, st.session_state.history)

def reward_for_chat(show_msg: bool = False):
    add_xp(10, show_msg=show_msg, reason="Asked a question")

def check_badges():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    badges = gam.get("badges", [])
    milestones = [
        (100, "💬 Active Learner"),
        (300, "🏋️ Fitness Champ"),
        (500, "🔥 Consistency King"),
        (1000, "🎯 Peak Performer"),
    ]
    for threshold, name in milestones:
        if xp >= threshold and name not in badges:
            badges.append(name)
            # only show message if we are on Chat/Progress pages (avoid in profile flow)
            try:
                st.info(f"🏅 Badge unlocked: {name}")
            except Exception:
                pass
    gam["badges"] = badges

# -----------------------------
# Weekly challenge system
# -----------------------------
def generate_weekly_challenge():
    options = [
        {"type": "chat", "target": 5, "reward": 100, "desc": "Ask 5 fitness questions this week."},
        {"type": "login", "target": 3, "reward": 150, "desc": "Log in on 3 different days this week."},
        {"type": "streak", "target": 5, "reward": 200, "desc": "Maintain a 5-day login streak."},
    ]
    return random.choice(options)

def check_and_reset_weekly_challenge(save: bool = True):
    gam = st.session_state.gamification
    today = datetime.now().date()
    start_date = gam.get("challenge_start_date")
    if start_date and isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date).date()
        except Exception:
            start_date = None
    if gam.get("weekly_challenge") is None or start_date is None:
        gam["weekly_challenge"] = generate_weekly_challenge()
        gam["challenge_start_date"] = today
        gam["challenge_progress"] = 0
        gam["challenge_completed"] = False
    else:
        days = (today - start_date).days if start_date else 8
        if days >= 7:
            gam["weekly_challenge"] = generate_weekly_challenge()
            gam["challenge_start_date"] = today
            gam["challenge_progress"] = 0
            gam["challenge_completed"] = False
    if save and AUTO_SAVE:
        save_persistent_state(gam, st.session_state.profile, st.session_state.history)

def update_challenge_progress(action_type: str):
    gam = st.session_state.gamification
    challenge = gam.get("weekly_challenge")
    if not challenge or gam.get("challenge_completed"):
        return
    if challenge.get("type") == action_type:
        gam["challenge_progress"] = int(gam.get("challenge_progress", 0)) + 1
        if gam["challenge_progress"] >= challenge.get("target", 9999):
            gam["challenge_completed"] = True
            add_xp(challenge.get("reward", 0), show_msg=True, reason=f"Completed weekly challenge: {challenge.get('desc')}")
    if AUTO_SAVE:
        save_persistent_state(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# Daily login & streaks
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
        # consecutive?
        if last and (today - last).days == 1:
            gam["streak"] = int(gam.get("streak", 0)) + 1
        else:
            gam["streak"] = 1
        gam["last_login"] = today
        # award XP quietly by default
        add_xp(50, show_msg=not silent, reason="Daily login bonus")
        update_challenge_progress("login")
        if AUTO_SAVE:
            save_persistent_state(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# UI rendering helpers
# -----------------------------
def render_progress_sidebar():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    level = int(gam.get("level", 1))
    streak = int(gam.get("streak", 0))
    badges = gam.get("badges", [])

    st.markdown("---")
    st.subheader("🏆 Progress")
    progress_val = (xp % 200) / 200 if 200 > 0 else 0.0
    st.progress(progress_val)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 **Streak:** {streak} days")
    if badges:
        st.markdown("**Badges:**")
        cols = st.columns(min(3, len(badges)))
        for i, b in enumerate(badges):
            try:
                cols[i % len(cols)].markdown(f"• {b}")
            except Exception:
                st.markdown(f"- {b}")
    # Weekly challenge
    check_and_reset_weekly_challenge(save=False)
    challenge = gam.get("weekly_challenge")
    if challenge:
        st.markdown("---")
        st.subheader("🎯 Weekly Challenge")
        st.markdown(f"**{challenge.get('desc')}**")
        prog = int(gam.get("challenge_progress", 0))
        target = int(challenge.get("target", 1))
        st.progress(min(prog / target, 1.0))
        if gam.get("challenge_completed"):
            st.success("✅ Completed — reward claimed")

# -----------------------------
# Utilities
# -----------------------------
def save_all_state():
    """Save gamification/profile/history to disk."""
    save_persistent_state(st.session_state.gamification, st.session_state.profile, st.session_state.history)

def reset_progress_file():
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            st.success("Progress file removed.")
        else:
            st.info("No progress file to remove.")
    except Exception as e:
        st.error(f"Could not remove file: {e}")
