# gamification.py — persistent gamification system for FitBot
import json
import os
import random
import streamlit as st
from datetime import datetime
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
        st.warning(f"⚠️ Could not read progress file: {e}")
    return None


def _write_json(path: str, obj: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not save progress file: {e}")
        return False


def load_persistent_state() -> Dict[str, Any]:
    """Load persisted state (profile, history, gamification)."""
    state = _read_json(PROGRESS_FILE) or {}
    gam = state.get("gamification", {})
    profile = state.get("profile", {})
    history = state.get("history", [])

    # Convert last_login string to date
    if gam.get("last_login"):
        try:
            gam["last_login"] = datetime.fromisoformat(gam["last_login"]).date()
        except Exception:
            gam["last_login"] = None

    return {"gamification": gam, "profile": profile, "history": history}


def save_persistent_state(gamification: Dict[str, Any], profile: Dict[str, Any], history: List[Dict[str, Any]]):
    """Save all user data persistently."""
    gam_copy = dict(gamification)
    if isinstance(gam_copy.get("last_login"), datetime):
        gam_copy["last_login"] = gam_copy["last_login"].isoformat()
    elif hasattr(gam_copy.get("last_login"), "isoformat"):
        gam_copy["last_login"] = gam_copy["last_login"].isoformat()

    data = {
        "gamification": gam_copy,
        "profile": profile,
        "history": history,
    }
    _write_json(PROGRESS_FILE, data)

# -----------------------------
# Initialization
# -----------------------------
def initialize_gamification():
    """Initialize gamification from saved file or defaults."""
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

    if "profile" not in st.session_state:
        st.session_state.profile = persisted.get("profile", {})
    if "history" not in st.session_state:
        st.session_state.history = persisted.get("history", [])

    # Ensure proper date format
    if isinstance(st.session_state.gamification.get("last_login"), str):
        try:
            st.session_state.gamification["last_login"] = datetime.fromisoformat(
                st.session_state.gamification["last_login"]
            ).date()
        except Exception:
            st.session_state.gamification["last_login"] = None

    check_and_reset_weekly_challenge(save=False)
    if AUTO_SAVE:
        save_persistent_state(st.session_state.gamification, st.session_state.profile, st.session_state.history)

# -----------------------------
# XP, Levels, and Badges
# -----------------------------
def add_xp(amount: int, show_msg: bool = True, reason: str = ""):
    gam = st.session_state.gamification
    gam["xp"] = int(gam.get("xp", 0)) + int(amount)

    if show_msg:
        st.success(f"💪 +{amount} XP! {reason}")

    leveled = False
    while gam["xp"] >= gam["level"] * 200:
        gam["level"] += 1
        leveled = True
    if leveled:
        st.balloons()
        st.success(f"🎉 Level Up! You are now Level {gam['level']}")

    check_badges()
    if AUTO_SAVE:
        save_persistent_state(gam, st.session_state.profile, st.session_state.history)


def reward_for_chat():
    """Reward XP when user asks a question."""
    add_xp(10, reason="Good question!")
    update_challenge_progress("chat")


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
            st.success(f"🏅 Badge unlocked: {name}")
    gam["badges"] = badges

# -----------------------------
# Challenges
# -----------------------------
def generate_weekly_challenge():
    challenges = [
        {"type": "chat", "target": 5, "reward": 100, "desc": "Ask 5 fitness questions this week."},
        {"type": "login", "target": 3, "reward": 150, "desc": "Login 3 different days this week."},
        {"type": "streak", "target": 5, "reward": 200, "desc": "Maintain a 5-day login streak."},
    ]
    return random.choice(challenges)


def check_and_reset_weekly_challenge(save=True):
    gam = st.session_state.gamification
    today = datetime.now().date()
    start_date = gam.get("challenge_start_date")

    if start_date:
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date).date()
            except Exception:
                start_date = None

    # Set new challenge if none or expired
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
        st.info(f"🎯 Challenge progress: {gam['challenge_progress']}/{challenge.get('target')}")
        if gam["challenge_progress"] >= challenge.get("target", 0):
            gam["challenge_completed"] = True
            add_xp(challenge.get("reward", 0), reason=f"🏆 Completed weekly challenge: {challenge.get('desc')}")

    if AUTO_SAVE:
        save_persistent_state(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# Daily login / streak
# -----------------------------
def update_daily_login():
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
        add_xp(50, reason="Daily login bonus")
        update_challenge_progress("login")
        if AUTO_SAVE:
            save_persistent_state(gam, st.session_state.profile, st.session_state.history)

# -----------------------------
# UI helpers
# -----------------------------
def render_progress_sidebar():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    level = int(gam.get("level", 1))
    streak = int(gam.get("streak", 0))
    badges = gam.get("badges", [])

    st.markdown("---")
    st.subheader("🏆 Progress")
    progress_val = (xp % 200) / 200
    st.progress(progress_val)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 Streak:** {streak} days**")

    if badges:
        st.markdown("**🏅 Badges:**")
        for b in badges:
            st.markdown(f"- {b}")

    challenge = gam.get("weekly_challenge")
    if challenge:
        st.markdown("---")
        st.subheader("🎯 Weekly Challenge")
        st.markdown(f"**{challenge.get('desc')}**")
        prog = int(gam.get("challenge_progress", 0))
        target = int(challenge.get("target", 1))
        st.progress(min(prog / target, 1.0))
        if gam.get("challenge_completed"):
            st.success("✅ Challenge completed!")

# -----------------------------
# Utilities
# -----------------------------
def save_all_state():
    """Save the entire state to disk (called by app)."""
    save_persistent_state(
        st.session_state.get("gamification", {}),
        st.session_state.get("profile", {}),
        st.session_state.get("history", []),
    )


def reset_progress_file():
    """Deletes saved progress."""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            st.success("Progress file deleted.")
        else:
            st.info("No progress file found.")
    except Exception as e:
        st.error(f"Could not remove file: {e}")
