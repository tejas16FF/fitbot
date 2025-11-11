# gamification.py
"""
Gamification utilities for FitBot.

Features:
- Persistent store in user_progress.json
- Weekly challenge generation (multiple types)
- XP, level, badges, streaks
- show_challenge_popup(message) -> fades in/out without blocking
- Render helpers used only on Challenges/Progress pages
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, date
from typing import Dict, Any, List

import streamlit as st

# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------
PROGRESS_FILE = "user_progress.json"
AUTO_SAVE = True


def _read_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"(Gamification) read error: {e}")
    return {}


def _write_json(path: str, obj: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return True
    except Exception as e:
        st.warning(f"(Gamification) write error: {e}")
        return False


def load_persistent() -> Dict[str, Any]:
    return _read_json(PROGRESS_FILE)


def save_persistent(gam: Dict[str, Any], profile: Dict[str, Any], history: List[Dict[str, Any]]):
    """Save all relevant state in a single JSON."""
    gam_copy = dict(gam)
    # Ensure JSON-serializable last_login
    if isinstance(gam_copy.get("last_login"), (date, datetime)):
        gam_copy["last_login"] = gam_copy["last_login"].isoformat()
    data = {"gamification": gam_copy, "profile": profile, "history": history}
    _write_json(PROGRESS_FILE, data)


def save_all_state():
    """Public helper used from app.py."""
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


# ------------------------------------------------------------
# Defaults & init
# ------------------------------------------------------------
DEFAULT_GAM: Dict[str, Any] = {
    "xp": 0,
    "level": 1,
    "streak": 0,
    "last_login": None,             # ISO date string or None
    "badges": [],
    "weekly_challenge": None,       # dict
    "challenge_progress": 0,
    "challenge_completed": False,
    "challenge_start_date": None,   # ISO date string
}

# A larger pool to keep variety week-to-week
CHALLENGE_POOL: List[Dict[str, Any]] = [
    {"type": "chat",   "target": 5,   "reward": 100, "desc": "Ask 5 fitness questions this week."},
    {"type": "chat",   "target": 10,  "reward": 160, "desc": "Chat 10 times with FitBot this week."},
    {"type": "login",  "target": 3,   "reward": 120, "desc": "Open FitBot on 3 different days this week."},
    {"type": "login",  "target": 5,   "reward": 200, "desc": "Open FitBot on 5 days this week."},
    {"type": "streak", "target": 3,   "reward": 150, "desc": "Maintain a 3-day login streak."},
    {"type": "manual", "target": 2,   "reward": 140, "desc": "Log 2 completed workouts (manual) this week."},
    {"type": "xp",     "target": 200, "reward": 180, "desc": "Earn 200 XP this week."},
    {"type": "badge",  "target": 1,   "reward": 120, "desc": "Unlock 1 badge this week."},
]


def _parse_iso_date(maybe_str):
    if isinstance(maybe_str, str):
        try:
            return datetime.fromisoformat(maybe_str).date()
        except Exception:
            return None
    return maybe_str


def initialize_gamification():
    """Load persisted info into session state; quiet, no UI."""
    persisted = load_persistent()

    # Merge persisted gamification with defaults
    merged = DEFAULT_GAM.copy()
    merged.update((persisted.get("gamification") or {}))

    # Parse dates
    merged["last_login"] = _parse_iso_date(merged.get("last_login"))
    merged["challenge_start_date"] = _parse_iso_date(merged.get("challenge_start_date"))

    if "gamification" not in st.session_state:
        st.session_state.gamification = merged

    # Optionally hydrate profile/history from persistence if app hasn't defined them
    if "profile" in persisted and "profile" not in st.session_state:
        st.session_state.profile = persisted["profile"]
    if "history" in persisted and "history" not in st.session_state:
        st.session_state.history = persisted["history"]

    _ensure_weekly_challenge(save=False)


# ------------------------------------------------------------
# XP, Levels, Badges
# ------------------------------------------------------------
def _level_from_xp(xp: int) -> int:
    # Simple curve: level up every 200 XP
    return 1 + xp // 200


def add_xp(amount: int, show_msg: bool = False, reason: str = ""):
    gam = st.session_state.gamification
    gam["xp"] = int(gam.get("xp", 0)) + int(amount)
    old_level = int(gam.get("level", 1))
    gam["level"] = _level_from_xp(gam["xp"])

    if show_msg and amount > 0:
        try:
            st.success(f"💪 +{amount} XP! {reason}".strip())
        except Exception:
            pass

    # Level up indicator
    if gam["level"] > old_level and show_msg:
        try:
            st.balloons()
            st.success(f"🎉 Level up! You are now Level {gam['level']}")
        except Exception:
            pass

    _check_badges()

    if AUTO_SAVE:
        save_all_state()


def _check_badges():
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    badges = set(gam.get("badges", []))

    milestones = [
        (100, "Active Learner"),
        (300, "Fitness Champ"),
        (600, "Consistency King"),
        (1000, "Iron Will"),
    ]
    for threshold, name in milestones:
        if xp >= threshold and name not in badges:
            badges.add(name)
            try:
                st.info(f"🏅 Badge unlocked: {name}")
            except Exception:
                pass

    gam["badges"] = sorted(list(badges))


def reward_for_chat(show_msg: bool = False):
    """Small reward per chat to encourage engagement."""
    add_xp(10, show_msg=show_msg, reason="Asked a question")
    _increase_xp_goal_counter(10)


def _increase_xp_goal_counter(amount: int):
    """If the weekly challenge is XP-based, increase progress accordingly."""
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge")
    if ch and ch.get("type") == "xp" and not gam.get("challenge_completed"):
        gam["challenge_progress"] = int(gam.get("challenge_progress", 0)) + int(amount)
        _maybe_complete_challenge()


# ------------------------------------------------------------
# Weekly Challenges
# ------------------------------------------------------------
def generate_weekly_challenge() -> Dict[str, Any]:
    return random.choice(CHALLENGE_POOL)


def _ensure_weekly_challenge(save: bool = True):
    gam = st.session_state.gamification
    today = datetime.now().date()
    start = gam.get("challenge_start_date")

    if start is None or gam.get("weekly_challenge") is None:
        gam["weekly_challenge"] = generate_weekly_challenge()
        gam["challenge_start_date"] = today
        gam["challenge_progress"] = 0
        gam["challenge_completed"] = False
    else:
        # Reset weekly challenge every 7 days
        if (today - start).days >= 7:
            gam["weekly_challenge"] = generate_weekly_challenge()
            gam["challenge_start_date"] = today
            gam["challenge_progress"] = 0
            gam["challenge_completed"] = False

    if save and AUTO_SAVE:
        save_all_state()


def update_challenge_progress(action_type: str, increment: int = 1):
    """
    Increment challenge progress if the action matches the active challenge type.

    action_type can be: "chat", "login", "manual", "streak", "xp", "badge"
    (app.py typically uses 'chat', 'login', 'manual')
    """
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch or gam.get("challenge_completed"):
        return

    # If types match, or if the challenge is manual and user logs manual action
    if ch.get("type") == action_type or (ch.get("type") == "manual" and action_type == "manual"):
        gam["challenge_progress"] = int(gam.get("challenge_progress", 0)) + int(increment)

    # If the challenge is 'streak', progress mirrors current streak
    if ch.get("type") == "streak":
        gam["challenge_progress"] = int(gam.get("streak", 0))

    _maybe_complete_challenge()


def _maybe_complete_challenge():
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch:
        return

    prog = int(gam.get("challenge_progress", 0))
    target = int(ch.get("target", 1))

    if prog >= target and not gam.get("challenge_completed"):
        gam["challenge_completed"] = True
        reward = int(ch.get("reward", 0))
        add_xp(reward, show_msg=False, reason=f"Completed: {ch.get('desc','challenge')}")

        # Popup (non-blocking)
        try:
            show_challenge_popup(f"🎉 Challenge completed — {ch.get('desc','Great job!')}  +{reward} XP!")
        except Exception:
            try:
                st.success(f"Challenge completed! +{reward} XP")
            except Exception:
                pass

        if AUTO_SAVE:
            save_all_state()


# ------------------------------------------------------------
# Daily login & streaks
# ------------------------------------------------------------
def update_daily_login(silent: bool = True):
    """Call this once when the user opens/continues the session (e.g., after profile submit)."""
    gam = st.session_state.gamification
    today = datetime.now().date()
    last = gam.get("last_login")

    if isinstance(last, str):
        last = _parse_iso_date(last)

    # Only count once per day
    if last is None or today > last:
        # Streak handling
        if last and (today - last).days == 1:
            gam["streak"] = int(gam.get("streak", 0)) + 1
        else:
            gam["streak"] = 1

        gam["last_login"] = today
        add_xp(50, show_msg=not silent, reason="Daily login bonus")
        update_challenge_progress("login")

        if AUTO_SAVE:
            save_all_state()


# ------------------------------------------------------------
# UI sections (render ONLY on Challenges / Progress pages)
# ------------------------------------------------------------
def render_progress_sidebar_full():
    """Compact progress panel — call on Progress or Challenges page."""
    gam = st.session_state.gamification
    xp = int(gam.get("xp", 0))
    level = int(gam.get("level", 1))
    streak = int(gam.get("streak", 0))

    st.subheader("🏆 Progress")
    st.progress((xp % 200) / 200)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 **Streak:** {streak} days")

    badges = gam.get("badges", [])
    if badges:
        st.markdown("**Badges:** " + ", ".join(badges))


def render_weekly_challenge_section():
    """Details of the current weekly challenge + progress bar."""
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
    target = max(1, int(ch.get("target", 1)))
    st.progress(min(prog / target, 1.0))

    if gam.get("challenge_completed"):
        st.success("✅ Completed — reward claimed")
    else:
        # Optional subtle hint
        st.caption(f"Progress: {prog} / {target}")


# ------------------------------------------------------------
# Popup: fade in/out (non-blocking)
# ------------------------------------------------------------
def show_challenge_popup(message: str, duration_ms: int = 2800):
    uid = random.randint(100000, 999999)
    safe = str(message).replace("'", "\\'")
    html = f"""
    <div id="popup_{uid}" style="
        position: fixed;
        top: 18%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: rgba(15,179,139,0.97);
        color: white;
        padding: 14px 18px;
        border-radius: 12px;
        font-weight:700;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        opacity: 0;
        transition: opacity 0.5s ease-in-out;
    ">
      {safe}
    </div>
    <script>
      const el = document.getElementById('popup_{uid}');
      setTimeout(()=>{{ el.style.opacity = 1; }}, 50);
      setTimeout(()=>{{ el.style.opacity = 0; }}, {duration_ms});
      setTimeout(()=>{{ if (el && el.parentNode) el.parentNode.removeChild(el); }}, {duration_ms} + 650);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)
