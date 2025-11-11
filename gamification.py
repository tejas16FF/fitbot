# gamification.py — Pro version (badges, animated rings, soft sounds, slide-in popups)
import json
import os
import time
import random
import streamlit as st
from datetime import date, datetime, timedelta

GAMIFICATION_FILE = "gamification_state.json"

# -------------------------------------------------
# Defaults & Challenge Catalog
# -------------------------------------------------
def _default_state():
    return {
        "xp": 0,
        "chat_count": 0,
        "daily_login_streak": 0,
        "last_login_date": None,

        # Weekly challenge object
        "weekly_challenge": None,            # {id, title, type, target, reward, started_on}
        "weekly_progress": 0,
        "challenge_completed": False,

        # Badges & levels
        "badges": [],                        # list of badge ids
    }

# Simple catalog; types: chat/login/manual/xp
CHALLENGE_CATALOG = [
    {"id": "chat_5",    "title": "Ask 5 questions",             "type": "chat",  "target": 5,  "reward": 100},
    {"id": "chat_10",   "title": "Ask 10 questions",            "type": "chat",  "target": 10, "reward": 180},
    {"id": "login_3",   "title": "Login on 3 days",             "type": "login", "target": 3,  "reward": 120},
    {"id": "login_5",   "title": "Login on 5 days",             "type": "login", "target": 5,  "reward": 200},
    {"id": "manual_2",  "title": "Log 2 workouts (manual)",     "type": "manual","target": 2,  "reward": 120},
    {"id": "xp_200",    "title": "Earn 200 XP this week",       "type": "xp",    "target": 200,"reward": 150},
]

# Optional badge images (place in assets/badges/)
BADGE_META = {
    "weekly_champion": {
        "label": "Weekly Champion",
        "file": "assets/badges/weekly_champion.png",
        "emoji": "🏆",
        "unlock_reason": "Completed a weekly challenge",
    },
    "streak_3": {
        "label": "Streak 3",
        "file": "assets/badges/streak_3.png",
        "emoji": "🔥",
        "unlock_reason": "3-day login streak",
    },
    "streak_7": {
        "label": "Streak 7",
        "file": "assets/badges/streak_7.png",
        "emoji": "⚡",
        "unlock_reason": "7-day login streak",
    },
    "level_5": {
        "label": "Level 5",
        "file": "assets/badges/level_5.png",
        "emoji": "🎖️",
        "unlock_reason": "Reached Level 5",
    },
    "level_10": {
        "label": "Level 10",
        "file": "assets/badges/level_10.png",
        "emoji": "🏅",
        "unlock_reason": "Reached Level 10",
    },
}

# -------------------------------------------------
# File IO
# -------------------------------------------------
def _load_state_from_file():
    if not os.path.exists(GAMIFICATION_FILE):
        return _default_state()
    try:
        with open(GAMIFICATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**_default_state(), **data}
    except Exception:
        return _default_state()

def _save_state_to_file(state):
    try:
        with open(GAMIFICATION_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

# -------------------------------------------------
# Public API (called by app.py)
# -------------------------------------------------
def initialize_gamification():
    """Load or init into session state. Also ensure a current weekly challenge exists."""
    if "gamification" not in st.session_state:
        st.session_state.gamification = _load_state_from_file()
    _ensure_weekly_challenge()

def save_all_state():
    if "gamification" in st.session_state:
        _save_state_to_file(st.session_state.gamification)

def reset_progress_file():
    if os.path.exists(GAMIFICATION_FILE):
        try:
            os.remove(GAMIFICATION_FILE)
            st.success("Gamification state reset.")
        except Exception as e:
            st.error(f"Reset failed: {e}")
    st.session_state.gamification = _default_state()

# -------------------------------------------------
# Core logic: level, XP, badges
# -------------------------------------------------
def _level_from_xp(xp: int) -> int:
    # Level every 200 XP; tweak if needed
    return 1 + xp // 200

def _grant_badge(badge_id: str, silent: bool = False):
    g = st.session_state.gamification
    if badge_id not in g["badges"]:
        g["badges"].append(badge_id)
        save_all_state()
        meta = BADGE_META.get(badge_id, {})
        label = meta.get("label", badge_id)
        reason = meta.get("unlock_reason", "")
        if not silent:
            show_challenge_popup(f"🏅 Badge unlocked: {label}! {reason}")

def _check_auto_badges():
    g = st.session_state.gamification
    xp = int(g["xp"])
    streak = int(g["daily_login_streak"])
    level = _level_from_xp(xp)

    if streak >= 3:
        _grant_badge("streak_3", silent=True)
    if streak >= 7:
        _grant_badge("streak_7", silent=True)
    if level >= 5:
        _grant_badge("level_5", silent=True)
    if level >= 10:
        _grant_badge("level_10", silent=True)

def _add_xp(amount: int, reason: str = "", popup: bool = False):
    g = st.session_state.gamification
    before_level = _level_from_xp(g["xp"])
    g["xp"] += int(amount)
    after_level = _level_from_xp(g["xp"])
    save_all_state()

    if popup:
        show_challenge_popup(f"💪 +{amount} XP {('- ' + reason) if reason else ''}")

    if after_level > before_level:
        show_challenge_popup(f"🎉 Level up! You are now Level {after_level}")
    _check_auto_badges()

# -------------------------------------------------
# Daily login & chat rewards
# -------------------------------------------------
def update_daily_login(silent: bool = False):
    g = st.session_state.gamification
    today = date.today().isoformat()
    last = g.get("last_login_date")

    if last != today:
        # Update streak
        if last:
            try:
                last_d = datetime.fromisoformat(last).date()
                if (date.today() - last_d) == timedelta(days=1):
                    g["daily_login_streak"] += 1
                else:
                    g["daily_login_streak"] = 1
            except Exception:
                g["daily_login_streak"] = 1
        else:
            g["daily_login_streak"] = 1

        g["last_login_date"] = today
        _add_xp(50, reason="Daily login", popup=not silent)
        _bump_weekly("login", 1)
        save_all_state()

def reward_for_chat(show_msg: bool = False):
    g = st.session_state.gamification
    g["chat_count"] += 1
    save_all_state()
    _add_xp(10, reason="Asked a question", popup=show_msg)
    _bump_weekly("chat", 1)

# -------------------------------------------------
# Weekly challenges
# -------------------------------------------------
def _pick_new_weekly():
    return random.choice(CHALLENGE_CATALOG).copy()

def _ensure_weekly_challenge():
    g = st.session_state.gamification
    ch = g.get("weekly_challenge")
    now = date.today()

    # Start new if none OR older than 7 days
    start_str = ch.get("started_on") if ch else None
    age_days = None
    if start_str:
        try:
            started = datetime.fromisoformat(start_str).date()
            age_days = (now - started).days
        except Exception:
            age_days = None

    if (not ch) or (age_days is None) or (age_days >= 7):
        g["weekly_challenge"] = _pick_new_weekly()
        g["weekly_challenge"]["started_on"] = now.isoformat()
        g["weekly_progress"] = 0
        g["challenge_completed"] = False
        save_all_state()

def _bump_weekly(action: str, inc: int):
    """Increase weekly progress if action type matches challenge."""
    g = st.session_state.gamification
    ch = g.get("weekly_challenge") or {}
    if not ch or g.get("challenge_completed"):
        return

    typ = ch.get("type")
    if action == "xp" and typ == "xp":
        g["weekly_progress"] += inc
    elif action == typ:
        g["weekly_progress"] += inc
    elif typ == "manual" and action == "manual":
        g["weekly_progress"] += inc

    # completion check
    if g["weekly_progress"] >= int(ch.get("target", 1)) and not g["challenge_completed"]:
        g["challenge_completed"] = True
        reward = int(ch.get("reward", 0))
        _add_xp(reward, reason=f"Completed: {ch.get('title','Weekly')}", popup=True)
        _grant_badge("weekly_champion", silent=True)
        # celebratory popup + sound
        show_challenge_popup(f"🎉 Challenge completed — {ch.get('title','Weekly')} (+{reward} XP)", play_sound=True)

    save_all_state()

def update_challenge_progress(action: str):
    """
    action in {'chat','login','manual'} — app.py triggers this.
    """
    if action == "manual":
        _bump_weekly("manual", 1)
    elif action == "login":
        _bump_weekly("login", 1)
    elif action == "chat":
        _bump_weekly("chat", 1)

# -------------------------------------------------
# UI Renders
# -------------------------------------------------
def _ring_gauge(value: float, max_value: float, label: str, color="#0FB38B", height=220):
    """
    Animated radial progress using Plotly gauge (fallback to st.progress if Plotly missing).
    """
    try:
        import plotly.graph_objects as go
        pct = 0 if max_value == 0 else max(0.0, min(1.0, float(value) / float(max_value)))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(value),
                gauge={
                    "axis": {"range": [None, float(max_value)]},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#ddd",
                    "steps": [],
                },
                title={"text": label},
            )
        )
        fig.update_layout(margin=dict(l=16, r=16, t=40, b=16), height=height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback to linear bar
        st.write(label)
        p = 0 if max_value == 0 else float(value) / float(max_value)
        st.progress(min(1.0, max(0.0, p)))

def render_progress_sidebar_full():
    """Compact, clean section for Progress/Challenges pages."""
    g = st.session_state.gamification
    xp = int(g["xp"])
    level = _level_from_xp(xp)
    next_level_xp_start = (level - 1) * 200
    to_next = max(0, 200 - (xp - next_level_xp_start))

    st.subheader("🏆 Progress")
    c1, c2 = st.columns(2)
    with c1:
        _ring_gauge(value=xp - next_level_xp_start, max_value=200, label=f"Level {level} — XP to next: {to_next}")
    with c2:
        st.metric("🔥 Login Streak", f"{g.get('daily_login_streak',0)} days")
        st.metric("💬 Total Chats", f"{g.get('chat_count',0)}")

    # Badges
    st.markdown("### 🏅 Badges")
    badges = g.get("badges", [])
    if not badges:
        st.info("No badges yet. Keep going! 💪")
    else:
        cols = st.columns(4)
        idx = 0
        for bid in badges:
            meta = BADGE_META.get(bid, {})
            label = meta.get("label", bid)
            path = meta.get("file")
            emoji = meta.get("emoji", "🏅")
            with cols[idx % 4]:
                if path and os.path.exists(path):
                    st.image(path, caption=label, use_container_width=True)
                else:
                    st.markdown(f"**{emoji} {label}**")
            idx += 1

def render_weekly_challenge_section():
    """Detailed weekly challenge with animated ring."""
    _ensure_weekly_challenge()
    g = st.session_state.gamification
    ch = g.get("weekly_challenge") or {}
    title = ch.get("title", "Weekly Challenge")
    target = int(ch.get("target", 1))
    progress = int(g.get("weekly_progress", 0))
    reward = int(ch.get("reward", 0))
    started = ch.get("started_on", date.today().isoformat())

    st.subheader("🎯 Weekly Challenge")
    st.write(f"**{title}**")
    st.caption(f"Started on: {started} • Reward: **+{reward} XP**")

    _ring_gauge(value=progress, max_value=target, label=f"Progress: {progress}/{target}")

    if g.get("challenge_completed"):
        st.success("✅ Completed — reward already credited!")
    else:
        st.info("Keep going! Complete it to earn XP and a badge.")
    # Hint to rotate weekly
    st.caption("New challenge will auto-refresh every 7 days.")

# -------------------------------------------------
# Popup (slide-in, soft sound)
# -------------------------------------------------
_SOFT_BEEP_BASE64 = (
    # A tiny, pleasant beep (22050Hz mono WAV ~short). Replace with your own if desired.
    "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)

def show_challenge_popup(message: str, duration_ms: int = 3200, play_sound: bool = False):
    """
    Slide-in toast (top-right). Optionally plays a soft sound (embedded).
    """
    uid = random.randint(100000, 999999)
    safe = str(message).replace("'", "\\'")
    audio_tag = f"<audio id='snd_{uid}' src='{_SOFT_BEEP_BASE64}' preload='auto'></audio>" if play_sound else ""

    html = f"""
    <div id="toast_{uid}" style="
        position: fixed; top: 20px; right: -400px; z-index: 9999;
        background: { '#0FB38B' }; color: white; padding: 14px 18px;
        border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.25);
        font-weight: 700; min-width: 260px; transition: right .35s ease, opacity .35s ease;">
      {safe}
    </div>
    {audio_tag}
    <script>
      const el = document.getElementById('toast_{uid}');
      el.style.right = '20px';
      setTimeout(() => {{
        el.style.opacity = 0;
        el.style.right = '-400px';
      }}, {duration_ms});
      setTimeout(() => el.remove(), {duration_ms} + 500);
      {'try{document.getElementById("snd_'+str(uid)+'").play();}catch(e){}' if play_sound else ''}
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)
    # tiny delay to ensure render sequence
    time.sleep(0.05)
