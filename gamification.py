# gamification.py
import random
import streamlit as st
from datetime import datetime, timedelta

# ------------------------------------
# 🏆 FitBot Gamification System
# ------------------------------------

def initialize_gamification():
    """Initialize gamification state if missing."""
    if "gamification" not in st.session_state:
        st.session_state.gamification = {
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

def update_daily_login():
    """Handle daily login streak logic."""
    gamedata = st.session_state.gamification
    today = datetime.now().date()
    last = gamedata["last_login"]

    if last is None or today > last:
        # continue or reset streak
        if last and today - last == timedelta(days=1):
            gamedata["streak"] += 1
        else:
            gamedata["streak"] = 1
        gamedata["last_login"] = today
        add_xp(50, reason="🔥 Daily login streak active!")

def add_xp(amount, reason=""):
    """Add XP, check for level-up and badges."""
    gamedata = st.session_state.gamification
    gamedata["xp"] += amount
    st.toast(f"💪 +{amount} XP! {reason}")

    # Level-up logic
    while gamedata["xp"] >= gamedata["level"] * 200:
        gamedata["level"] += 1
        st.balloons()
        st.success(f"🎉 Level Up! You’re now Level {gamedata['level']} 💪")

    check_badges()

def reward_for_chat():
    """Reward XP for each chat and update weekly challenge."""
    add_xp(10, reason="💭 Great question!")
    update_challenge_progress("chat")

def check_badges():
    """Grant badges at milestones."""
    gamedata = st.session_state.gamification
    xp = gamedata["xp"]
    badges = gamedata["badges"]

    badge_list = [
        (100, "💬 Active Learner"),
        (300, "🏋️ Fitness Champ"),
        (500, "🔥 Consistency King"),
        (1000, "🎯 Peak Performer"),
        (1500, "💎 Limit Breaker"),
        (2000, "👑 Ultimate Trainer"),
    ]

    for threshold, badge in badge_list:
        if xp >= threshold and badge not in badges:
            badges.append(badge)
            st.toast(f"🏅 New Badge Unlocked: {badge}")

# ------------------------------------
# 🗓️ Weekly Challenge System
# ------------------------------------

def generate_weekly_challenge():
    """Generate a random weekly challenge."""
    challenges = [
        {"type": "chat", "target": 5, "reward": 100, "desc": "Ask 5 questions this week."},
        {"type": "login", "target": 3, "reward": 150, "desc": "Log in on 3 different days this week."},
        {"type": "streak", "target": 5, "reward": 200, "desc": "Maintain a 5-day streak."},
    ]
    return random.choice(challenges)

def check_and_reset_weekly_challenge():
    """Check if a week has passed since challenge started."""
    gamedata = st.session_state.gamification
    today = datetime.now().date()

    if gamedata["weekly_challenge"] is None or gamedata["challenge_start_date"] is None:
        gamedata["weekly_challenge"] = generate_weekly_challenge()
        gamedata["challenge_start_date"] = today
        gamedata["challenge_progress"] = 0
        gamedata["challenge_completed"] = False
    else:
        days_since_start = (today - gamedata["challenge_start_date"]).days
        if days_since_start >= 7:
            gamedata["weekly_challenge"] = generate_weekly_challenge()
            gamedata["challenge_start_date"] = today
            gamedata["challenge_progress"] = 0
            gamedata["challenge_completed"] = False

def update_challenge_progress(action_type):
    """Update progress toward current weekly challenge."""
    gamedata = st.session_state.gamification
    challenge = gamedata["weekly_challenge"]

    if not challenge or gamedata["challenge_completed"]:
        return

    if challenge["type"] == action_type:
        gamedata["challenge_progress"] += 1
        target = challenge["target"]
        st.toast(f"🎯 Challenge progress: {gamedata['challenge_progress']}/{target}")

        if gamedata["challenge_progress"] >= target:
            gamedata["challenge_completed"] = True
            add_xp(challenge["reward"], reason=f"🏆 Weekly Challenge Complete: {challenge['desc']}")

# ------------------------------------
# 🎨 UI Rendering
# ------------------------------------

def render_progress_sidebar():
    """Display progress, streaks, badges, and challenges."""
    gamedata = st.session_state.gamification
    xp = gamedata["xp"]
    level = gamedata["level"]
    streak = gamedata["streak"]
    badges = gamedata["badges"]

    st.markdown("---")
    st.subheader("🏆 Your Progress")
    st.progress((xp % 200) / 200)
    st.markdown(f"**Level:** {level}")
    st.markdown(f"**XP:** {xp}")
    st.markdown(f"🔥 Streak:** {streak} days**")

    # 🎖️ Badge Grid Display
    if badges:
        st.markdown("**🎖️ Badges:**")
        badge_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        for b in badges:
            badge_html += f"<div style='padding:6px 10px;border-radius:8px;background:rgba(0,191,166,0.1);border:1px solid rgba(0,191,166,0.4);color:#00BFA6;font-size:14px;font-weight:600;'>{b}</div>"
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

    # 🎯 Weekly Challenge Section
    check_and_reset_weekly_challenge()
    challenge = gamedata["weekly_challenge"]

    if challenge:
        st.markdown("---")
        st.subheader("🎯 Weekly Challenge")
        progress = gamedata["challenge_progress"]
        target = challenge["target"]
        st.markdown(f"**{challenge['desc']}**")
        st.progress(min(progress / target, 1.0))
        if gamedata["challenge_completed"]:
            st.success("✅ Challenge completed! Claim your reward below.")
        else:
            st.caption(f"Progress: {progress}/{target} • Reward: +{challenge['reward']} XP")
