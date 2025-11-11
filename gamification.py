# gamification.py — Expanded Gamification + Achievements (memory-only)
import streamlit as st
import time
import random
from datetime import date, datetime

# -----------------------------
# State Initialization
# -----------------------------
def init_gamification(profile=None):
    """Initialize gamification state once, then reuse."""
    if "xp" not in st.session_state:
        st.session_state.xp = 0
    if "level" not in st.session_state:
        st.session_state.level = 1
    if "streak" not in st.session_state:
        st.session_state.streak = 0
    if "last_login" not in st.session_state:
        st.session_state.last_login = None

    if "completed_challenges" not in st.session_state:
        st.session_state.completed_challenges = []
    if "active_challenge" not in st.session_state:
        st.session_state.active_challenge = None

    if "unlocked_achievements" not in st.session_state:
        st.session_state.unlocked_achievements = []

    # Behavior counters (for AI-driven achievements)
    if "nutrition_queries" not in st.session_state:
        st.session_state.nutrition_queries = 0
    if "workout_queries" not in st.session_state:
        st.session_state.workout_queries = 0
    if "motivation_queries" not in st.session_state:
        st.session_state.motivation_queries = 0

    # Update streak (once per day when profile is available)
    if profile and profile.get("name"):
        update_daily_streak()

# -----------------------------
# Core Rules: XP / Level / Streak
# -----------------------------
LEVEL_UP_THRESHOLDS = {
    1: 0,
    2: 200,
    3: 400,
    4: 800,
    5: 1500,
    6: 2500,
}

def gain_xp(amount: int, profile=None):
    st.session_state.xp += int(amount)
    _check_level_up(profile)

def _check_level_up(profile=None):
    current_xp = st.session_state.xp
    new_level = st.session_state.level
    for lvl, threshold in LEVEL_UP_THRESHOLDS.items():
        if current_xp >= threshold:
            new_level = lvl
    if new_level > st.session_state.level:
        st.session_state.level = new_level
        _toast(f"🎉 Level up! You’re now Level {new_level}.")
        check_all_achievements(profile)

def update_daily_streak():
    """Increment or reset daily streak based on last_login."""
    today = date.today()
    last = st.session_state.last_login
    if last:
        try:
            last_d = datetime.fromisoformat(last).date()
            delta = (today - last_d).days
            if delta == 1:
                st.session_state.streak += 1
            elif delta >= 1:
                st.session_state.streak = 1
        except Exception:
            st.session_state.streak = 1
    else:
        st.session_state.streak = 1
    st.session_state.last_login = today.isoformat()
    gain_xp(25)  # small XP for opening the app today

# -----------------------------
# Weekly Challenges
# -----------------------------
WEEKLY_CHALLENGES = [
    {"id":"steps_8k_3d","name":"Walk 8,000 steps daily (3 days)","xp":100},
    {"id":"push_20","name":"Complete 20 push-ups","xp":100},
    {"id":"water_3l","name":"Drink 3 litres of water today","xp":100},
    {"id":"hiit_15","name":"15-minute HIIT session","xp":100},
    {"id":"clean_24h","name":"Eat clean (no junk) for 24 hours","xp":100},
    {"id":"stretch_10","name":"Stretch for 10 minutes before bed","xp":100},
    {"id":"walk_5k","name":"Walk 5,000 steps today","xp":100},
    {"id":"core_10","name":"10-minute core workout","xp":100},
]

def _pick_uncompleted_challenge():
    done = set(st.session_state.completed_challenges)
    remaining = [c for c in WEEKLY_CHALLENGES if c["id"] not in done]
    return random.choice(remaining) if remaining else None

def complete_active_challenge(profile=None):
    ch = st.session_state.active_challenge
    if not ch:
        return
    if ch["id"] not in st.session_state.completed_challenges:
        st.session_state.completed_challenges.append(ch["id"])
        gain_xp(ch["xp"], profile=profile)
        _toast("✅ Challenge completed! +XP")
        check_all_achievements(profile)
    st.session_state.active_challenge = None

def render_challenges_page(profile=None):
    """Challenges UI — only appears on the Challenges page."""
    init_gamification(profile)
    st.title("🎯 Weekly Fitness Challenges")

    _render_progress_summary()

    if st.session_state.active_challenge is None:
        next_ch = _pick_uncompleted_challenge()
        if next_ch:
            st.session_state.active_challenge = next_ch

    ch = st.session_state.active_challenge
    st.markdown("---")
    if ch:
        st.subheader("🔸 Current Challenge")
        st.info(ch["name"])
        if st.button("✅ Mark as Completed", key="btn_complete_challenge"):
            complete_active_challenge(profile=profile)
            st.rerun()
    else:
        st.success("🎉 No active challenge. You’ve completed all available ones!")

    if st.session_state.completed_challenges:
        st.markdown("---")
        st.subheader("✅ Completed")
        for cid in st.session_state.completed_challenges:
            meta = next((c for c in WEEKLY_CHALLENGES if c["id"] == cid), None)
            if meta:
                st.markdown(f"- {meta['name']}")

# -----------------------------
# Achievements (Expanded)
# -----------------------------
# Rarity: common / rare / epic / legendary
ACHIEVEMENTS = [
    {"id":"first_xp","name":"🔥 First Steps","desc":"Earn your first XP",
     "rarity":"common","emoji":"🟢","xp_reward":10,
     "condition":lambda s,p: s["xp"]>=1},

    {"id":"level_2","name":"⚡ Level 2 Achieved","desc":"Reach Level 2",
     "rarity":"rare","emoji":"🔵","xp_reward":25,
     "condition":lambda s,p: s["level"]>=2},

    {"id":"level_3","name":"🏋️ Power Level","desc":"Reach Level 3",
     "rarity":"epic","emoji":"🟣","xp_reward":50,
     "condition":lambda s,p: s["level"]>=3},

    {"id":"challenge_1","name":"✅ Challenger","desc":"Complete your first challenge",
     "rarity":"rare","emoji":"🔵","xp_reward":25,
     "condition":lambda s,p: len(s["completed_challenges"])>=1},

    {"id":"challenge_3","name":"🔥 Consistency King","desc":"Complete 3 challenges",
     "rarity":"epic","emoji":"🟣","xp_reward":50,
     "condition":lambda s,p: len(s["completed_challenges"])>=3},

    {"id":"iron_will_7","name":"🛡️ Week Warrior","desc":"7-day streak",
     "rarity":"epic","emoji":"🟣","xp_reward":50,
     "condition":lambda s,p: s.get("streak",0)>=7},

    {"id":"iron_will_30","name":"🔥 Iron Will","desc":"30-day streak",
     "rarity":"legendary","emoji":"🟠","xp_reward":100,
     "condition":lambda s,p: s.get("streak",0)>=30},

    {"id":"goal_muscle","name":"💪 Muscle Ambition","desc":"Set goal: Muscle gain",
     "rarity":"common","emoji":"🟢","xp_reward":10,
     "condition":lambda s,p: (p or {}).get("goal")=="Muscle gain"},

    {"id":"goal_weight","name":"🔥 Fat Burner","desc":"Set goal: Weight loss",
     "rarity":"common","emoji":"🟢","xp_reward":10,
     "condition":lambda s,p: (p or {}).get("goal")=="Weight loss"},

    {"id":"nutrition_5","name":"🥗 Nutrition Ninja","desc":"Ask 5 diet/nutrition questions",
     "rarity":"rare","emoji":"🔵","xp_reward":25,
     "condition":lambda s,p: s.get("nutrition_queries",0)>=5},
]

def check_all_achievements(profile=None):
    """Evaluate all achievements; unlock if condition met."""
    state = {
        "xp": st.session_state.xp,
        "level": st.session_state.level,
        "completed_challenges": st.session_state.completed_challenges,
        "streak": st.session_state.streak,
        "nutrition_queries": st.session_state.nutrition_queries,
    }
    for ach in ACHIEVEMENTS:
        aid = ach["id"]
        if aid in st.session_state.unlocked_achievements:
            continue
        try:
            ok = ach["condition"](state, profile)
        except Exception:
            ok = False
        if ok:
            _unlock_achievement(ach)

def _unlock_achievement(ach):
    st.session_state.unlocked_achievements.append(ach["id"])
    st.session_state.xp += ach.get("xp_reward", 0)
    _achievement_toast(ach)

def render_achievements_page(profile=None):
    """A clean Achievements hub with progress + lists."""
    init_gamification(profile)
    check_all_achievements(profile)

    st.title("🏆 Achievements")

    total = len(ACHIEVEMENTS)
    unlocked = len(st.session_state.unlocked_achievements)
    pct = int((unlocked/total)*100) if total else 0

    st.subheader("Your Progress")
    st.progress(pct/100)
    st.caption(f"Unlocked: {unlocked}/{total} ({pct}%) • Level {st.session_state.level} • XP {st.session_state.xp}")
    st.markdown("---")

    st.subheader("🎉 Unlocked")
    any_unlocked = False
    for a in ACHIEVEMENTS:
        if a["id"] in st.session_state.unlocked_achievements:
            any_unlocked = True
            _render_badge(a, unlocked=True)
    if not any_unlocked:
        st.info("No achievements yet — keep going! 💪")

    st.markdown("---")
    st.subheader("🔒 Locked")
    for a in ACHIEVEMENTS:
        if a["id"] not in st.session_state.unlocked_achievements:
            _render_badge(a, unlocked=False)

# -----------------------------
# Small Helpers (UI + Toasts + Summary)
# -----------------------------
def _render_badge(a, unlocked: bool):
    opacity = 1.0 if unlocked else 0.35
    rarity_color = {"common":"#7bdcb5","rare":"#50b3ff","epic":"#b679ff","legendary":"#ff9f40"}.get(a["rarity"], "#cfe")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:8px 0;opacity:{opacity};">
            <div style="
                width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;
                background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);font-size:22px;">
                {a["emoji"]}
            </div>
            <div>
                <div style="font-weight:800">{a["name"]}</div>
                <div style="opacity:.8;font-size:13px">{a["desc"]}</div>
                <div style="margin-top:4px;font-size:12px;color:{rarity_color}">{a["rarity"].title()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _render_progress_summary():
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Level", st.session_state.level)
    with c2: st.metric("XP", st.session_state.xp)
    with c3: st.metric("🔥 Streak (days)", st.session_state.streak)

def _toast(message: str, duration_ms: int = 2600):
    uid = random.randint(100000, 999999)
    safe = str(message).replace("'", "\\'")
    html = f"""
    <div id="toast_{uid}" style="
      position: fixed; top: 20px; right: -420px; z-index: 9999;
      background: #0FB38B; color: #fff; padding: 12px 16px;
      border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.25);
      font-weight: 700; min-width: 240px; transition: right .35s ease, opacity .35s ease;">
      {safe}
    </div>
    <script>
      const el = document.getElementById('toast_{uid}');
      el.style.right = '18px';
      setTimeout(()=>{{ el.style.opacity=0; el.style.right='-420px'; }}, {duration_ms});
      setTimeout(()=>{{ el.remove(); }}, {duration_ms}+450);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True); time.sleep(0.05)

def _achievement_toast(ach):
    colors = {
        "common":"linear-gradient(135deg,#43e97b,#38f9d7)",
        "rare":"linear-gradient(135deg,#00c6ff,#0072ff)",
        "epic":"linear-gradient(135deg,#a18cd1,#fbc2eb)",
        "legendary":"linear-gradient(135deg,#f6d365,#fda085)",
    }
    bg = colors.get(ach["rarity"], "linear-gradient(135deg,#4facfe,#00f2fe)")
    uid = random.randint(100000,999999)
    html = f"""
    <div id="achv_{uid}" style="
      position: fixed; top: 20px; right: -420px; z-index: 10000;
      background: {bg}; color: #10131d; padding: 14px 18px;
      border-radius: 14px; box-shadow: 0 12px 30px rgba(0,0,0,.25);
      font-weight: 800; min-width: 260px; transition: right .35s ease, opacity .35s ease;">
      🏆 Achievement Unlocked — {ach["emoji"]} {ach["name"]} • +{ach.get("xp_reward",0)} XP
      <div style="font-weight:600;opacity:.85;font-size:12px;margin-top:4px">{ach["desc"]}</div>
    </div>
    <script>
      const el = document.getElementById('achv_{uid}');
      el.style.right = '18px';
      setTimeout(()=>{{ el.style.opacity=0; el.style.right='-420px'; }}, 3600);
      setTimeout(()=>{{ el.remove(); }}, 4100);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True); time.sleep(0.06)

# -----------------------------
# Behavior counters — call from app on each query
# -----------------------------
def record_query_metrics(user_query: str):
    """Count topical interest for AI-driven achievements."""
    q = (user_query or "").lower()
    if any(k in q for k in ["diet","calorie","protein","meal","nutrition","food","carb","fat"]):
        st.session_state.nutrition_queries += 1
    if any(k in q for k in ["workout","exercise","set","rep","plan","program","strength","hypertrophy","cardio"]):
        st.session_state.workout_queries += 1
    if any(k in q for k in ["motivation","discipline","consistency","mindset"]):
        st.session_state.motivation_queries += 1
