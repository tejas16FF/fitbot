# app.py — FitBot (Stable UI, Local KB, Bottom Nav Desktop + Mobile Drawer)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    render_progress_sidebar_full,
    render_weekly_challenge_section,
    show_challenge_popup,
    save_all_state,
)

# ---------- Setup ----------
load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
ACCENT = "#0FB38B"

# ---------- Session Defaults ----------
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": 25,
        "weight": 70,
        "goal": "General fitness",
        "level": "Beginner",
        "diet": "No preference",
    }
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Profile"  # start at profile only once
if "profile_completed" not in st.session_state:
    st.session_state.profile_completed = False
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1_000_000, 9_999_999)
if "tip_after_profile" not in st.session_state:
    st.session_state.tip_after_profile = None
if "drawer_open" not in st.session_state:
    st.session_state.drawer_open = False

initialize_gamification()

# ---------- Knowledge Base ----------
DATA_FILE = "data.txt"
def load_kb() -> str:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    # safe fallback
    return (
        "Regular exercise improves cardiovascular health and builds muscle strength.\n"
        "A balanced diet should include proteins, carbohydrates, healthy fats, vitamins, and minerals.\n"
        "Hydration, sleep, and recovery are essential."
    )
KB_TEXT = load_kb()

def local_lookup_answer(query: str) -> str:
    """Lightweight keyword-based answer from data.txt."""
    q = query.lower()
    paras = [p.strip() for p in KB_TEXT.split("\n\n") if p.strip()]
    if not paras:
        return "Train consistently, eat a balanced diet, hydrate well, and prioritize sleep."
    best, score = "", 0
    for p in paras:
        s = sum(1 for w in q.split() if w and w in p.lower())
        if s > score:
            score, best = s, p
    if score == 0:
        return "Stay consistent with training, eat a balanced diet, hydrate well, and prioritize sleep."
    return best

# ---------- Navigation (Mobile Drawer + Desktop Bottom Bar) ----------
def render_mobile_drawer():
    """Top-left ☰ button toggles drawer (pure Streamlit, no query params)."""
    # Hamburger (visible on mobile via CSS below)
    ham_col = st.columns([0.15, 0.85])[0]
    if ham_col.button("☰ Menu", key="hamburger_btn"):
        st.session_state.drawer_open = True

    if st.session_state.drawer_open:
        st.markdown(
            """
            <style>
              .drawer-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,0.35); }
              .drawer-panel   { position: fixed; top:0; left:0; bottom:0; width: 76%;
                                max-width: 300px; background: #fff; z-index: 1001;
                                box-shadow: 2px 0 20px rgba(0,0,0,0.15); padding: 16px; }
              @media (min-width: 769px) { .mobile-only { display:none; } }
              @media (max-width: 768px) { .desktop-only { display:none; } }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="drawer-overlay"></div>', unsafe_allow_html=True)
        st.markdown('<div class="drawer-panel">', unsafe_allow_html=True)
        col_a, col_b = st.columns([0.7, 0.3])
        with col_a:
            st.markdown(f"### <span style='color:{ACCENT}'>FitBot</span>", unsafe_allow_html=True)
        with col_b:
            if st.button("✖ Close"):
                st.session_state.drawer_open = False
                st.rerun()

        st.markdown("---")
        # Drawer nav buttons
        if st.button("🏠 Chat", use_container_width=True):
            st.session_state.page = "Chat"; st.session_state.drawer_open = False; st.rerun()
        if st.button("📜 History", use_container_width=True):
            st.session_state.page = "History"; st.session_state.drawer_open = False; st.rerun()
        if st.button("🎯 Challenges", use_container_width=True):
            st.session_state.page = "Challenges"; st.session_state.drawer_open = False; st.rerun()
        if st.button("🏆 Progress", use_container_width=True):
            st.session_state.page = "Progress"; st.session_state.drawer_open = False; st.rerun()
        if st.button("⚙️ Profile", use_container_width=True):
            st.session_state.page = "Profile"; st.session_state.drawer_open = False; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def render_bottom_nav():
    """Desktop bottom nav bar — emoji buttons."""
    st.markdown(
        f"""
        <style>
          @media (max-width:768px) {{ .bottom-nav {{ display:none; }} }}
          .bottom-nav {{
            position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(6px);
            border-top: 1px solid rgba(0,0,0,0.08);
            padding: 6px 10px;
          }}
          .nav-btn {{
            width: 100%; padding: 10px 0; border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08); background: white;
            font-weight: 700; color: #111;
          }}
          .nav-active {{ border-color: {ACCENT}; color: {ACCENT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("🏠 Chat", key="nav_chat"):
            st.session_state.page = "Chat"; st.rerun()
        if c2.button("📜 History", key="nav_history"):
            st.session_state.page = "History"; st.rerun()
        if c3.button("🎯 Challenges", key="nav_challenges"):
            st.session_state.page = "Challenges"; st.rerun()
        if c4.button("🏆 Progress", key="nav_progress"):
            st.session_state.page = "Progress"; st.rerun()
        if c5.button("⚙️ Profile", key="nav_profile"):
            st.session_state.page = "Profile"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- FAQ ----------
FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Vegetarian Protein": "List high-protein vegetarian foods.",
    "🔥 Fat Loss Tips": "How can I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give a 10-minute morning yoga stretch routine.",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
}
def faq_key(i: int, label: str) -> str:
    return f"faq_{st.session_state.session_id}_{i}_{label.replace(' ','_')}"

# ---------- Pages ----------
def page_profile():
    st.title("🏋️ Create Your Fitness Profile")
    st.markdown("Let’s personalize FitBot for you. (Clean start — no sidebars here.)")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile.get("name", ""))
        age = st.number_input("Age", 10, 80, int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", 30, 200, int(st.session_state.profile.get("weight", 70)))
        goal = st.selectbox("Primary Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"], index=0)
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"], index=0)
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"], index=0)
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight,
            "goal": goal, "level": level, "diet": diet,
        })
        st.session_state.profile_completed = True
        update_daily_login(silent=True)
        save_all_state()
        # Tip of the Day — show ONCE after profile submit
        st.session_state.tip_after_profile = random.choice([
            "Consistency beats intensity — train smart and steady.",
            "Fuel your body well and your workouts will follow.",
            "Recovery is training — sleep, hydrate, stretch.",
        ])
        st.success("✅ Profile saved. Launching FitBot...")
        time.sleep(0.6)
        st.session_state.page = "Chat"
        st.rerun()

def page_chat():
    # Mobile drawer button (hidden on desktop via CSS)
    render_mobile_drawer()

    st.title("💬 Chat — FitBot")
    # One-time tip after profile submit
    if st.session_state.tip_after_profile:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_after_profile}")
        st.session_state.tip_after_profile = None

    st.markdown("Ask me anything about workouts, diet, or motivation.")

    # FAQ buttons (unique keys; always respond)
    cols = st.columns(3)
    for i, (label, q) in enumerate(list(FAQ_QUERIES.items())):
        if cols[i % 3].button(label, key=faq_key(i, label)):
            run_query(q)  # call directly; do not persist FAQ in state

    # Chat input — ensure FAQ memory never overrides typed question
    user_q = st.chat_input("Ask FitBot your question:")
    if user_q:
        if "selected_faq" in st.session_state:
            st.session_state.pop("selected_faq")
        run_query(user_q)

    # Recent Q&A
    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {turn['user'][:72]}"):
                st.markdown(f"**A:** {turn['assistant']}")

    # Desktop bottom nav (fixed)
    render_bottom_nav()

def run_query(query: str):
    """Answer a query with rotating loading tips; log to history; update gamification."""
    # Loading box with rotating tips
    tips = [
        "Small steps daily lead to big wins.",
        "Hydration powers your performance.",
        "Form first, then intensity.",
        "Recovery fuels growth.",
        "Discipline > motivation. Show up.",
    ]
    html = f"""
    <div id="motibox" style="text-align:center; margin:10px 0; padding:10px; border-radius:10px;
         color:{ACCENT}; background:rgba(15,179,139,.06); font-weight:700; transition:opacity .5s;">
      💭 {random.choice(tips)}
    </div>
    <script>
      const tips = {tips};
      let idx = 0;
      const box = document.getElementById('motibox');
      function nextTip() {{
        box.style.opacity = 0;
        setTimeout(()=>{{ box.innerText = '💭 ' + tips[idx]; box.style.opacity = 1; idx=(idx+1)%tips.length; }}, 350);
      }}
      const timer = setInterval(nextTip, 3000);
      setTimeout(()=>{{ clearInterval(timer); }}, 9000);
    </script>
    """
    ph = st.empty()
    ph.markdown(html, unsafe_allow_html=True)

    start = time.time()
    answer = local_lookup_answer(query)
    latency = round(time.time() - start, 2)
    ph.empty()

    st.session_state.history.append({"user": query, "assistant": answer, "time": latency})
    try:
        reward_for_chat(show_msg=False)
        update_challenge_progress("chat")
        save_all_state()
    except Exception:
        pass
    st.success(answer)

    # Keep bottom nav visible after answering on desktop
    render_bottom_nav()

def page_history():
    render_mobile_drawer()
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for i, t in enumerate(reversed(st.session_state.history[-50:])):
            with st.expander(f"Q {i+1}: {t['user'][:70]}"):
                st.markdown(f"**Q:** {t['user']}")
                st.markdown(f"**A:** {t['assistant']}")
                st.caption(f"Time: {t.get('time')}s")
    render_bottom_nav()

def page_challenges():
    render_mobile_drawer()
    st.title("🎯 Challenges")
    render_progress_sidebar_full()
    render_weekly_challenge_section()

    st.markdown("---")
    st.markdown("Manual actions (log to progress challenges):")
    c1, c2, c3 = st.columns(3)
    if c1.button("Log: Completed a workout (manual)"):
        update_challenge_progress("manual"); save_all_state(); st.success("Workout logged — progress updated.")
    if c2.button("Log: Did a check-in (manual)"):
        update_challenge_progress("login"); save_all_state(); st.success("Check-in logged — progress updated.")
    if c3.button("Claim weekly reward"):
        gam = st.session_state.gamification
        if gam.get("challenge_completed"):
            show_challenge_popup("🎉 Weekly reward already claimed. Great job!")
        else:
            st.info("Challenge not complete yet.")

    render_bottom_nav()

def page_progress():
    render_mobile_drawer()
    st.title("🏆 Progress")
    render_progress_sidebar_full()
    st.markdown("### Badges")
    badges = st.session_state.gamification.get("badges", [])
    if not badges:
        st.info("No badges yet.")
    else:
        st.write(", ".join(badges))
    render_bottom_nav()

# ---------- Main ----------
def main():
    # If page is Profile but user already completed it, keep them on Chat unless they explicitly go to Profile
    if st.session_state.page == "Profile" and st.session_state.profile_completed:
        # do nothing (they intentionally chose Profile) — leave as is
        pass

    page = st.session_state.page
    if page == "Profile":
        page_profile()
    elif page == "Chat":
        page_chat()
    elif page == "History":
        page_history()
    elif page == "Challenges":
        page_challenges()
    elif page == "Progress":
        page_progress()
    else:
        st.session_state.page = "Chat"
        page_chat()

if __name__ == "__main__":
    main()
