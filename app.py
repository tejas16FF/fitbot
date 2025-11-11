# app.py — FitBot (Desktop bottom nav + Mobile drawer, FAQ fixed, Tip once after profile)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# Gamification
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    render_progress_sidebar_full,
    render_weekly_challenge_section,
    show_challenge_popup,
    save_all_state,
    reset_progress_file,
)

load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

ACCENT = "#0FB38B"

# -----------------------------
# Session defaults
# -----------------------------
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
    st.session_state.page = "Profile"
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1_000_000, 9_999_999)
if "tip_after_profile" not in st.session_state:
    st.session_state.tip_after_profile = None

initialize_gamification()

# -----------------------------
# Knowledge base (fallback text)
# -----------------------------
DATA_FILE = "data.txt"
def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Regular exercise improves cardiovascular health and builds muscle strength."

KB_TEXT = load_kb()

def local_lookup_answer(query: str) -> str:
    q = query.lower()
    paras = [p.strip() for p in KB_TEXT.split("\n\n") if p.strip()]
    best, score = "", 0
    for p in paras:
        s = sum(1 for w in q.split() if w and w in p.lower())
        if s > score:
            score, best = s, p
    if score == 0:
        return "Stay consistent with training, eat a balanced diet, hydrate well, and prioritize sleep."
    return best

# -----------------------------
# Navigation — Desktop bottom bar & Mobile drawer
# -----------------------------
def nav_render():
    # CSS to show bottom bar only on >=768px; drawer button on <768px
    st.markdown(f"""
    <style>
      /* Bottom bar (desktop) */
      .bottom-nav {{
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
        background: rgba(0,0,0,0.03);
        backdrop-filter: blur(6px);
        border-top: 1px solid rgba(0,0,0,0.08);
        padding: 8px 12px;
        display: none;
      }}
      @media (min-width: 769px) {{
        .bottom-nav {{ display: block; }}
        .mobile-hamburger {{ display: none !important; }}
      }}
      @media (max-width: 768px) {{
        .bottom-nav {{ display: none; }}
        .mobile-hamburger {{ display: block; }}
      }}
      .btn-nav {{
        display:inline-block; margin: 0 6px; padding: 10px 12px;
        border-radius: 10px; border: 1px solid rgba(0,0,0,0.08);
        background: white;
        font-weight: 600; cursor: pointer; text-decoration:none; color:#111;
      }}
      .btn-nav.active {{ border-color: {ACCENT}; color: {ACCENT}; }}
      .mobile-hamburger {{
        position: fixed; top: 10px; left: 10px; z-index: 1000;
        background: white; border-radius: 10px; padding: 8px 12px; border: 1px solid rgba(0,0,0,0.08);
      }}
      /* Drawer */
      #drawer {{
        position: fixed; top: 0; left: 0; bottom: 0; width: 76%;
        max-width: 300px; background: #fff; z-index: 1200;
        box-shadow: 2px 0 20px rgba(0,0,0,0.15);
        transform: translateX(-105%);
        transition: transform .28s ease-in-out;
        padding: 16px;
      }}
      #drawer.open {{ transform: translateX(0%); }}
      .drawer-item {{
        display:block; padding: 12px 8px; margin: 6px 0; font-weight: 700; color: #111; border-radius:10px; border:1px solid rgba(0,0,0,0.06);
      }}
    </style>
    <div class="mobile-hamburger">
      <button onclick="document.getElementById('drawer').classList.add('open');">☰ Menu</button>
    </div>
    <div id="drawer">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong style="font-size:18px; color:{ACCENT};">FitBot</strong>
        <button onclick="document.getElementById('drawer').classList.remove('open');">✖</button>
      </div>
      <div style="margin-top:12px;">
        <form method="get">
          <button name="nav" value="Chat" class="drawer-item">🏠 Chat</button>
          <button name="nav" value="History" class="drawer-item">📜 History</button>
          <button name="nav" value="Challenges" class="drawer-item">🎯 Challenges</button>
          <button name="nav" value="Progress" class="drawer-item">🏆 Progress</button>
          <button name="nav" value="Profile" class="drawer-item">⚙️ Profile</button>
        </form>
      </div>
    </div>
    <div class="bottom-nav">
      <form method="get" style="text-align:center;">
        <button name="nav" value="Chat" class="btn-nav {'active' if st.session_state.page=='Chat' else ''}">🏠 Chat</button>
        <button name="nav" value="History" class="btn-nav {'active' if st.session_state.page=='History' else ''}">📜 History</button>
        <button name="nav" value="Challenges" class="btn-nav {'active' if st.session_state.page=='Challenges' else ''}">🎯 Challenges</button>
        <button name="nav" value="Progress" class="btn-nav {'active' if st.session_state.page=='Progress' else ''}">🏆 Progress</button>
        <button name="nav" value="Profile" class="btn-nav {'active' if st.session_state.page=='Profile' else ''}">⚙️ Profile</button>
      </form>
    </div>
    """, unsafe_allow_html=True)

    # Read nav param (works both desktop & drawer) and update page
    nav_target = st.query_params.get("nav")
    if nav_target:
        # prevent unintended jumps; only switch if user clicked
        st.session_state.page = nav_target
        # close drawer via small JS
        st.markdown("<script>const d=document.getElementById('drawer'); if(d){d.classList.remove('open');}</script>", unsafe_allow_html=True)
        # clear param so back button doesn't keep firing
        st.query_params.clear()
        st.rerun()

# -----------------------------
# FAQ
# -----------------------------
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

# -----------------------------
# Pages
# -----------------------------
def page_profile():
    st.title("🏋️ Create Your Fitness Profile")
    st.markdown("Let’s personalize FitBot for you. (No sidebars here; clean start.)")

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
        update_daily_login(silent=True)
        save_all_state()
        # Tip of the Day — show ONCE after profile submit (Option A)
        st.session_state.tip_after_profile = random.choice([
            "Consistency beats intensity — train smart and steady.",
            "Fuel your body well and your workouts will follow.",
            "Recovery is training — sleep, hydrate, stretch.",
        ])
        st.success("✅ Profile saved. Launching FitBot...")
        time.sleep(0.7)
        st.session_state.page = "Chat"
        st.rerun()

def page_chat():
    st.title("💬 Chat — FitBot")

    # One-time “Tip of the Day” after profile
    if st.session_state.tip_after_profile:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_after_profile}")
        st.session_state.tip_after_profile = None

    st.markdown("Ask me anything about workouts, diet, or motivation.")

    # FAQ buttons (always work; unique keys)
    cols = st.columns(3)
    for i, (label, q) in enumerate(list(FAQ_QUERIES.items())[:6]):
        if cols[i % 3].button(label, key=faq_key(i, label)):
            run_query(q)

    # Chat input
    user_q = st.chat_input("Ask FitBot your question:")
    if user_q:
        run_query(user_q)

    # Recent
    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {turn['user'][:72]}"):
                st.markdown(f"**A:** {turn['assistant']}")

def run_query(query: str):
    # Rotating motivational tip while loading (fades in/out)
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

def page_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No chats yet.")
        return
    for i, t in enumerate(reversed(st.session_state.history[-50:])):
        with st.expander(f"Q {i+1}: {t['user'][:70]}"):
            st.markdown(f"**Q:** {t['user']}")
            st.markdown(f"**A:** {t['assistant']}")
            st.caption(f"Time: {t.get('time')}s")

def page_challenges():
    st.title("🎯 Challenges")
    render_progress_sidebar_full()
    render_weekly_challenge_section()
    st.markdown("---")
    st.markdown("Manual actions (log to progress challenges):")
    c1, c2, c3 = st.columns(3)
    if c1.button("Log: Completed a workout (manual)"):
        update_challenge_progress("manual")
        save_all_state()
        st.success("Workout logged — progress updated.")
    if c2.button("Log: Did a check-in (manual)"):
        update_challenge_progress("login")
        save_all_state()
        st.success("Check-in logged — progress updated.")
    if c3.button("Claim weekly reward"):
        gam = st.session_state.gamification
        if gam.get("challenge_completed"):
            show_challenge_popup("🎉 Weekly reward already claimed. Great job!")
        else:
            st.info("Challenge not complete yet.")

def page_progress():
    st.title("🏆 Progress")
    render_progress_sidebar_full()
    st.markdown("### Badges")
    badges = st.session_state.gamification.get("badges", [])
    if not badges:
        st.info("No badges yet.")
    else:
        st.write(", ".join(badges))

# -----------------------------
# Main
# -----------------------------
def main():
    # Render nav (hidden drawer on mobile, bottom bar on desktop)
    if st.session_state.page != "Profile":
        nav_render()

    # Route
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
        page_chat()

if __name__ == "__main__":
    main()
