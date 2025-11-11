import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# Gamification imports
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    render_progress_sidebar_full,
    render_weekly_challenge_section,
    show_challenge_popup,
    save_all_state
)

load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

ACCENT = "#0FB38B"

# -------------------------
# Session Defaults
# -------------------------
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

# Load gamification
initialize_gamification()


# -------------------------
# Knowledge Base
# -------------------------
DATA_FILE = "data.txt"

def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Regular exercise improves cardiovascular health and builds muscle strength."

KB_TEXT = load_kb()

def local_lookup_answer(query: str) -> str:
    q = query.lower().strip()
    paras = [p.strip() for p in KB_TEXT.split("\n\n") if p.strip()]
    best = ""
    best_score = 0

    for p in paras:
        text_words = p.lower().split()
        q_words = q.split()

        score = sum(1 for w in q_words if w in text_words)

        if score > best_score:
            best_score = score
            best = p

    # Threshold check to avoid FAQ override
    if best_score < 2:
        return (
            "Here’s a tailored suggestion: stay consistent, balance your workouts, "
            "prioritize nutrition, hydration, and rest to support your goal."
        )

    return best


# -------------------------
# Navigation (TOP ONLY)
# -------------------------
def nav_render_top():
    st.markdown(f"""
    <style>
        .top-nav {{
            position: fixed;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999;
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(6px);
            width: fit-content;
            border-radius: 14px;
            padding: 10px 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        }}

        .top-nav button {{
            background: none;
            border: none;
            padding: 8px 14px;
            font-weight: 600;
            font-size: 15px;
            cursor: pointer;
            border-radius: 8px;
            transition: 0.2s;
        }}

        .top-nav button.active {{
            color: {ACCENT};
            border-bottom: 2px solid {ACCENT};
        }}

        /* Add padding to body so content doesn't hide behind nav bar */
        .stApp {{
            padding-top: 70px;
        }}
    </style>

    <div class="top-nav">
        <form method="get">
            <button name="nav" value="Chat" class="{'active' if st.session_state.page=='Chat' else ''}">🏠 Chat</button>
            <button name="nav" value="History" class="{'active' if st.session_state.page=='History' else ''}">📜 History</button>
            <button name="nav" value="Challenges" class="{'active' if st.session_state.page=='Challenges' else ''}">🎯 Challenges</button>
            <button name="nav" value="Progress" class="{'active' if st.session_state.page=='Progress' else ''}">🏆 Progress</button>
            <button name="nav" value="Profile" class="{'active' if st.session_state.page=='Profile' else ''}">⚙️ Profile</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

    nav_target = st.query_params.get("nav")

    # Switch page without looping to profile
    if nav_target and nav_target != st.session_state.page:
        st.session_state.page = nav_target
        st.query_params.clear()


# -------------------------
# FAQ
# -------------------------
FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Vegetarian Protein": "List high-protein vegetarian foods.",
    "🔥 Fat Loss Tips": "How can I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give a 10-minute morning yoga stretch routine.",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
}

def faq_key(i, label):
    return f"faq_{st.session_state.session_id}_{i}_{label.replace(' ','_')}"


# -------------------------
# Pages
# -------------------------
def page_profile():
    st.title("🏋️ Create Your Fitness Profile")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile.get("name", ""))
        age = st.number_input("Age", 10, 80, int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", 30, 200, int(st.session_state.profile.get("weight", 70)))
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"])
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])

        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({
            "name": name,
            "age": age,
            "weight": weight,
            "goal": goal,
            "level": level,
            "diet": diet
        })

        update_daily_login(silent=True)
        save_all_state()

        st.session_state.tip_after_profile = random.choice([
            "Consistency beats intensity — train smart and steady.",
            "Fuel your body well and your workouts will follow.",
            "Recovery is training — sleep, hydrate, stretch.",
        ])

        st.success("✅ Profile saved! Launching FitBot...")
        time.sleep(0.6)

        st.session_state.page = "Chat"


def page_chat():
    st.title("💬 FitBot — Chat")

    if st.session_state.tip_after_profile:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_after_profile}")
        st.session_state.tip_after_profile = None

    st.markdown("Ask me anything:")

    cols = st.columns(3)
    for i, (label, q) in enumerate(list(FAQ_QUERIES.items())):
        if cols[i % 3].button(label, key=faq_key(i, label)):
            run_query(q)

    user_q = st.chat_input("Your question…")
    if user_q:
        run_query(user_q)

    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {turn['user'][:60]}"):
                st.markdown(f"**A:** {turn['assistant']}")


def run_query(query: str):
    tips = [
        "Small steps daily lead to big wins.",
        "Hydration powers performance.",
        "Form first, then intensity.",
        "Recovery fuels growth.",
        "Discipline > motivation.",
    ]

    loading = st.empty()
    loading.markdown(f"""
    <div id="loading_tip" style="
        text-align:center;
        padding:10px;
        color:{ACCENT};
        font-weight:700;
        background:rgba(15,179,139,.06);
        border-radius:10px;">
        💭 {random.choice(tips)}
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.3)
    answer = local_lookup_answer(query)
    loading.empty()

    st.session_state.history.append({"user": query, "assistant": answer, "time": time.time()})

    try:
        reward_for_chat()
        update_challenge_progress("chat")
        save_all_state()
    except:
        pass

    st.success(answer)


def page_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No past questions yet.")
        return

    for i, t in enumerate(reversed(st.session_state.history[-50:])):
        with st.expander(f"Q {i+1}: {t['user'][:70]}"):
            st.markdown(f"**Q:** {t['user']}")
            st.markdown(f"**A:** {t['assistant']}")


def page_challenges():
    st.title("🎯 Challenges")
    render_progress_sidebar_full()
    render_weekly_challenge_section()


def page_progress():
    st.title("🏆 Progress Overview")
    render_progress_sidebar_full()


# -------------------------
# Main Controller
# -------------------------
def main():
    if st.session_state.page != "Profile":
        nav_render_top()

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
