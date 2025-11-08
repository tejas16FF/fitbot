# app.py — FitBot (Polished UI + Persistent Gamification + Fixed FAQ + Loading Screen)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# load environment
load_dotenv(".env")

# Optional heavy imports — fail gracefully
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANG_OK = True
except Exception:
    LANG_OK = False
    CharacterTextSplitter = None
    HuggingFaceEmbeddings = None
    FAISS = None
    ChatGoogleGenerativeAI = None

# Import gamification
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    render_progress_sidebar,
    save_all_state,
    update_challenge_progress,
)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")
DATA_FILE = "data.txt"
ACCENT_COLOR = "#0FB38B"

# -----------------------------
# INIT STATE
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

if "current_page" not in st.session_state:
    st.session_state.current_page = "Profile"

if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1000, 9999)

if "loading_transition" not in st.session_state:
    st.session_state.loading_transition = False

initialize_gamification()

# -----------------------------
# KNOWLEDGE BASE (fallback)
# -----------------------------
def load_kb(path="data.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Regular exercise improves health and builds strength."

KB_TEXT = load_kb(DATA_FILE)

# -----------------------------
# SIMPLE ANSWER FALLBACK
# -----------------------------
def local_lookup_answer(query: str) -> str:
    lines = [p for p in KB_TEXT.split("\n\n") if p.strip()]
    q = query.lower()
    for l in lines:
        if any(k in l.lower() for k in q.split()):
            return l
    return "Stay consistent with workouts, eat balanced meals, and rest well."

# -----------------------------
# BUILD LLM WRAPPER (if available)
# -----------------------------
def create_chain():
    if LANG_OK and GOOGLE_KEY:
        try:
            model = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_KEY, temperature=0.3)
            class Chain:
                def predict(self, **kwargs):
                    q = kwargs.get("question") or ""
                    return model.invoke(q).content if hasattr(model, "invoke") else str(model(q))
            return Chain()
        except Exception:
            return None
    return None

_chain = create_chain()

def get_answer(query: str):
    if _chain:
        try:
            return _chain.predict(question=query)
        except Exception:
            return local_lookup_answer(query)
    else:
        return local_lookup_answer(query)

# -----------------------------
# SIDEBAR
# -----------------------------
def sidebar():
    st.sidebar.title("FitBot Navigation")
    if st.sidebar.button("🏠 Chat"):
        st.session_state.current_page = "Chat"
    if st.sidebar.button("📜 History"):
        st.session_state.current_page = "History"
    if st.sidebar.button("🎯 Challenges"):
        st.session_state.current_page = "Challenges"
    if st.sidebar.button("🏆 Progress"):
        st.session_state.current_page = "Progress"
    if st.sidebar.button("⚙️ Profile"):
        st.session_state.current_page = "Profile"

    if st.session_state.current_page != "Profile" and not st.session_state.loading_transition:
        render_progress_sidebar()

# -----------------------------
# FAQ Section
# -----------------------------
FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Vegetarian Protein": "List high-protein vegetarian foods.",
    "🔥 Fat Loss Tips": "How can I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give a 10-minute morning yoga stretch routine.",
}

def faq_button_key(i: int, label: str) -> str:
    return f"faq_{st.session_state.session_id}_{i}_{label.replace(' ', '_')}"

# -----------------------------
# PAGES
# -----------------------------
def page_profile():
    st.title("🏋️ Create Your Fitness Profile")
    st.markdown("Enter a few details to personalize FitBot for you.")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile.get("name", ""))
        age = st.number_input("Age", 10, 80, st.session_state.profile.get("age", 25))
        weight = st.number_input("Weight (kg)", 30, 200, st.session_state.profile.get("weight", 70))
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({
            "name": name,
            "age": age,
            "weight": weight,
            "goal": goal,
            "level": level,
            "diet": diet,
        })
        save_all_state()
        update_daily_login(silent=True)
        st.session_state.loading_transition = True
        st.rerun()

    if st.session_state.loading_transition:
        # loading animation page
        st.markdown(
            f"""
            <div style='text-align:center; padding-top:100px;'>
                <h2>💪 Setting up your personalized FitBot...</h2>
                <p style='color:{ACCENT_COLOR}; font-weight:500;'>Loading workouts, nutrition data, and challenges...</p>
            </div>
            <script>
                setTimeout(function() {{
                    window.location.reload();
                }}, 2500);
            </script>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(2.5)
        st.session_state.loading_transition = False
        st.session_state.current_page = "Chat"
        st.rerun()

def page_chat():
    st.title("💬 Chat — FitBot")
    st.markdown("Ask me anything about workouts, diet, or motivation.")

    faq_cols = st.columns(3)
    for i, (label, q) in enumerate(FAQ_QUERIES.items()):
        key = faq_button_key(i, label)
        if faq_cols[i % 3].button(label, key=key):
            handle_query(q)

    user_q = st.chat_input("Ask FitBot your question:")
    if user_q:
        handle_query(user_q)

def handle_query(user_query: str):
    placeholder = st.empty()
    placeholder.info("💭 Generating answer... please wait.")
    start = time.time()
    answer = get_answer(user_query)
    latency = round(time.time() - start, 2)
    placeholder.empty()

    st.session_state.history.append({"user": user_query, "assistant": answer, "time": latency})
    reward_for_chat(show_msg=False)
    update_challenge_progress("chat")
    save_all_state()

    st.success(answer)

def page_history():
    st.title("📜 Chat History")
    if not st.session_state.history:
        st.info("No chats yet — start by asking FitBot something!")
        return
    for i, turn in enumerate(reversed(st.session_state.history[-20:])):
        with st.expander(f"Q: {turn['user'][:50]}"):
            st.markdown(f"**Q:** {turn['user']}")
            st.markdown(f"**A:** {turn['assistant']}")
            st.caption(f"Time: {turn['time']} sec")

def page_progress():
    st.title("🏆 Progress")
    render_progress_sidebar()

def page_challenges():
    st.title("🎯 Challenges")
    gam = st.session_state.gamification
    ch = gam.get("weekly_challenge", {})
    if not ch:
        st.info("No active weekly challenge yet.")
    else:
        st.markdown(f"**{ch.get('desc', 'No description')}**")
        st.progress(min(gam.get("challenge_progress", 0) / ch.get("target", 1), 1.0))
        if gam.get("challenge_completed"):
            st.success("✅ Challenge completed!")

# -----------------------------
# MAIN
# -----------------------------
def main():
    sidebar()
    page = st.session_state.current_page
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
