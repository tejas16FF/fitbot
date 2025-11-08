# app.py (updated to show progress only on Challenges page + popup)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv(".env")

# Import gamification utilities
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

st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

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

if "current_page" not in st.session_state:
    st.session_state.current_page = "Profile"

if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1000, 9999)

initialize_gamification()

# -----------------------------
# Knowledge base
# -----------------------------
DATA_FILE = "data.txt"
def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Regular exercise improves cardiovascular health and builds muscle strength."

KB_TEXT = load_kb()

# -----------------------------
# Simple fallback answer function
# -----------------------------
def local_lookup_answer(query: str) -> str:
    q = query.lower()
    paragraphs = [p.strip() for p in KB_TEXT.split("\n\n") if p.strip()]
    if not paragraphs:
        return "I don't have data right now — try a different question."
    best = None
    best_score = 0
    for p in paragraphs:
        score = sum(1 for w in q.split() if w and w in p.lower())
        if score > best_score:
            best_score = score
            best = p
    if best_score > 0:
        return best
    return "I couldn't find an exact match. General advice: be consistent, eat balanced, and recover well."

# -----------------------------
# UI helpers
# -----------------------------
def sidebar_nav():
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
    st.sidebar.markdown("---")
    # small profile summary
    st.sidebar.markdown(f"**Name:** {st.session_state.profile.get('name','—')}")
    st.sidebar.markdown(f"**Goal:** {st.session_state.profile.get('goal','—')}")
    st.sidebar.markdown("---")
    # Testing tools (optional)
    if st.sidebar.button("Reset progress (dev only)"):
        reset_progress_file()
        initialize_gamification()
        st.experimental_rerun()

# FAQ definitions
FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Vegetarian Protein": "List high-protein vegetarian foods.",
    "🔥 Fat Loss Tips": "How can I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give a 10-minute morning yoga stretch routine.",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout."
}

def faq_button_key(i: int, label: str) -> str:
    return f"faq_{st.session_state.session_id}_{i}_{label.replace(' ','_')}"

# -----------------------------
# Pages
# -----------------------------
def page_profile():
    st.title("🏋️ Profile")
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
            "name": name,
            "age": age,
            "weight": weight,
            "goal": goal,
            "level": level,
            "diet": diet,
        })
        # update login silently and persist
        update_daily_login(silent=True)
        save_all_state()
        st.success("Profile saved. Launching chat...")
        time.sleep(0.6)
        st.session_state.current_page = "Chat"
        st.experimental_rerun()

def page_chat():
    st.title("💬 Chat — FitBot")
    st.markdown("Ask me anything about workouts, diet, or motivation.")
    cols = st.columns(3)
    displayed = list(FAQ_QUERIES.items())[:3]
    for i, (label, q) in enumerate(displayed):
        if cols[i].button(label, key=faq_button_key(i, label)):
            handle_query(q)
    user_q = st.chat_input("Ask FitBot your question:")
    if user_q:
        handle_query(user_q)
    # show small history preview
    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-5:]):
            with st.expander(f"Q: {turn['user'][:60]}"):
                st.markdown(f"**A:** {turn['assistant']}")

def handle_query(query_text: str):
    placeholder = st.empty()
    # small loading indicator so user knows it's running
    with placeholder.container():
        st.info("💭 Generating answer — please wait...")
    start = time.time()
    # obtain answer (local fallback)
    answer = local_lookup_answer(query_text)
    latency = round(time.time() - start, 2)
    # clear placeholder
    placeholder.empty()
    # append history + persist
    st.session_state.history.append({"user": query_text, "assistant": answer, "time": latency})
    # reward chat xp, update weekly progress, persist
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
    # show full progress + challenge (only here)
    render_progress_sidebar_full()
    render_weekly_challenge_section()
    st.markdown("---")
    st.markdown("Manual actions (log to progress challenges):")
    col1, col2, col3 = st.columns(3)
    if col1.button("Log: Completed a workout (manual)"):
        update_challenge_progress("manual")
        save_all_state()
        st.success("Logged workout — progress updated.")
    if col2.button("Log: Did a check-in (manual)"):
        update_challenge_progress("login")
        save_all_state()
        st.success("Logged check-in — progress updated.")
    if col3.button("Claim weekly reward (if complete)"):
        gam = st.session_state.gamification
        if gam.get("challenge_completed"):
            # popup already shown by gamification when completed, but repeat gentle message
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
    sidebar_nav()
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
