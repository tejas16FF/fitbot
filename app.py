import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    show_challenge_popup,
    save_all_state,
)

# -----------------------------
# Basic Setup
# -----------------------------
load_dotenv()
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

# Hide Streamlit menu, footer, and default nav
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ACCENT = "#0FB38B"
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_KEY:
    genai.configure(api_key=GOOGLE_KEY)

# -----------------------------
# Session State
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
    st.session_state.session_id = random.randint(100000, 999999)

if "tip_after_profile" not in st.session_state:
    st.session_state.tip_after_profile = None

initialize_gamification()

# -----------------------------
# Navigation Bar (TOP CENTER)
# -----------------------------
def render_top_nav():
    st.markdown(f"""
    <style>
      .nav-container {{
        position: fixed;
        top: 0;
        left: 0; right: 0;
        z-index: 1000;
        background: white;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        padding: 10px 5px;
        text-align: center;
      }}
      .nav-btn {{
        display: inline-block;
        margin: 0 8px;
        padding: 10px 15px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.08);
        font-weight: 600;
        cursor: pointer;
        background: white;
      }}
      .active-nav {{
        color: {ACCENT};
        border: 1px solid {ACCENT};
      }}
    </style>

    <div class="nav-container">
      <form method="get">
        <button name="nav" value="Chat" class="nav-btn {'active-nav' if st.session_state.page=='Chat' else ''}">🏠 Chat</button>
        <button name="nav" value="History" class="nav-btn {'active-nav' if st.session_state.page=='History' else ''}">📜 History</button>
        <button name="nav" value="Challenges" class="nav-btn {'active-nav' if st.session_state.page=='Challenges' else ''}">🎯 Challenges</button>
        <button name="nav" value="Progress" class="nav-btn {'active-nav' if st.session_state.page=='Progress' else ''}">🏆 Progress</button>
        <button name="nav" value="Profile" class="nav-btn {'active-nav' if st.session_state.page=='Profile' else ''}">⚙️ Profile</button>
      </form>
    </div>

    <div style="height:70px;"></div>
    """, unsafe_allow_html=True)

    nav_target = st.query_params.get("nav")
    if nav_target:
        st.session_state.page = nav_target
        st.query_params.clear()
        st.rerun()

# -----------------------------
# Build local embedding store
# -----------------------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DATA_FILE = "data.txt"
def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Fitness basics: workout consistently, eat healthy, hydrate, sleep well."

KB_TEXT = load_kb()

def build_embeddings():
    docs = [p.strip() for p in KB_TEXT.split("\n\n") if p.strip()]
    emb = EMBED_MODEL.encode(docs)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return docs, index

DOCS, FAISS_INDEX = build_embeddings()

def kb_search(query):
    v = EMBED_MODEL.encode([query])
    D, I = FAISS_INDEX.search(np.array(v), 1)
    return DOCS[I[0][0]]

# -----------------------------
# Gemini Query
# -----------------------------
def ask_gemini(query, fallback_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
You are FitBot. Use this knowledge context only to enhance accuracy:
----
{fallback_text}
----
User question: {query}
Provide a detailed helpful fitness answer.
"""
    resp = model.generate_content(prompt)
    return resp.text


# -----------------------------
# FAQ Buttons
# -----------------------------
FAQ_QUESTIONS = {
    "🏋️ Beginner plan": "Give me a 3 day beginner workout plan",
    "🔥 Fat loss": "How can I lose fat safely?",
    "🥗 Diet tips": "Give me healthy diet suggestions",
    "💪 Muscle gain": "How do I build muscle effectively?",
    "🍽️ Veg protein": "High protein vegetarian food?",
    "⌛ Warmups": "Give me dynamic warmup routine",
}

def faq_key(i, label):
    return f"faq_{st.session_state.session_id}_{i}"

# -----------------------------
# Profile Page
# -----------------------------
def page_profile():
    st.title("🏋️ Create Your Fitness Profile")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile["name"])
        age = st.number_input("Age", 10, 80, st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", 30, 200, st.session_state.profile["weight"])
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])

        ok = st.form_submit_button("Save & Continue")

    if ok:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight,
            "goal": goal, "level": level, "diet": diet
        })
        update_daily_login(silent=True)
        st.session_state.tip_after_profile = random.choice([
            "Consistency beats intensity — train smart and steady.",
            "Hydrate well, your body depends on it.",
            "Recovery is part of training.",
        ])
        st.session_state.page = "Chat"
        st.rerun()

# -----------------------------
# Chat Page
# -----------------------------
def page_chat():
    st.title("💬 Chat with FitBot")

    # Tip only once
    if st.session_state.tip_after_profile:
        st.info(f"💡 {st.session_state.tip_after_profile}")
        st.session_state.tip_after_profile = None

    # FAQ Buttons
    cols = st.columns(3)
    faq_list = list(FAQ_QUESTIONS.items())
    for i, (label, q) in enumerate(faq_list):
        if cols[i % 3].button(label, key=faq_key(i, label)):
            answer_query(q)

    # Manual Query
    user_q = st.chat_input("Ask me anything...")
    if user_q:
        answer_query(user_q)

    st.markdown("### Recent Questions")
    for item in st.session_state.history[-8:][::-1]:
        st.caption(f"**Q:** {item['user']}")
        st.write(item["assistant"])
        st.write("---")

# -----------------------------
# Answer Query
# -----------------------------
def answer_query(query):
    # Show rotating motivational tip (patched)
    html = """
    <div id="motibox" style="text-align:center; margin:10px 0; padding:10px; border-radius:10px;
         color:#0FB38B; background:rgba(15,179,139,.06); font-weight:700; transition:opacity .5s;">
      💭 Thinking...
    </div>

    <script>
      const tips = ["Small steps daily lead to big wins.",
                    "Hydration powers your performance.",
                    "Form first, then intensity.",
                    "Recovery fuels growth.",
                    "Discipline > motivation. Show up."];
      let idx = 0;
      const box = document.getElementById("motibox");
      function nxt() {
        box.style.opacity = 0;
        setTimeout(function() {
          box.innerHTML = "💭 " + tips[idx];
          box.style.opacity = 1;
          idx = (idx + 1) % tips.length;
        }, 350);
      }
      const t = setInterval(nxt, 3000);
      setTimeout(function(){ clearInterval(t); }, 9000);
    </script>
    """
    ph = st.empty()
    ph.markdown(html, unsafe_allow_html=True)

    kb_context = kb_search(query)
    answer = ask_gemini(query, kb_context)

    ph.empty()

    st.session_state.history.append({
        "user": query,
        "assistant": answer,
        "time": round(random.uniform(0.2, 1.2), 2)
    })
    reward_for_chat(False)
    update_challenge_progress("chat")
    save_all_state()

    st.success(answer)

# -----------------------------
# History
# -----------------------------
def page_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No past questions yet.")
        return

    for item in st.session_state.history[::-1]:
        with st.expander(f"Q: {item['user'][:50]}"):
            st.write(item["assistant"])

# -----------------------------
# Challenges
# -----------------------------
def page_challenges():
    st.title("🎯 Weekly Challenges")
    from gamification import (
        render_progress_sidebar_full, render_weekly_challenge_section
    )
    render_progress_sidebar_full()
    render_weekly_challenge_section()

# -----------------------------
# Progress Page
# -----------------------------
def page_progress():
    st.title("🏆 Progress Report")
    from gamification import render_progress_sidebar_full
    render_progress_sidebar_full()

# -----------------------------
# Main Routing
# -----------------------------
def main():
    if st.session_state.page != "Profile":
        render_top_nav()

    if st.session_state.page == "Profile":
        page_profile()
    elif st.session_state.page == "Chat":
        page_chat()
    elif st.session_state.page == "History":
        page_history()
    elif st.session_state.page == "Challenges":
        page_challenges()
    elif st.session_state.page == "Progress":
        page_progress()

if __name__ == "__main__":
    main()
