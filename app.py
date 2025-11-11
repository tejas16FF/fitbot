# ==========================
# FitBot — Final Stable Version (Gemini-1.5-Pro + HuggingFace + FAISS)
# ==========================

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Gamification Imports
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

# ---------------------------------------------
# Hide default Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

ACCENT = "#0FB38B"

# ---------------------------------------------
# ENV + Gemini setup
# ---------------------------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ---------------------------------------------
# Knowledge Base + Embeddings
# ---------------------------------------------
DATA_FILE = "data.txt"

@st.cache_resource
def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

KB_TEXT = load_kb()

# Load embedding model once
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# Build FAISS index once
@st.cache_resource
def build_faiss_kb(kb_text):
    docs = [x.strip() for x in kb_text.split("\n") if x.strip()]
    vectors = embedder.encode(docs, convert_to_numpy=True)
    dims = vectors.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(vectors)
    return docs, index

KB_DOCS, KB_INDEX = build_faiss_kb(KB_TEXT)

def retrieve_relevant_context(query):
    if not KB_DOCS:
        return ""
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = KB_INDEX.search(q_vec, 3)
    chunks = [KB_DOCS[idx] for idx in I[0] if idx < len(KB_DOCS)]
    return "\n".join(chunks)

# ---------------------------------------------
# Gemini LLM
# ---------------------------------------------
def ask_gemini(query, context):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={
                "max_output_tokens": 600,
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 40
            },
            safety_settings="block_none"
        )
        prompt = f"""
You are FitBot, a professional fitness coach.

Use the context to answer the user's question, but DO NOT repeat the context.
Be accurate, clear, and motivating. Avoid generic answers.

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ---------------------------------------------
# Session State Initialization
# ---------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Profile"

if "history" not in st.session_state:
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": 25,
        "weight": 70,
        "goal": "General fitness",
        "level": "Beginner",
        "diet": "No preference"
    }

initialize_gamification()

# ---------------------------------------------
# UI — Navigation Bar (top center)
# ---------------------------------------------
def navigation_bar():
    st.markdown(f"""
    <style>
    .nav-container {{
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
        margin-top: 10px;
    }}
    .nav-btn {{
        background: white;
        padding: 10px 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        font-weight: 600;
        cursor: pointer;
    }}
    .nav-active {{
        border-color: {ACCENT};
        color: {ACCENT};
        font-weight: 800;
    }}
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    buttons = ["Chat", "History", "Challenges", "Progress", "Profile"]

    for i, label in enumerate(buttons):
        active = "nav-active" if st.session_state.page == label else ""
        if cols[i].button(f"{label}", key=f"nav_{label}"):
            st.session_state.page = label
            st.rerun()

# ---------------------------------------------
# FAQ
# ---------------------------------------------
FAQ = {
    "🏋️ 3-Day Plan": "Create a 3-day beginner workout plan.",
    "🥗 Post-workout": "What should I eat after workout?",
    "💪 Veg Protein": "List vegetarian high-protein foods.",
    "🔥 Burn Fat": "How to reduce body fat safely?",
    "🧘 Morning Stretch": "Give a 10-minute yoga stretch.",
    "🚶 Warm-up": "Suggest warm-up exercises."
}

# ---------------------------------------------
# Chat Logic
# ---------------------------------------------
def answer_query(query):
    context = retrieve_relevant_context(query)
    answer = ask_gemini(query, context)

    st.session_state.history.append({
        "user": query,
        "assistant": answer,
        "time": round(time.time(), 2)
    })

    reward_for_chat(show_msg=False)
    update_challenge_progress("chat")
    save_all_state()

    st.success(answer)

# ---------------------------------------------
# PAGES
# ---------------------------------------------
def page_profile():
    st.title("🏋️ Your Fitness Profile")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile["name"])
        age = st.number_input("Age", 10, 80, st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", 30, 250, st.session_state.profile["weight"])
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"])
        level = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        submit = st.form_submit_button("Save & Continue")

    if submit:
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
        st.session_state.page = "Chat"
        st.rerun()

def page_chat():
    navigation_bar()
    st.title("💬 Chat with FitBot")

    cols = st.columns(3)
    faq_list = list(FAQ.items())

    for i in range(len(faq_list)):
        label, q = faq_list[i]
        if cols[i % 3].button(label, key=f"faq_{i}"):
            answer_query(q)

    user_q = st.chat_input("Ask anything about fitness, nutrition or workouts:")
    if user_q:
        answer_query(user_q)

    st.write("### Recent Conversations")
    for turn in reversed(st.session_state.history[-6:]):
        with st.expander(f"{turn['user']}"):
            st.write(turn["assistant"])

def page_history():
    navigation_bar()
    st.title("📜 History")

    if not st.session_state.history:
        st.info("No history yet.")
        return

    for turn in reversed(st.session_state.history):
        with st.expander(turn["user"]):
            st.write(turn["assistant"])

def page_challenges():
    navigation_bar()
    st.title("🎯 Weekly Challenges")
    render_progress_sidebar_full()
    render_weekly_challenge_section()

def page_progress():
    navigation_bar()
    st.title("🏆 Your Progress")
    render_progress_sidebar_full()

# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
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
