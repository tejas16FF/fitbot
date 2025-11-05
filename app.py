# app.py — FitBot (Final Version: Left Profile + Right History + Gamification)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 🏆 Gamification system import
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    render_progress_sidebar,
)

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
FALLBACK_KB = "⚠️ Missing data.txt — please add your knowledge base file."

# -----------------------------
# DYNAMIC CONTENT
# -----------------------------
DAILY_TIPS = [
    "🏋️ Stay consistent — results come with patience.",
    "💧 Drink enough water daily to stay energized.",
    "🧠 Train your mind as much as your body.",
    "🥗 Fuel your body, don’t starve it.",
    "🔥 Progress is progress — even small steps count!",
    "🧘 Take deep breaths; stress kills gains.",
    "💪 Your only competition is your past self.",
    "🏃 Move more today than you did yesterday.",
    "😴 Recovery is part of training. Sleep well!",
    "💥 The body achieves what the mind believes.",
]

FAQ_QUERIES = {
    "🏋️ Beginner Workout": "Give me a 3-day beginner workout plan.",
    "🍎 Nutrition Tips": "Suggest a healthy balanced meal plan.",
    "💪 Motivation": "How can I stay consistent with workouts?",
    "🔥 Weight Loss": "What are effective fat-burning exercises?",
    "🧘 Flexibility": "Suggest a 10-minute morning yoga routine.",
    "😴 Sleep": "Why is rest important for recovery?",
    "🥗 Protein Sources": "List best vegetarian protein sources.",
    "🚶 Warm-up Ideas": "What are good warm-up exercises before workout?",
    "🍳 Pre-Workout Meal": "What should I eat before workout?",
    "🧍 Posture Tips": "How to maintain proper workout posture?",
}

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": 25,
        "weight": 70,
        "goal": "General fitness",
        "level": "Beginner",
        "gender": "Prefer not to say",
        "diet": "No preference",
        "workout_time": "Morning",
    }
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)

# -----------------------------
# HELPERS
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else FALLBACK_KB

@st.cache_resource(show_spinner=False)
def build_vectorstore(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    if not api_key:
        return None, None
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.3)
    template = """
You are FitBot, a professional and friendly AI fitness coach.
Use the user's profile data to give personalized responses.
Be motivational, polite, and never mention internal workings.

User Profile: {profile}
Conversation: {chat_history}
Context: {context}
Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["profile", "chat_history", "context", "question"])
    return llm, LLMChain(llm=llm, prompt=prompt)

def format_history(history: List[Dict[str, Any]], limit=6) -> str:
    recent = history[-limit:]
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])

def retrieve_context(vectorstore, query: str, k=3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

def generate_answer(chain: LLMChain, vectorstore, query: str, profile: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items())
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"⚠️ Model Error: {e}")
        return "Sorry, something went wrong while generating the answer."

# -----------------------------
# PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your fitness journey 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state.profile["weight"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        goal = st.selectbox("Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({
            "name": name,
            "age": age,
            "weight": weight,
            "gender": gender,
            "goal": goal,
            "level": level,
            "diet": diet,
            "workout_time": workout_time,
        })
        st.session_state.profile_submitted = True
        st.success("✅ Profile saved! Redirecting...")
        time.sleep(1)
        st.rerun()

# -----------------------------
# MAIN CHAT PAGE
# -----------------------------
def page_chat():
    initialize_gamification()
    update_daily_login()

    st.title("💬 FitBot — Your AI Fitness Assistant")

    # --- Layout Columns (Left = Profile, Center = Chat, Right = History)
    col_left, col_center, col_right = st.columns([1.2, 2.2, 1.2])

    # LEFT SIDEBAR — PROFILE + GAMIFICATION
    with col_left:
        st.header("👤 Your Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")

        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

        render_progress_sidebar()

    # CENTER — CHAT + FAQ
    with col_center:
        st.markdown("### 💡 Ask me about workouts, nutrition, recovery, or motivation")
        st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")

        kb_text = read_knowledge_base("data.txt")
        vectorstore = build_vectorstore(kb_text)
        llm, chain = create_llm_chain(GOOGLE_KEY)
        if not chain:
            st.error("❌ Gemini API not initialized. Check your API key.")
            return

        # FAQ Buttons
        st.markdown("#### ⚡ Quick Fitness Queries")
        faq_items = random.sample(list(FAQ_QUERIES.items()), 4)
        cols = st.columns(4)
        for i, (label, question) in enumerate(faq_items):
            key = f"faq_{i}_{random.randint(1000,9999)}"
            if cols[i].button(label, key=key):
                with st.spinner("🏋️ FitBot is thinking..."):
                    answer = generate_answer(chain, vectorstore, question, st.session_state.profile, st.session_state.history)
                    reward_for_chat()
                    st.session_state.history.append({"user": question, "assistant": answer})
                    st.success(answer)
                    st.experimental_rerun()

        # User Chat
        user_query = st.chat_input("Ask FitBot your question:")
        if user_query:
            with st.spinner(random.choice(DAILY_TIPS)):
                answer = generate_answer(chain, vectorstore, user_query, st.session_state.profile, st.session_state.history)
                st.session_state.history.append({"user": user_query, "assistant": answer})
                reward_for_chat()
                st.success(answer)
                st.experimental_rerun()

    # RIGHT SIDEBAR — CHAT HISTORY
    with col_right:
        st.header("📜 Chat History")
        if not st.session_state.history:
            st.info("No chats yet. Start asking 👇")
        else:
            for i, turn in enumerate(reversed(st.session_state.history[-10:])):
                with st.expander(f"Q{i+1}: {turn['user'][:40]}..."):
                    st.markdown(f"**Q:** {turn['user']}")
                    st.markdown(f"**A:** {turn['assistant']}")

# -----------------------------
# CONTROL FLOW
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Personalized AI Fitness Coach | XP, Badges, Challenges & Motivation 💪")
