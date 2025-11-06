# app.py — FitBot with persistent gamification (uses gamification.py)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain / embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Gamification (persistent)
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    render_progress_sidebar,
    save_all_state,
)

# -----------------------------
# Config
# -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

FALLBACK_KB = "Knowledge base not found. Add data.txt to project root."

# Tips and FAQ pool
DAILY_TIPS = [
    "🏋️ Stay consistent — results come with patience.",
    "💧 Drink water — hydrate for performance.",
    "🧠 Small daily wins compound to big results.",
    "🥗 Eat whole foods most of the time.",
    "😴 Recovery matters — prioritize sleep.",
]
FAQ_QUERIES = {
    "🏋️ Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-Workout Meal": "What should I eat after a workout for recovery?",
    "💪 Vegetarian Protein": "List vegetarian high-protein foods.",
    "🔥 Fat Loss Tips": "How do I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give me a 10-minute morning yoga routine.",
    "🚶 Warm-up": "Suggest dynamic warm-up exercises before workouts.",
}

# -----------------------------
# Session init
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "age": 25, "weight": 70, "goal": "General fitness", "level": "Beginner", "gender": "Prefer not to say", "diet": "No preference", "workout_time": "Morning"}
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1, 10**9)

# -----------------------------
# LangChain helpers
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else FALLBACK_KB

@st.cache_resource(show_spinner=False, ttl=600)
def build_vectorstore(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    if not api_key:
        return None, None
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.25)
    template = """
You are FitBot, an expert and friendly AI fitness coach.
Use user's profile and recent chat history to give tailored, safe advice.
Profile: {profile}
History: {chat_history}
Context: {context}
Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["profile", "chat_history", "context", "question"])
    return llm, LLMChain(llm=llm, prompt=prompt)

def format_history(history: List[Dict[str, Any]], limit=8) -> str:
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
        st.error(f"Model error: {e}")
        return "Sorry — I couldn't generate an answer right now."

# -----------------------------
# UI sections
# -----------------------------
def page_profile():
    st.title("🏋️ Profile")
    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
        age = st.number_input("Age", min_value=10, max_value=80, value=int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=int(st.session_state.profile.get("weight", 70)))
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], index=["Male","Female","Other","Prefer not to say"].index(st.session_state.profile.get("gender","Prefer not to say")))
        goal = st.selectbox("Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"], index=["Weight loss","Muscle gain","Endurance","General fitness"].index(st.session_state.profile.get("goal","General fitness")))
        level = st.selectbox("Experience Level", ["Beginner","Intermediate","Advanced"], index=["Beginner","Intermediate","Advanced"].index(st.session_state.profile.get("level","Beginner")))
        diet = st.selectbox("Diet Preference", ["No preference","Vegetarian","Vegan","Non-vegetarian"], index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile.get("diet","No preference")))
        workout_time = st.selectbox("Preferred Workout Time", ["Morning","Afternoon","Evening"], index=["Morning","Afternoon","Evening"].index(st.session_state.profile.get("workout_time","Morning")))
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({"name": name, "age": age, "weight": weight, "gender": gender, "goal": goal, "level": level, "diet": diet, "workout_time": workout_time})
        st.session_state.profile_submitted = True
        # initialize and load persistent gamification (this will also load saved profile/history into session)
        initialize_gamification()
        update_daily_login()
        # immediate save of full state to disk
        save_all_state()
        st.success("Profile saved and persisted.")
        time.sleep(1)
        st.rerun()

def section_chat():
    st.title("💬 Chat")
    st.info(f"💡 Tip: {st.session_state.tip_of_the_day}")
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)
    if not chain:
        st.error("Gemini API not initialized.")
        return

    st.subheader("⚡ Quick Queries")
    faq_pool = list(FAQ_QUERIES.items())
    n = min(4, len(faq_pool))
    faq_items = random.sample(faq_pool, n)
    cols = st.columns(n)
    for i, (label, q) in enumerate(faq_items):
        if cols[i].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            with st.spinner("Thinking..."):
                ans = generate_answer(chain, vectorstore, q, st.session_state.profile, st.session_state.history)
                st.session_state.history.append({"user": q, "assistant": ans, "time": time.time()})
                reward_for_chat()
                # persist after adding
                save_all_state()
                st.success(ans)

    query = st.chat_input("Ask your question:")
    if query:
        with st.spinner(random.choice(DAILY_TIPS)):
            ans = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
            st.session_state.history.append({"user": query, "assistant": ans, "time": time.time()})
            reward_for_chat()
            save_all_state()
            st.success(ans)

def section_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No history available yet.")
        return
    for i, turn in enumerate(reversed(st.session_state.history[-30:])):
        with st.expander(f"Q{i+1}: {turn['user'][:80]}"):
            st.markdown(f"**Q:** {turn['user']}")
            st.markdown(f"**A:** {turn['assistant']}")
            st.caption(f"Time: {turn.get('time', 0):.2f}s")

def section_challenges():
    st.title("🏆 Challenges")
    gam = st.session_state.gamification
    check_and = None
    try:
        # check_and_reset_weekly_challenge exists in gamification; call indirectly by rendering sidebar
        pass
    except Exception:
        pass
    render_progress_sidebar()  # reuse UI
    if gam.get("weekly_challenge"):
        st.markdown("**Current challenge:**")
        ch = gam.get("weekly_challenge")
        st.markdown(f"- {ch.get('desc')} (Progress: {gam.get('challenge_progress',0)}/{ch.get('target')})")
    st.caption("Complete challenges to earn bonus XP. Progress persists between runs.")

def section_progress():
    st.title("🎖 Progress")
    render_progress_sidebar()

def section_statistics():
    st.title("📊 Statistics")
    gam = st.session_state.gamification
    # XP timeline: use history of saved snapshots — we'll synthesize from history length as simple demo
    try:
        # Build a simple XP-over-time sample from history: assume +10 per chat + 50 for daily logins recorded in gam
        xp = gam.get("xp", 0)
        level = gam.get("level", 1)
        st.markdown(f"**Current XP:** {xp}  •  **Level:** {level}")
        # Basic metrics
        completed = sum(1 for _ in gam.get("badges", []))
        st.markdown(f"**Badges unlocked:** {completed}")
        st.markdown(f"**Streak:** {gam.get('streak',0)} days")
    except Exception as e:
        st.warning(f"Could not render statistics: {e}")

# -----------------------------
# Sidebar navigation
# -----------------------------
def sidebar():
    st.sidebar.header("👤 Profile")
    for k, v in st.session_state.profile.items():
        st.sidebar.markdown(f"**{k.capitalize()}**: {v}")
    if st.sidebar.button("Edit Profile"):
        st.session_state.profile_submitted = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigate")
    if st.sidebar.button("Chat"):
        st.session_state.current_page = "Chat"
    if st.sidebar.button("History"):
        st.session_state.current_page = "History"
    if st.sidebar.button("Challenges"):
        st.session_state.current_page = "Challenges"
    if st.sidebar.button("Progress"):
        st.session_state.current_page = "Progress"
    if st.sidebar.button("Statistics"):
        st.session_state.current_page = "Statistics"

    st.sidebar.markdown("---")
    st.sidebar.caption("FitBot — Persistent Progress")

# -----------------------------
# Control flow
# -----------------------------
# On first run (or after profile saved), initialize persistent gamification
if "initialized_persistence" not in st.session_state:
    initialize_gamification()
    update_daily_login()
    st.session_state.initialized_persistence = True
    # ensure profile/history/gamification persisted on first run
    save_all_state()

if not st.session_state.profile_submitted:
    page_profile()
else:
    sidebar()
    page = st.session_state.current_page or "Chat"
    if page == "Chat":
        section_chat()
    elif page == "History":
        section_history()
    elif page == "Challenges":
        section_challenges()
    elif page == "Progress":
        section_progress()
    elif page == "Statistics":
        section_statistics()

st.markdown("---")
st.caption("FitBot — Persistent Progress. Saved to user_progress.json")
