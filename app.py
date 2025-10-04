import os
import time
import streamlit as st
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, Union

# LangChain & vector tools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(".env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Full body workout — Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio — Running 30 mins, Jump rope 10 mins
Day 3: Upper body — Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High-protein diet, avoid processed sugars, hydrate 2-3L daily.
Recovery: Sleep 8 hrs, light stretching, yoga twice a week.
END FITNESS KB
"""

DAILY_TIPS = [
    "Tip: Hydrate often — your body performs better with enough fluids!",
    "Tip: Consistency beats intensity. Stick to your plan daily.",
    "Tip: Stretch before and after workouts to reduce injury risk.",
    "Tip: Track your progress — small wins matter!"
]

FAQ_QUERIES = {
    "💪 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Nutrition": "What are some post-workout meals for muscle gain?",
    "🧘 Flexibility": "Suggest a 10-minute daily stretching routine.",
    "⚡ Motivation": "Share some motivation tips to stay consistent.",
}

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": "",
        "weight": "",
        "goal": "General fitness",
        "level": "Beginner",
        "gender": "Prefer not to say",
        "diet": "No preference",
        "workout_time": "Morning"
    }

if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

if "initial_tip" not in st.session_state:
    st.session_state.initial_tip = random.choice(DAILY_TIPS)

# -----------------------------
# HELPER FUNCTIONS
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
You are FitBot, an AI fitness coach. Always respond politely and professionally.
Focus only on fitness, health, nutrition, and motivation.
If asked unrelated questions, gently steer the user back to fitness topics.

Profile: {profile}
Chat History: {chat_history}
Context: {context}
Question: {question}

Provide structured, detailed, and encouraging answers. Avoid technical jargon.
"""
    prompt = PromptTemplate(
        input_variables=["profile", "chat_history", "context", "question"],
        template=template,
    )
    return llm, LLMChain(llm=llm, prompt=prompt)

def retrieve_context(vectorstore, query: str, k=3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)

def format_history(history: List[Dict[str, Any]], limit=6) -> str:
    recent = history[-limit:]
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return "⚠️ Sorry, I'm having trouble answering right now. Please try again later."

# -----------------------------
# PAGE 1 — PROFILE SETUP
# -----------------------------
def page_profile():
    st.title("💪 Welcome to FitBot!")
    st.markdown("Let's personalize your AI fitness coach experience 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        age = st.number_input("Age", min_value=10, max_value=80, value=25)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])

        submitted = st.form_submit_button("Start FitBot")

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
        st.success("✅ Profile saved! Click on 'Chat with FitBot' in the menu.")
        st.rerun()

# -----------------------------
# PAGE 2 — MAIN CHAT
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # --- Layout with 3 columns (Left: Profile | Center: Chat | Right: History)
    col_profile, col_chat, col_history = st.columns([1.5, 3, 1.5])

    # ---- LEFT SIDEBAR PROFILE ----
    with col_profile:
        st.subheader("👤 Your Profile")
        for key, value in st.session_state.profile.items():
            st.markdown(f"**{key.capitalize()}**: {value}")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # ---- MIDDLE CHAT AREA ----
    with col_chat:
        st.subheader("🤖 Chat with FitBot")
        st.info(st.session_state.initial_tip)

        kb_text = read_knowledge_base("data.txt")
        vectorstore = build_vectorstore(kb_text)
        llm, chain = create_llm_chain(GOOGLE_KEY)

        # Quick Question Buttons
        cols = st.columns(len(FAQ_QUERIES))
        for i, (label, q) in enumerate(FAQ_QUERIES.items()):
            if cols[i].button(label):
                st.session_state["last_quick"] = q
                st.rerun()

        user_query = st.chat_input("Ask a question about workouts, nutrition, or motivation:")

        if user_query or "last_quick" in st.session_state:
            query = user_query or st.session_state.pop("last_quick")
            with st.spinner("💭 Thinking..."):
                start = time.time()
                answer = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
                duration = time.time() - start
            st.session_state.history.append({"user": query, "assistant": answer, "time": duration})
            st.success(answer)

        # Show last conversation
        if st.session_state.history:
            st.markdown("### 🗨️ Latest Exchange")
            last = st.session_state.history[-1]
            st.markdown(f"**You:** {last['user']}")
            st.markdown(f"**FitBot:** {last['assistant']}")
            st.caption(f"⏱️ Response time: {last['time']:.2f}s")

    # ---- RIGHT SIDEBAR HISTORY ----
    with col_history:
        st.subheader("📜 Conversation History")
        if not st.session_state.history:
            st.info("No previous questions yet.")
        else:
            for i, turn in enumerate(reversed(st.session_state.history)):
                with st.expander(f"🧩 {turn['user'][:50]}..."):
                    st.markdown(f"**Q:** {turn['user']}")
                    st.markdown(f"**A:** {turn['assistant']}")
                    st.caption(f"⏱️ {turn['time']:.2f}s")

            if st.button("🧹 Clear All History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# -----------------------------
# PAGE CONTROL
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Smart AI Fitness Coach | Capstone Project | Always consult professionals for medical advice.")
