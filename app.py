import os
import time
import streamlit as st
import random
from dotenv import load_dotenv
from typing import List

# LangChain & vector tools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage 

# --- Configuration ---
# Path for FAISS index persistence (Caches index to local disk for fast restarts)
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_FOLDER_PATH = "."

# Load environment variables
load_dotenv(".env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini API key
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# -----------------------------
# Helpful defaults / fallback KB (used only if data.txt missing)
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
3-day beginner workout:
Day 1: Squats (3x8), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio 20-30 min (brisk walk, cycling)
Day 3: Deadlifts (3x6), Lunges (3x8), Rows (3x8)

Post-workout meals:
- Protein + carbs within 60 minutes: e.g. chicken + rice, eggs + toast, protein shake + banana

Hydration:
- Aim for 2-3 liters of water daily; increase with intensive training.

Recovery:
- Sleep 7-9 hours; add mobility and light stretching on rest days.

Motivation tip:
- Set small achievable goals and track progress.

END FITNESS KB
"""

# -----------------------------
# Seeds: Daily Tips (Expanded for the "Wow" feature)
# -----------------------------
DAILY_TIPS = [
    "Tip: Drinking water before meals helps reduce calorie intake and keeps you hydrated!",
    "Tip: Focus on progressive overload to build muscle strength—lift slightly more, or do one extra rep next time!",
    "Tip: Try a dynamic warm-up before your workout to reduce injury risk and improve performance.",
    "Tip: Active recovery, like light stretching or walking on rest days, helps flush out soreness.",
    "Tip: Aim for 7-9 hours of quality sleep tonight. Your muscles repair and grow while you rest!",
    "Tip: Protein keeps you full longer. Ensure every meal has a high-protein source for better satiety and muscle support."
]

# -----------------------------
# Seeds: Frequently Asked Questions (Buttons)
# -----------------------------
FAQ_QUERIES = {
    "3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "Post-Workout Meal": "What is a good post-workout meal to support recovery?",
    "Protein Subs": "I am vegetarian. What are non-meat, high-protein foods I can eat?",
    "Motivation Tips": "Give me tips on how to stay consistent and motivated over the long term.",
}

# -----------------------------
# Session-state initialization
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "age": 25, "weight": 70, "goal": "Weight loss", "level": "Beginner", "gender": "Prefer not to say"} 

# New state for the two-page flow
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
    
if "initial_tip" not in st.session_state:
    st.session_state.initial_tip = random.choice(DAILY_TIPS)

# -----------------------------
# Helper: load knowledge base file
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return FALLBACK_KB

# -----------------------------
# Cached: build vectorstore (expensive) — uses disk cache for FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_from_text(text: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(FAISS_FOLDER_PATH) and any(f.startswith(FAISS_INDEX_PATH) for f in os.listdir(FAISS_FOLDER_PATH)):
        try:
            vectorstore = FAISS.load_local(FAISS_FOLDER_PATH, embeddings, FAISS_INDEX_PATH)
            return vectorstore
        except Exception:
            pass

    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_FOLDER_PATH, index_name=FAISS_INDEX_PATH)
    return vectorstore

# -----------------------------
# Cached: create LLM (Gemini) and LLMChain with prompt
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_llm_and_chain():
    if not GOOGLE_KEY:
        return None, None

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_KEY,
        temperature=0.2
    )

    template = """
You are FitBot, a professional and friendly AI fitness coach. Respond in a helpful, supportive, and safe manner. NEVER mention internal mechanics (like "knowledge base", "context", or "retrieved docs").
If very specific medical guidance is requested, give a general guideline and recommend consulting a professional.
If the question is completely out of scope, politely refuse and state your specialization: "I specialize in fitness and wellness."

User profile (if available): {profile}
Fitness Level: {level}
Gender: {gender}

Conversation so far:
{chat_history}

Relevant information:
{context}

User question:
{question}

Answer concisely but fully, with practical steps, optionally a short 1-2 line motivational ending.
If the user asks about diet and lists restrictions (e.g., vegetarian), provide substitutions.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["profile", "level", "gender", "chat_history", "context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return llm, chain

# -----------------------------
# Utility: make chat_history string for prompt
# -----------------------------
def format_history(history: List[dict], max_turns: int = 6) -> str:
    h = history[-max_turns:]
    lines = []
    for turn in h:
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
    return "\n".join(lines) if lines else "No previous conversation turns."

# -----------------------------
# Retrieve & answer (manual RAG Pipeline)
# -----------------------------
def retrieve_relevant_context(vectorstore, query: str, k: int = 3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context

def answer_query_pipeline(chain: LLMChain, vectorstore, query: str, profile: dict, history: List[dict]):
    context_query = f"{history[-1]['user'] if history else ''} {query}" 
    context = retrieve_relevant_context(vectorstore, context_query, k=3) 
    
    if not context.strip():
        context = "General fitness knowledge is used if specific context is unavailable."

    chat_history_str = format_history(history)
    profile_str = f"Name: {profile.get('name','')}; Age: {profile.get('age','')}; Weight: {profile.get('weight','')}; Goal: {profile.get('goal','')}"
    
    try:
        answer = chain.predict(
            profile=profile_str, 
            level=profile.get('level', 'Beginner'), 
            gender=profile.get('gender', 'Prefer not to say'), 
            chat_history=chat_history_str, 
            context=context, 
            question=query
        )
    except Exception as e:
        answer = "Sorry — I'm having trouble generating an answer right now. Please try again in a moment."
        st.error(f"LLM error: {e}")
    return answer

# -----------------------------
# Seeds: Frequently Asked Questions (Buttons)
# -----------------------------
FAQ_QUERIES = {
    "3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "Post-Workout Meal": "What is a good post-workout meal to support recovery?",
    "Protein Subs": "I am vegetarian. What are non-meat, high-protein foods I can eat?",
    "Motivation Tips": "Give me tips on how to stay consistent and motivated over the long term.",
}

# -----------------------------
# Main UI Logic
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

# --- Page 1: Profile Setup ---
if not st.session_state.profile_submitted:
    st.title("💪 Welcome to FitBot! Let's get started.")
    st.markdown("Please enter your details to get personalized fitness advice.")
    
    with st.form("profile_form"):
        st.subheader("Your Profile")
        name = st.text_input("Name")
        age = st.text_input("Age (Years)")
        weight = st.text_input("Weight (kg)")
        
        gender_options = ["Male", "Female", "Other", "Prefer not to say"]
        gender = st.selectbox("Gender", gender_options, index=3)
        
        goal_options = ["Muscle gain", "Weight loss", "Endurance", "General health"]
        goal = st.selectbox("Primary Goal", goal_options)
        
        level_options = ["Beginner", "Intermediate", "Advanced"]
        level = st.selectbox("Level", level_options)
        
        submitted = st.form_submit_button("Start Chatting!")
    
    if submitted and all([name, age, weight]):
        st.session_state.profile.update({
            "name": name,
            "age": age,
            "weight": weight,
            "goal": goal,
            "level": level,
            "gender": gender
        })
        st.session_state.profile_submitted = True
        st.rerun()
    elif submitted:
        st.error("Please fill in your name, age, and weight to continue.")

# --- Page 2: Main Chat Page ---
else:
    st.title("💪 FitBot — Your AI Fitness Assistant")
    st.caption("Retrieval-Augmented Generation (RAG) System | Gemini + HuggingFace + FAISS")

    # Sidebar: Dedicated to Conversation History
    with st.sidebar:
        st.subheader("💬 Conversation History")
        st.markdown("---")
        
        st.info(f"**Profile Set:** Goal: {st.session_state.profile['goal']} | Level: {st.session_state.profile['level']}")
        
        # Display history in the sidebar
        if not st.session_state.history:
            st.markdown(f"**FitBot:** Hello, {st.session_state.profile['name']}! I'm your AI fitness coach. **{st.session_state.initial_tip}** How can I support your goals today?")
        else:
            for turn in reversed(st.session_state.history): 
                with st.chat_message("user"):
                    st.markdown(turn['user'])
                with st.chat_message("assistant"):
                    st.markdown(turn['assistant'])
                st.caption(f"⏱️ Response Time: {turn.get('time', 0):.2f}s")
                
        # Optional: Button to clear history
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.initial_tip = random.choice(DAILY_TIPS)
            st.rerun()

    # Main Body: Active Conversation Area
    
    # Load RAG components (cached)
    if not GOOGLE_KEY:
        st.error("Setup Error: GOOGLE_API_KEY environment variable is missing. Please set it in your .env file.")
        st.stop()
        
    with st.spinner(f"Preparing RAG components: loading knowledge base and FAISS index..."):
        kb_text = read_knowledge_base("data.txt")
        vectorstore = build_vectorstore_from_text(kb_text)
        llm, llm_chain = create_llm_and_chain()
        
        if llm_chain is None:
            st.error("Setup Error: Gemini API key not configured or model failed to load.")
            st.stop()

    # Quick Start Buttons
    st.markdown("---")
    st.subheader("💡 Quick Start Questions")
    button_cols = st.columns(len(FAQ_QUERIES))
    btn_keys = list(FAQ_QUERIES.keys())
    for i, c in enumerate(button_cols):
        if i < len(btn_keys):
            q_label = btn_keys[i]
            if c.button(q_label):
                st.session_state["last_quick"] = FAQ_QUERIES[q_label]
                st.rerun() 
    st.markdown("---")

    # Main Chat Input (Triggered by Enter Key)
    initial_input = st.session_state.pop("last_quick", "") 
    user_query = st.chat_input("Ask FitBot your question (Press Enter to submit):", key="main_input")

    if user_query and user_query.strip():
        with st.spinner(f"🤔 Thinking with Gemini, retrieving context..."):
            start = time.time()
            resp = answer_query_pipeline(llm_chain, vectorstore, user_query, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        st.session_state.history.append({"user": user_query, "assistant": resp, "time": latency})
        st.rerun()
    elif initial_input: # If quick button was pressed, run pipeline with stored question
        with st.spinner(f"🤔 Thinking with Gemini, retrieving context..."):
            start = time.time()
            resp = answer_query_pipeline(llm_chain, vectorstore, initial_input, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        st.session_state.history.append({"user": initial_input, "assistant": resp, "time": latency})
        st.rerun()

    # The active conversation display (only shows the latest few turns for a clean main page)
    st.subheader("Your Conversation")
    if st.session_state.history:
        latest_turn = st.session_state.history[-1]
        with st.chat_message("user"):
            st.markdown(latest_turn['user'])
        with st.chat_message("assistant"):
            st.markdown(latest_turn['assistant'])
    else:
        st.markdown(f"**FitBot:** Hello, {st.session_state.profile['name']}! I'm your AI fitness coach. How can I support your goals today?")
        
    st.markdown("---")
    st.caption("FitBot — Capstone Project (RAG, Memory, Personalization). Always consult a licensed professional for medical issues.")
