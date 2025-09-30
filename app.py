import os
import time
import streamlit as st
import random # Added for Daily Tip
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

# -----------------------------
# Load environment variables
# -----------------------------
# Edit this path if your .env is elsewhere
load_dotenv(".env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini API key
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
# Session-state initialization
# -----------------------------
if "history" not in st.session_state:
    # history: list of dictionaries representing conversation turns
    st.session_state.history = []

if "profile" not in st.session_state:
    # Initialize full profile, including level for prompt clarity
    st.session_state.profile = {"name": "", "age": 25, "weight": 70, "goal": "Weight loss", "level": "Beginner"} 

if "vectorstore_built" not in st.session_state:
    st.session_state.vectorstore_built = False
    
if "initial_tip" not in st.session_state:
    st.session_state.initial_tip = random.choice(DAILY_TIPS)

# -----------------------------
# Helper: load knowledge base file
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # fallback 
    return FALLBACK_KB

# -----------------------------
# Cached: build vectorstore (expensive) — cached so it doesn't rebuild on each interaction
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_from_text(text: str):
    """
    Splits text into chunks, creates embeddings using MiniLM, and builds FAISS index.
    Returns: FAISS vectorstore object
    """
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
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
        temperature=0.2  # lower temperature for reliable answers
    )

    # Prompt template includes profile and chat_history (so bot can personalize & remember)
    template = """
You are FitBot, a professional and friendly AI fitness coach. Respond in a helpful, supportive,
and safe manner. NEVER mention internal mechanics (like "knowledge base", "context", or "retrieved docs").
If very specific medical guidance is requested, give a general guideline and recommend consulting a professional.
If the question is completely out of scope, politely refuse and state your specialization: "I specialize in fitness and wellness."


User profile (if available): {profile}
Fitness Level: {level}

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
        input_variables=["profile", "level", "chat_history", "context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return llm, chain

# -----------------------------
# Utility: make chat_history string for prompt
# -----------------------------
def format_history(history: List[dict], max_turns: int = 6) -> str:
    """Format last N turns as a compact chat history string."""
    h = history[-max_turns:]
    lines = []
    for turn in h:
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
    return "\n".join(lines) if lines else "No previous conversation turns."

# -----------------------------
# Retrieve & answer (manual RAG — we control variables passed to LLM)
# -----------------------------
def retrieve_relevant_context(vectorstore, query: str, k: int = 3) -> str:
    """
    Return a single text block concatenating top-k retrieved chunks.
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    # join with separators to preserve chunk boundaries
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context

def answer_query_pipeline(chain: LLMChain, vectorstore, query: str, profile: dict, history: List[dict]):
    """Retrieve context, format prompt inputs, call LLM chain, and return answer string."""
    # 1) retrieve context based on the new query + potentially last turn of history
    # Augment query with last turn for better retrieval, especially for follow-up questions
    context_query = f"{history[-1]['user'] if history else ''} {query}" 
    context = retrieve_relevant_context(vectorstore, context_query, k=3) 
    
    if not context.strip():
        context = "General fitness knowledge is used if specific context is unavailable."

    # 2) format history & profile
    chat_history_str = format_history(history)
    profile_str = f"Name: {profile.get('name','')}; Age: {profile.get('age','')}; Weight: {profile.get('weight','')}; Goal: {profile.get('goal','')}"
    
    try:
        # 3) run chain (execute LLM)
        answer = chain.predict(
            profile=profile_str, 
            level=profile.get('level', 'Beginner'), # Pass level for structured advice
            chat_history=chat_history_str, 
            context=context, 
            question=query
        )
    except Exception as e:
        # friendly fallback if LLM/API fails
        answer = "Sorry — I'm having trouble generating an answer right now. Please try again in a moment."
        st.error(f"LLM error: {e}")
    return answer

# -----------------------------
# Seeds: Quick Replies for Demo
# -----------------------------
PREDEFINED_QUERIES = {
    "3-day workout plan": "Give me a 3-day beginner full-body workout plan.",
    "Post-workout meal idea": "What is a good post-workout meal to support recovery?",
    "Vegetarian protein sub": "I am vegetarian. What are non-meat, high-protein foods I can eat?",
    "Injury advice (Knee)": "What are safe, general guidelines for someone with knee pain (non-medical advice)?"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
st.title("💪 FitBot — Your AI Fitness Assistant")
st.caption("Retrieval-Augmented Generation (RAG) System | Gemini + HuggingFace + FAISS")

# --- Profile form (Critical for Personalization) ---
profile_cols = st.columns([1,1,1,1,1])
with profile_cols[0]:
    name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
with profile_cols[1]:
    age = st.text_input("Age", value=st.session_state.profile.get("age", 25))
with profile_cols[2]:
    weight = st.text_input("Weight (kg)", value=st.session_state.profile.get("weight", 70))
with profile_cols[3]:
    goal = st.selectbox("Primary Goal", ["Muscle gain", "Weight loss", "Endurance", "General health"],
                         index=["Muscle gain", "Weight loss", "Endurance", "General health"].index(st.session_state.profile.get("goal", "Weight loss")))
with profile_cols[4]:
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"],
                        index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.profile.get("level", "Beginner")))

# Auto-update profile state on input change
st.session_state.profile.update({"name": name, "age": age, "weight": weight, "goal": goal, "level": level})
st.info(f"**Profile Set:** Goal: **{st.session_state.profile['goal']}** | Level: **{st.session_state.profile['level']}**. Ask a question for personalized advice.")


# --- Quick Buttons (Enrichment Feature) ---
st.markdown("##### Quick Questions for Demo:")
cols = st.columns(len(PREDEFINED_QUERIES))
btn_keys = list(PREDEFINED_QUERIES.keys())
for i, c in enumerate(cols):
    if i < len(btn_keys):
        q_label = btn_keys[i]
        if c.button(q_label):
            st.session_state["last_quick"] = PREDEFINED_QUERIES[q_label]
            # FIX: Use st.rerun() instead of st.experimental_rerun()
            st.rerun() 


# --- Load KB and Models (Cached) ---
with st.spinner("Preparing RAG components: loading knowledge base, embeddings, and Gemini model..."):
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore_from_text(kb_text)
    llm, llm_chain = create_llm_and_chain()
    if llm is None or llm_chain is None:
        st.error("Setup Error: Gemini API key not configured or model failed to load. Check GOOGLE_API_KEY.")
        st.stop()


# --- Main Chat Input ---
# Use the buffer if a quick button was clicked, otherwise use the text input
initial_input = st.session_state.pop("last_quick", "") 
user_query = st.text_input("Enter your fitness question:", value=initial_input, key="main_input")
ask_button = st.button("Ask FitBot")

# Automatically trigger execution if a quick query was entered
if initial_input and initial_input == user_query:
    ask_button = True


# --- Handle Execution ---
if ask_button and user_query.strip():
    # 1. Run pipeline
    with st.spinner(f"🤔 Thinking with Gemini, retrieving context for {user_query[:30]}..."):
        start = time.time()
        resp = answer_query_pipeline(llm_chain, vectorstore, user_query, st.session_state.profile, st.session_state.history)
        latency = time.time() - start

    # 2. Save and display immediately
    st.session_state.history.append({"user": user_query, "assistant": resp, "time": latency})
    
    # Rerun to clear the input and populate the history display cleanly
    st.rerun() 

# --- Display History ---
st.markdown("---")
st.subheader("💬 Conversation History")

# Inject the daily tip as the first message if no conversation exists
if not st.session_state.history:
    st.markdown(f"**FitBot:** Hello! I am FitBot, your AI fitness coach. **{st.session_state.initial_tip}** How can I support your goals today?")
    st.info("Set your profile above for personalized advice!")
else:
    # Display reverse chronological order
    for turn in reversed(st.session_state.history): 
        with st.chat_message("user"):
            st.markdown(turn['user'])
        with st.chat_message("assistant"):
            st.markdown(turn['assistant'])
        st.caption(f"⏱️ Response Time: {turn.get('time', 0):.2f}s")


# small footer
st.markdown("---")
st.caption("FitBot — Capstone Project (RAG, Memory, Personalization). Always consult a licensed professional for medical issues.")
