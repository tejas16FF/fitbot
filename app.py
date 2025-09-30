# app.py — FitBot (RAG + memory + personalization + enhanced knowledge)
import os
import time
import streamlit as st
from dotenv import load_dotenv
from typing import List

# LangChain & vector tools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Load environment variables
# -----------------------------
# Edit this path if your .env is elsewhere
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
# Session-state initialization
# -----------------------------
if "history" not in st.session_state:
    # history: list of (user, assistant) tuples
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "age": "", "weight": "", "goal": ""}

if "vectorstore_built" not in st.session_state:
    st.session_state.vectorstore_built = False

# -----------------------------
# Helper: load knowledge base file
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # fallback (not ideal for final; replace with data.txt)
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
        temperature=0.2  # lower temperature for reliable answers; adjust if needed
    )

    # Prompt template includes profile and chat_history (so bot can personalize & remember)
    template = """
You are FitBot, a professional and friendly AI fitness coach. Respond in a helpful, supportive,
and safe manner. NEVER mention internal mechanics (like "knowledge base", "context", or "retrieved docs").
If very specific medical guidance is requested, give a general guideline and recommend consulting a professional.

User profile (if available): {profile}

Conversation so far:
{chat_history}

Relevant information:
{context}

User question:
{question}

Answer concisely but fully, with practical steps, optionally a short 1-2 line motivational ending.
If the user asked for a plan or step-by-step routine, present numbered steps or day-by-day schedule.
If the user asks about diet and lists restrictions (e.g., vegetarian), provide substitutions.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["profile", "chat_history", "context", "question"]
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
    return "\n".join(lines) if lines else "No previous messages."

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
    # 1) retrieve
    context = retrieve_relevant_context(vectorstore, query, k=3)
    if not context.strip():
        # if nothing found (rare if fallback KB used), set a small generic prompt
        context = "General fitness knowledge."

    # 2) format history & profile
    chat_history_str = format_history(history)
    profile_str = f"Name: {profile.get('name','')}; Age: {profile.get('age','')}; Weight: {profile.get('weight','')}; Goal: {profile.get('goal','')}"
    try:
        # 3) run chain (execute LLM)
        answer = chain.predict(profile=profile_str, chat_history=chat_history_str, context=context, question=query)
    except Exception as e:
        # friendly fallback if LLM/API fails
        answer = "Sorry — I'm having trouble generating an answer right now. Please try again in a moment."
        st.error(f"LLM error: {e}")
    return answer

# -----------------------------
# Seeds: some built-in quick replies and content (you can expand data.txt instead)
# -----------------------------
PREDEFINED_QUERIES = {
    "3-day beginner workout": "Give me a 3-day beginner full-body workout plan with reps and sets.",
    "Post-workout meal": "What is a good post-workout meal to support recovery?",
    "Improve flexibility": "How can I improve flexibility with a daily routine?",
    "Motivation tip": "Give me a short motivational coaching message for someone losing motivation.",
    "Injury: knee pain": "What are safe, general guidelines for someone with knee pain (non-medical advice)?"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")
st.title("💪 FitBot — Your AI Fitness Assistant")

st.markdown(
    "Ask about workouts, nutrition, recovery, motivation, or general fitness. "
    "Use the profile form to get personalized recommendations."
)

# Profile form (no sidebar)
with st.expander("Set / update your profile (optional)", expanded=False):
    name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
    age = st.text_input("Age", value=st.session_state.profile.get("age", ""))
    weight = st.text_input("Weight (kg)", value=st.session_state.profile.get("weight", ""))
    goal = st.selectbox("Primary goal", ["", "Muscle gain", "Weight loss", "Endurance", "Flexibility", "General health"],
                        index=0 if not st.session_state.profile.get("goal") else None)
    if st.button("Save profile"):
        st.session_state.profile.update({"name": name, "age": age, "weight": weight, "goal": goal})
        st.success("Profile updated!")

# quick buttons row
cols = st.columns([1,1,1,1,1])
btn_keys = list(PREDEFINED_QUERIES.keys())
for i, c in enumerate(cols):
    if i < len(btn_keys):
        q = btn_keys[i]
        if c.button(q):
            st.session_state["last_quick"] = PREDEFINED_QUERIES[q]
            st.experimental_rerun()

# load KB and models (build vectorstore cached)
with st.spinner("Preparing knowledge base and model (first run may take a bit)..."):
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore_from_text(kb_text)
    llm, llm_chain = create_llm_and_chain()
    if llm is None or llm_chain is None:
        st.error("API key for Gemini not found. Add GOOGLE_API_KEY to .env and reload.")
        st.stop()

# input area
user_query = st.text_input("Enter your fitness question:", value=st.session_state.get("last_quick", ""))
ask_button = st.button("Ask")

if ask_button and user_query.strip():
    # run pipeline
    with st.spinner("🤔 Thinking..."):
        start = time.time()
        resp = answer_query_pipeline(llm_chain, vectorstore, user_query, st.session_state.profile, st.session_state.history)
        latency = time.time() - start

    # save to history
    st.session_state.history.append({"user": user_query, "assistant": resp, "time": latency})
    st.session_state.last_quick = ""  # clear quick query
    st.success(resp)

# show chat history (reverse chronological)
if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation")
    for turn in reversed(st.session_state.history[-10:]):  # show last 10
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**FitBot:** {turn['assistant']}")
        st.caption(f"Response time: {turn.get('time', 0):.2f}s")
else:
    st.info("No conversation yet. Try a sample quick button or ask a question.")

# small footer
st.markdown("---")
st.caption("FitBot — prototype. For medical issues consult a licensed professional.")
