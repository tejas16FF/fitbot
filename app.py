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
from langchain_core.messages import AIMessage, HumanMessage 

# --- Configuration ---
# Path for FAISS index persistence (Caches index to local disk for fast restarts)
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_FOLDER_PATH = "."

# Load environment variables
load_dotenv(".env")

# The developer's key is used automatically since the user is not providing one.
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") 
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
...
END FITNESS KB
"""

# -----------------------------
# Seeds: Daily Tips and FAQs
# -----------------------------
DAILY_TIPS = [
    "Tip: Drinking water before meals helps reduce calorie intake and keeps you hydrated!",
    "Tip: Focus on progressive overload to build muscle strength—lift slightly more, or do one extra rep next time!",
    "Tip: Try a dynamic warm-up before your workout to reduce injury risk and improve performance.",
    "Tip: Aim for 7-9 hours of quality sleep tonight. Your muscles repair and grow while you rest!",
]

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

if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
    
if "initial_tip" not in st.session_state:
    st.session_state.initial_tip = random.choice(DAILY_TIPS)

if "selected_turn_index" not in st.session_state:
    st.session_state.selected_turn_index = -1 # -1 means no turn selected

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
    
    # Load from disk cache if available
    if os.path.exists(FAISS_FOLDER_PATH) and any(f.startswith(FAISS_INDEX_PATH) for f in os.listdir(FAISS_FOLDER_PATH)):
        try:
            vectorstore = FAISS.load_local(FAISS_FOLDER_PATH, embeddings, FAISS_INDEX_PATH)
            return vectorstore
        except Exception:
            pass # Proceed to rebuild if load fails

    # Build index (slow, runs only once)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save to disk for persistence
    vectorstore.save_local(FAISS_FOLDER_PATH, index_name=FAISS_INDEX_PATH)
    return vectorstore

# -----------------------------
# Cached: create LLM (Gemini) and LLMChain with prompt
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_llm_and_chain(api_key: str):
    if not api_key:
        return None, None

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
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
def format_history(history: List[Dict[str, Union[str, float]]], max_turns: int = 6) -> str:
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

def answer_query_pipeline(chain: LLMChain, vectorstore, query: str, profile: Dict[str, Any], history: List[Dict[str, Union[str, float]]]):
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
# --- Page 1: Profile Setup ---
# -----------------------------
def page_profile_setup():
    st.title("💪 Welcome to FitBot! Let's get started.")
    st.markdown("Please enter your details to get personalized fitness advice. This helps FitBot tailor plans for your **goals** and **level**.")
    
    with st.form("profile_form"):
        st.subheader("Personal Details")
        name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
        
        st.subheader("Fitness Metrics")
        age = st.text_input("Age (Years)", value=str(st.session_state.profile.get("age", 25)))
        weight = st.text_input("Weight (kg)", value=str(st.session_state.profile.get("weight", 70)))
        
        gender_options = ["Male", "Female", "Other", "Prefer not to say"]
        gender = st.selectbox("Gender", gender_options, index=gender_options.index(st.session_state.profile.get("gender", "Prefer not to say")))
        
        goal_options = ["Muscle gain", "Weight loss", "Endurance", "General health"]
        goal = st.selectbox("Primary Goal", goal_options, index=goal_options.index(st.session_state.profile.get("goal", "Weight loss")))
        
        level_options = ["Beginner", "Intermediate", "Advanced"]
        level = st.selectbox("Level", level_options, index=level_options.index(st.session_state.profile.get("level", "Beginner")))
        
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
        # Use the developer's key (GOOGLE_KEY) since the user is not providing one.
        st.session_state.user_api_key = GOOGLE_KEY 
        st.session_state.profile_submitted = True
        st.rerun()
    elif submitted:
        st.error("Please fill in your name, age, and weight to continue.")

# -----------------------------
# --- Page 2: Main Chat Page ---
# -----------------------------
def page_main_chat(user_api_key):
    
    # --- Load RAG components (cached) ---
    llm_key = user_api_key 
    if not llm_key:
        st.error("Fatal Error: Developer GOOGLE_API_KEY is not set in the environment. Please check .env file.")
        return
        
    with st.spinner(f"Preparing RAG components: loading knowledge base and FAISS index..."):
        kb_text = read_knowledge_base("data.txt")
        vectorstore = build_vectorstore_from_text(kb_text)
        llm, llm_chain = create_llm_and_chain(llm_key)
        
        if llm_chain is None:
            st.error("Setup Error: Failed to initialize LLM. Check developer's API key validity.")
            return

    # --- Layout: Main columns (Active Chat: 2 parts, History: 1 part) ---
    col_chat, col_history = st.columns([2, 1])

    # --- LEFT SIDEBAR: PROFILE ---
    with st.sidebar:
        st.subheader("👤 Your Profile")
        st.markdown("---")
        st.markdown(f"**Name:** {st.session_state.profile['name']}")
        st.markdown(f"**Goal:** {st.session_state.profile['goal']}")
        st.markdown(f"**Level:** {st.session_state.profile['level']}")
        st.markdown("---")
        if st.button("Change Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.session_state.selected_turn_index = -1 # Clear selection
            st.rerun()
            
    # --- RIGHT COLUMN: INTERACTIVE HISTORY ---
    with col_history:
        st.subheader("📚 History Index")
        st.caption("Click a question to view the detailed response.")
        
        history_labels = [f"Q{i+1}: {turn['user'][:40]}..." for i, turn in enumerate(st.session_state.history)]
        
        # Add 'Ask New Question' option
        history_labels.append("Ask New Question")
        
        # Default selection: the last question if history exists, or 'Ask New Question'
        default_index = len(st.session_state.history) if st.session_state.selected_turn_index == -1 else st.session_state.selected_turn_index
        
        # Update selection tracker based on radio click
        selected_label = st.radio(
            "Past Questions",
            options=history_labels,
            index=default_index,
            key='history_radio',
            label_visibility='collapsed'
        )

        # Set the index of the selected turn
        if selected_label == "Ask New Question":
            st.session_state.selected_turn_index = -1
        else:
            st.session_state.selected_turn_index = history_labels.index(selected_label)
            
        st.markdown("---")
        if st.button("Clear All History", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected_turn_index = -1
            st.rerun()

    # --- CENTER COLUMN: ACTIVE CHAT / SELECTED ANSWER ---
    with col_chat:
        st.subheader("💬 Active Coach")
        st.markdown(f"**{st.session_state.profile['name']}**, your goal is **{st.session_state.profile['goal']}**.")
        st.markdown("---")

        if st.session_state.selected_turn_index == -1:
            # --- Display New Chat View (Center Column) ---
            
            if not st.session_state.history:
                 st.markdown(f"**FitBot:** Hello! I'm your AI fitness coach. **{st.session_state.initial_tip}** How can I support your goals today?")
            else:
                 st.markdown("**Ask a new question below, or use the quick buttons.**")
            
            # Quick Start Buttons
            button_cols = st.columns(len(FAQ_QUERIES))
            btn_keys = list(FAQ_QUERIES.keys())
            for i, c in enumerate(button_cols):
                if i < len(btn_keys):
                    q_label = btn_keys[i]
                    if c.button(q_label):
                        st.session_state["last_quick"] = FAQ_QUERIES[q_label]
                        st.session_state.selected_turn_index = -1 
                        st.rerun() 
                        
            # Main Chat Input (Triggered by Enter Key)
            initial_input = st.session_state.pop("last_quick", "") 
            user_query = st.chat_input("Ask FitBot your question (Press Enter to submit):", key="main_input")

            # Handle execution (Quick Button OR Enter Key)
            if user_query or initial_input:
                final_query = user_query if user_query else initial_input
                
                # Run pipeline
                with st.spinner(f"🤔 Thinking with Gemini, retrieving context..."):
                    start = time.time()
                    resp = answer_query_pipeline(llm_chain, vectorstore, final_query, st.session_state.profile, st.session_state.history)
                    latency = time.time() - start

                # Save history and select the new turn
                st.session_state.history.append({"user": final_query, "assistant": resp, "time": latency})
                st.session_state.selected_turn_index = len(st.session_state.history) - 1 
                st.rerun()
            
            # Show the latest active conversation only
            if st.session_state.history and st.session_state.selected_turn_index == len(st.session_state.history) - 1:
                latest_turn = st.session_state.history[-1]
                with st.chat_message("user"):
                    st.markdown(latest_turn['user'])
                with st.chat_message("assistant"):
                    st.markdown(latest_turn['assistant'])
                
        else:
            # --- Display Selected History Section (Center Column) ---
            
            selected_turn = st.session_state.history[st.session_state.selected_turn_index]
            
            st.subheader("Selected Query:")
            with st.chat_message("user"):
                st.markdown(selected_turn['user'])
            
            st.subheader("FitBot Full Response:")
            with st.chat_message("assistant"):
                st.markdown(selected_turn['assistant'])
            
            st.caption(f"⏱️ Response Time: {selected_turn.get('time', 0):.2f}s")
            st.markdown("---")
            if st.button("Ask a New Question", use_container_width=True):
                st.session_state.selected_turn_index = -1
                st.rerun()

# -----------------------------
# Global Execution
# -----------------------------
if st.session_state.profile_submitted:
    page_main_chat(st.session_state.user_api_key)
else:
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")
    page_profile_setup()
    
# Footer (Display globally)
st.markdown("---")
st.caption("FitBot — Capstone Project (RAG, Memory, Personalization). Always consult a licensed professional for medical issues.")
