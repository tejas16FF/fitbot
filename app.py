# app.py — FitBot (Final Enhanced Version)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------- CONFIGURATION -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_FOLDER_PATH = "."

# ----------------------------- KNOWLEDGE BASE -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Push-ups, Squats, Plank.
Day 2: Running, Lunges, Jump rope.
Day 3: Deadlifts, Rows, Stretching.
END FITNESS KB
"""

# ----------------------------- DAILY TIPS & FAQS -----------------------------
DAILY_TIPS = [
    "Stay hydrated — water boosts your metabolism!",
    "Focus on consistency, not perfection.",
    "Don’t skip warm-ups — prevent injuries.",
    "Sleep 7–9 hours daily for muscle recovery.",
    "Discipline beats motivation every time!",
    "Small progress every day leads to big results.",
    "Eat clean 80% of the time, enjoy 20% guilt-free.",
    "Remember, fitness is a lifestyle, not a phase.",
]

FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Protein Alternatives": "I don’t eat eggs. Suggest vegetarian protein sources.",
    "🔥 Weight Loss Tips": "How can I burn fat effectively and safely?",
    "🧘 Recovery": "What are some post-workout recovery techniques?",
    "⚡ Motivation": "How to stay motivated for daily workouts?",
    "💤 Sleep & Fitness": "Why is sleep important for fitness?",
    "🏃 Cardio vs Strength": "Which is better for weight loss — cardio or strength training?",
}

# ----------------------------- STREAMLIT STATE INIT -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "age": 25, "weight": 70, "goal": "Weight loss", "level": "Beginner", "gender": "Prefer not to say"}
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
if "initial_tip" not in st.session_state:
    st.session_state.initial_tip = random.choice(DAILY_TIPS)
if "tip_shown" not in st.session_state:
    st.session_state.tip_shown = False

# ----------------------------- HELPERS -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return FALLBACK_KB

@st.cache_resource(show_spinner=False)
def build_vectorstore_from_text(text: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.2)
    template = """
You are FitBot, a professional and friendly AI fitness coach. 
Always give fitness, diet, and motivation-related advice with a warm and supportive tone.
Never mention your internal logic or documents.
User profile: {profile}
Conversation so far: {chat_history}
Relevant info: {context}
Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["profile", "chat_history", "context", "question"])
    return LLMChain(llm=llm, prompt=prompt)

def format_history(history: List[Dict[str, Union[str, float]]], max_turns=6) -> str:
    recent = history[-max_turns:]
    return "\n".join([f"User: {h['user']}\nBot: {h['assistant']}" for h in recent])

def retrieve_context(vectorstore, query, k=3):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

def answer_query(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_history = format_history(history)
    profile_str = f"Name: {profile['name']}, Age: {profile['age']}, Goal: {profile['goal']}, Level: {profile['level']}, Gender: {profile['gender']}"
    return chain.predict(profile=profile_str, chat_history=chat_history, context=context, question=query)

# ----------------------------- PAGES -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot")
    st.write("Please enter your details to get personalized fitness guidance 💪")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        age = st.number_input("Age", 10, 80, value=int(st.session_state.profile["age"]))
        weight = st.number_input("Weight (kg)", 30, 150, value=int(st.session_state.profile["weight"]))
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        goal = st.selectbox("Primary Goal", ["Muscle gain", "Weight loss", "Endurance", "General health"])
        level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
        submitted = st.form_submit_button("Start FitBot 🚀")

    if submitted:
        st.session_state.profile.update({"name": name, "age": age, "weight": weight, "goal": goal, "level": level, "gender": gender})
        st.session_state.profile_submitted = True
        st.session_state.tip_shown = False
        st.rerun()

def page_chat():
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore_from_text(kb_text)
    chain = create_llm_chain(GOOGLE_KEY)

    # Left sidebar → Profile
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.write(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("Edit Profile"):
            st.session_state.profile_submitted = False
            st.rerun()

    # Right sidebar → History (sectioned)
    with st.sidebar.container():
        st.header("📚 History")
        for i, turn in enumerate(st.session_state.history):
            if st.button(f"Q{i+1}: {turn['user'][:40]}...", key=f"hist_{i}"):
                st.info(f"**Q:** {turn['user']}\n\n**A:** {turn['assistant']}")

    st.title("💬 Chat with FitBot")

    # Show "Tip of the Day" popup once
    if not st.session_state.tip_shown:
        st.info(f"💡 Tip of the Day: {random.choice(DAILY_TIPS)}")
        st.session_state.tip_shown = True

    # Randomized FAQs
    st.markdown("### ⚡ Quick Fitness Queries")
    faq_items = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(4)
    for i, (label, query) in enumerate(faq_items):
        if cols[i].button(label, key=f"faq_{i}"):
            st.session_state["last_query"] = query
            st.rerun()

    user_query = st.chat_input("Ask me your question:")
    if user_query or "last_query" in st.session_state:
        query = user_query or st.session_state.pop("last_query")

        # Motivational animation during processing
        placeholder = st.empty()
        start = time.time()
        for _ in range(5):
            tip = random.choice(DAILY_TIPS)
            placeholder.markdown(f"<div style='text-align:center;color:var(--text-color);font-size:1.2rem;opacity:0.8;'>💭 {tip}</div>", unsafe_allow_html=True)
            time.sleep(1.5)
        answer = answer_query(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
        latency = time.time() - start
        placeholder.empty()

        st.session_state.history.append({"user": query, "assistant": answer, "time": latency})
        st.success(answer)

# ----------------------------- CONTROL FLOW -----------------------------
if st.session_state.profile_submitted:
    page_chat()
else:
    page_profile()

st.markdown("---")
st.caption("FitBot — Personalized Fitness Coach | Capstone Project")
