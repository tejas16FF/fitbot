# app.py — FitBot (RAG + Gamification + Smart FAQ Fix)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from gamification import (
    initialize_gamification,
    update_daily_login,
    render_progress_sidebar,
    reward_for_chat,
    save_all_state,
)

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

# Extended motivational tips
DAILY_TIPS = [
    "Stay hydrated — water boosts your metabolism!",
    "Focus on consistency, not perfection.",
    "Discipline beats motivation every time!",
    "Small progress each day leads to big results!",
    "Stretching improves recovery and flexibility.",
    "Eat clean 80% of the time, enjoy 20% guilt-free.",
    "Rest days are part of the process — recover and grow!",
    "Every rep counts, even when you’re tired.",
    "Your body can do it — it’s your mind you must convince.",
    "Be stronger than your excuses.",
    "Health is not about being better than someone else, it’s about being better than you used to be.",
]

# Smart FAQ bank
FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Protein Alternatives": "I don’t eat eggs. Suggest vegetarian protein sources.",
    "🔥 Weight Loss Tips": "How can I burn fat effectively and safely?",
    "🧘 Recovery": "What are some post-workout recovery techniques?",
    "⚡ Motivation": "How to stay motivated for daily workouts?",
    "💤 Sleep & Fitness": "Why is sleep important for fitness?",
    "🏃 Cardio vs Strength": "Which is better for weight loss — cardio or strength training?",
    "🍽️ Calorie Intake": "How do I calculate daily calorie needs?",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
}

# -----------------------------
# STATE INITIALIZATION
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
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1000, 9999)

# Initialize gamification system
initialize_gamification()
update_daily_login()

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
You are FitBot, a professional and friendly AI fitness coach.
Use the user's profile data to give personalized responses.
Be motivating, friendly, and clear.
Never mention AI internals or documents.

User profile: {profile}
Conversation so far: {chat_history}
Relevant info: {context}
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
    return "\n\n---\n\n".join(d.page_content for d in docs)

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return "⚠️ Sorry, I’m having trouble answering right now. Please try again later."

# -----------------------------
# PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let’s personalize your experience 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state.profile["weight"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])
        submitted = st.form_submit_button("Start FitBot")

    if submitted:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight,
            "gender": gender, "goal": goal, "level": level,
            "diet": diet, "workout_time": workout_time,
        })
        st.session_state.profile_submitted = True
        st.success("✅ Profile saved! Launching FitBot...")
        time.sleep(0.8)
        st.rerun()

# -----------------------------
# MAIN CHAT PAGE
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    render_progress_sidebar()  # Gamification Progress

    st.markdown("### 💡 Ask me about workouts, diet, or motivation")

    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    if not chain:
        st.error("❌ LLM not initialized. Check your Google API key.")
        return

    # --- Dynamic FAQ Selection based on user goal ---
    goal = st.session_state.profile.get("goal", "").lower()
    filtered_faqs = []
    for label, query in FAQ_QUERIES.items():
        if "muscle" in goal and ("gain" in query or "protein" in query):
            filtered_faqs.append((label, query))
        elif "weight loss" in goal and ("fat" in query or "cardio" in query):
            filtered_faqs.append((label, query))
    if len(filtered_faqs) < 4:
        filtered_faqs.extend(random.sample(list(FAQ_QUERIES.items()), 4 - len(filtered_faqs)))
    display_faqs = random.sample(filtered_faqs, min(4, len(filtered_faqs)))

    # --- Quick Fitness Queries ---
    st.markdown("#### ⚡ Quick Fitness Queries")
    cols = st.columns(len(display_faqs))

    def handle_query(q_text: str):
        placeholder = st.empty()
        tip_html = f"""
        <style>
        #tip_box {{
            text-align:center;
            font-weight:600;
            font-size:16px;
            transition: opacity 0.8s ease-in-out;
            color:#00A67E;
        }}
        </style>
        <div id="tip_box">💭 {random.choice(DAILY_TIPS)}</div>
        <script>
        const tips = {DAILY_TIPS};
        let idx = 0;
        const box = document.getElementById('tip_box');
        function changeTip(){{
            box.style.opacity = 0;
            setTimeout(() => {{
                box.innerText = '💭 ' + tips[idx];
                box.style.opacity = 1;
                idx = (idx + 1) % tips.length;
            }}, 400);
        }}
        setInterval(changeTip, 3000);
        </script>
        """
        placeholder.markdown(tip_html, unsafe_allow_html=True)
        with st.spinner("💪 Crafting your personalized answer..."):
            start = time.time()
            ans = generate_answer(chain, vectorstore, q_text, st.session_state.profile, st.session_state.history)
            latency = time.time() - start
        placeholder.empty()
        st.session_state.history.append({"user": q_text, "assistant": ans, "time": latency})
        reward_for_chat()
        st.success(ans)

    for i, (label, q) in enumerate(display_faqs):
        btn_key = f"faq_{st.session_state.session_id}_{i}_{random.randint(0,9999)}"
        if cols[i].button(label, key=btn_key):
            handle_query(q)

    user_query = st.chat_input("Ask FitBot your question:")
    if user_query:
        handle_query(user_query)

# -----------------------------
# CONTROL FLOW
# -----------------------------
if st.session_state.profile_submitted:
    page_chat()
else:
    page_profile()

st.markdown("---")
st.caption("FitBot — Personalized Fitness Coach | Capstone Project")
