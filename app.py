# app.py — Final Stable FitBot (Gemini + FAISS + HuggingFace)
import os
import time
import random
import asyncio
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ===============================
# 🔧 CONFIGURATION & PATCH
# ===============================
# AsyncIO loop fix for Streamlit + gRPC
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# ===============================
# 📘 KNOWLEDGE BASE
# ===============================
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

# ===============================
# 💡 DAILY TIPS
# ===============================
DAILY_TIPS = [
    "Stay hydrated — water fuels muscle recovery.",
    "Small progress each day adds up to big results!",
    "Discipline beats motivation every time.",
    "Focus on consistency, not perfection.",
    "Stretch before and after workouts to prevent injuries.",
    "Eat a protein-rich snack after training!",
    "You don’t need to be extreme, just consistent.",
    "Proper sleep = better muscle repair.",
    "Fuel your body, don’t starve it.",
    "Track your progress weekly — slow progress is still progress!",
    "A 1-hour workout is just 4% of your day — make it count!",
    "Train smart, eat clean, and stay humble.",
    "You are one workout away from a better mood.",
    "Hydration boosts energy and performance.",
    "Your body achieves what your mind believes.",
]

# ===============================
# ⚡ FAQ QUERIES
# ===============================
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
    "🏋️ Muscle Gain": "Give me a 4-day muscle gain workout plan.",
    "🥤 Supplements": "Should I use protein shakes or natural food for protein?",
    "🍳 Breakfast Ideas": "Suggest healthy breakfast ideas for fitness enthusiasts.",
    "🕓 Workout Timing": "Is it better to work out in the morning or evening?",
    "❤️ Heart Health": "How does exercise improve heart health?",
}

# ===============================
# 🧠 SESSION STATE INIT
# ===============================
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
    st.session_state.tip_of_the_day = None
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1, 10**9)

# ===============================
# 🧩 HELPER FUNCTIONS
# ===============================
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
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["profile", "chat_history", "context", "question"],
        template="""
You are FitBot, a professional AI fitness coach.
Use the user's profile data to give personalized, positive, and motivating advice.
Avoid mentioning documents or AI details.
User Profile: {profile}
Chat History: {chat_history}
Context: {context}
Question: {question}
Answer:
"""
    )
    return llm, LLMChain(llm=llm, prompt=prompt)

def retrieve_context(vectorstore, query: str, k=3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)

def format_history(history: List[Dict[str, Any]], limit=6) -> str:
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-limit:]])

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items())
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Error: {e}")
        return "Sorry, I’m having trouble generating an answer right now."

# ===============================
# 🧍 PROFILE PAGE
# ===============================
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
            "name": name, "age": age, "weight": weight, "gender": gender,
            "goal": goal, "level": level, "diet": diet, "workout_time": workout_time,
        })
        st.session_state.profile_submitted = True
        st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
        st.session_state.faq_display = random.sample(list(FAQ_QUERIES.items()), 6)
        st.success("Profile saved! Loading FitBot...")
        time.sleep(1)
        st.rerun()

# ===============================
# 💬 MAIN CHAT PAGE
# ===============================
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💪 FitBot — Your AI Fitness Coach")

    # Left Sidebar: Profile
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # Right Sidebar: History
    st.sidebar.header("📜 Chat History")
    if not st.session_state.history:
        st.sidebar.info("No chats yet.")
    else:
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.sidebar.expander(f"💬 {turn['user'][:40]}..."):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Build LLM + Vectorstore
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    # Daily Tip (shows only once after login)
    if st.session_state.tip_of_the_day:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")
        st.session_state.tip_of_the_day = None

    # Quick FAQs (randomized)
    st.markdown("#### ⚡ Quick Fitness Queries")
    display_faqs = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(4)

    def handle_query(query):
        placeholder = st.empty()
        tip_html = f"""
        <style>
        #tip_box {{
            text-align:center;
            padding:10px;
            border-radius:8px;
            transition: opacity 0.6s ease-in-out;
            font-weight:600;
            font-size:16px;
            color:#03B0A8;
        }}
        </style>
        <div id="tip_box">💭 {random.choice(DAILY_TIPS)}</div>
        <script>
        const tips = {DAILY_TIPS};
        let i = 0;
        const box = document.getElementById('tip_box');
        function changeTip(){{
            box.style.opacity=0;
            setTimeout(()=>{{ box.innerText='💭 '+tips[i]; box.style.opacity=1; i=(i+1)%tips.length; }},400);
        }}
        setInterval(changeTip,3000);
        </script>
        """
        placeholder.markdown(tip_html, unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            start = time.time()
            ans = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
            latency = time.time() - start
        placeholder.empty()
        st.session_state.history.append({"user": query, "assistant": ans, "time": latency})
        st.success(ans)

    for i, (label, q) in enumerate(display_faqs):
        key = f"faq_{st.session_state.session_id}_{i}_{random.randint(1,10**6)}"
        if cols[i].button(label, key=key):
            handle_query(q)

    user_query = st.chat_input("Ask FitBot your question:")
    if user_query:
        handle_query(user_query)

# ===============================
# 🚀 MAIN CONTROL
# ===============================
if st.session_state.profile_submitted:
    page_chat()
else:
    page_profile()

st.markdown("---")
st.caption("FitBot — Personalized Fitness Coach | Powered by Gemini & LangChain")
