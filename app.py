# app.py — FitBot (Extended FAQ + Personalized Queries + Dynamic Loading Tips)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

import asyncio

# --- AsyncIO event loop patch for Streamlit + gRPC ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# -----------------------------
# KNOWLEDGE BASE + DATA
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

# --- DAILY TIPS (Expanded) ---
DAILY_TIPS = [
    "💧 Stay hydrated — your muscles thrive on water!",
    "🔥 Progress happens slowly. Keep showing up!",
    "🧘 Don’t forget to stretch — flexibility is key to longevity.",
    "🏋️ Every rep counts — focus on your form.",
    "🥗 Nutrition fuels progress — eat whole, unprocessed foods.",
    "💤 Rest days aren’t lazy — they’re recovery in action!",
    "🚶 Walk after meals to help digestion and fat metabolism.",
    "📅 Consistency beats intensity — aim for small wins daily.",
    "🧠 Believe in progress, not perfection.",
    "🍎 Plan your meals — it keeps you on track.",
    "🏆 You’re stronger than you think!",
    "📈 Track your workouts — what gets measured improves.",
    "🥤 Skip sugary drinks — hydrate with water instead.",
    "🍳 Start your day with protein for steady energy.",
    "🚴 Add some cardio for heart and endurance health.",
]

# -----------------------------
# EXTENDED GOAL-BASED FAQ QUERIES (50+)
# -----------------------------
GOAL_FAQS = {
    "Weight loss": [
        ("🏃 Fat-Burning Cardio", "Give me a 20-minute fat-burning cardio plan."),
        ("🍎 Weight Loss Diet", "What should I eat to lose weight effectively?"),
        ("💧 Hydration", "How much water should I drink daily for fat loss?"),
        ("🔥 Motivation", "How to stay consistent during a weight loss journey?"),
        ("🍲 Meal Timing", "What’s the best time to eat for fat loss?"),
        ("🍵 Green Tea", "Does green tea actually help with weight loss?"),
        ("🚶 Walking Routine", "Is walking enough for weight loss?"),
        ("🍽️ Intermittent Fasting", "What is intermittent fasting and does it work?"),
        ("🥦 Low-Calorie Foods", "List healthy low-calorie snacks."),
        ("⚖️ Weight Plateau", "Why am I not losing weight even with exercise?"),
    ],
    "Muscle gain": [
        ("💪 Strength Plan", "Give me a 4-day strength training split."),
        ("🍳 Protein Sources", "List best vegetarian protein sources for muscle gain."),
        ("🏋️ Progressive Overload", "What is progressive overload and how to apply it?"),
        ("🥩 Protein Intake", "How much protein do I need per day for muscle growth?"),
        ("🥤 Supplements", "Should I take creatine or whey protein?"),
        ("🧘 Recovery", "What are the best recovery techniques for muscle gain?"),
        ("⚡ Pre-workout Meals", "What should I eat before a workout for energy?"),
        ("🍚 Bulking Tips", "How to bulk without gaining fat?"),
        ("🏋️ Compound Exercises", "List compound exercises for full-body strength."),
        ("🛌 Rest Days", "How many rest days per week for muscle growth?"),
    ],
    "Endurance": [
        ("🏃 Running Plan", "Give me a 5K training plan for beginners."),
        ("🚴 Cycling Routine", "Create a 30-minute cycling routine for stamina."),
        ("🧘 Yoga", "Which yoga poses improve endurance and breathing?"),
        ("🥦 Nutrition", "What kind of diet supports endurance training?"),
        ("💨 Breathing Techniques", "How to improve breathing control while running?"),
        ("🎯 Weekly Routine", "Design a weekly endurance workout schedule."),
        ("🥤 Energy Drinks", "Do energy drinks actually help in endurance?"),
        ("⚡ HIIT", "Can HIIT help improve endurance?"),
        ("💤 Rest", "How important is rest for endurance athletes?"),
    ],
    "General fitness": [
        ("🏋️ Full-Body Plan", "Give me a 3-day full-body beginner workout plan."),
        ("🍽️ Balanced Diet", "What should a balanced diet include daily?"),
        ("🧘 Recovery Routine", "Suggest a simple recovery routine."),
        ("🕒 Daily Routine", "How to maintain daily fitness with a busy schedule?"),
        ("💤 Sleep Importance", "Why is sleep crucial for fitness?"),
        ("🍎 Healthy Habits", "List simple daily habits for long-term fitness."),
        ("🎯 Goal Setting", "How to set realistic fitness goals?"),
        ("🥗 Breakfast Ideas", "Give me healthy breakfast ideas for fitness."),
        ("📱 Fitness Apps", "Which apps help track workouts and diet?"),
        ("🩺 Health Metrics", "What body metrics should I track for fitness?"),
    ],
}

# -----------------------------
# HELPER: Get Random FAQs (based on goal)
# -----------------------------
def get_random_faqs(goal: str, count: int = 6):
    base = GOAL_FAQS.get(goal, [])
    all_questions = sum(GOAL_FAQS.values(), [])
    combined = base + random.sample(all_questions, min(count, len(all_questions)))
    random.shuffle(combined)
    return combined[:count]

# -----------------------------
# CACHE + LLM CONFIG
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.3)
    template = """
You are FitBot, a professional AI fitness coach.
Use the user's profile data to provide personalized answers.
Be clear, supportive, and professional.
Never mention technical terms like embeddings or database.

User Profile: {profile}
Conversation so far: {chat_history}
Context: {context}
Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["profile", "chat_history", "context", "question"])
    return llm, LLMChain(llm=llm, prompt=prompt)

def retrieve_context(vectorstore, query: str, k=3):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n".join(d.page_content for d in docs)

def format_history(history, limit=6):
    recent = history[-limit:]
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items())
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "⚠️ Something went wrong while generating the response."

# -----------------------------
# PAGE 1 — PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your fitness journey 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", "")
        age = st.number_input("Age", 10, 80, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        goal = st.selectbox("Goal", list(GOAL_FAQS.keys()))
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])
        submitted = st.form_submit_button("Start FitBot")

    if submitted:
        st.session_state.profile = {
            "name": name, "age": age, "weight": weight,
            "gender": gender, "goal": goal,
            "level": level, "diet": diet, "workout_time": workout_time,
        }
        st.session_state.profile_submitted = True
        st.session_state.faqs = get_random_faqs(goal)
        st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
        st.rerun()

# -----------------------------
# PAGE 2 — CHAT PAGE
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # Left sidebar: Profile
    with st.sidebar:
        st.header("👤 Profile Info")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        if st.button("✏️ Edit Profile"):
            st.session_state.profile_submitted = False
            st.rerun()

    # Right sidebar: History
    st.sidebar.header("📜 Chat History")
    if st.session_state.get("history"):
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.sidebar.expander(f"{turn['user'][:40]}..."):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
    else:
        st.sidebar.info("No chats yet. Start chatting!")

    kb_text = FALLBACK_KB
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")

    st.markdown("#### ⚡ Quick Fitness Queries")
    faqs = get_random_faqs(st.session_state.profile["goal"])
    cols = st.columns(len(faqs))
    for i, (label, query) in enumerate(faqs):
        if cols[i].button(label):
            user_query = query
            break
    else:
        user_query = st.chat_input("Ask FitBot your question:")

    if user_query:
        with st.spinner("💭 Thinking of the best answer for you..."):
            start = time.time()
            answer = generate_answer(chain, vectorstore, user_query, st.session_state.profile, st.session_state.get("history", []))
            latency = time.time() - start
        st.session_state.history.append({"user": user_query, "assistant": answer, "time": latency})
        st.success(answer)

# -----------------------------
# MAIN CONTROL FLOW
# -----------------------------
if st.session_state.get("profile_submitted"):
    page_chat()
else:
    page_profile()

st.markdown("---")
st.caption("FitBot — Personalized AI Fitness Coach | Gemini + LangChain + FAISS | Capstone Project")
