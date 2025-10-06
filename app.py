# app.py — FitBot (fixed tip animation + rotating FAQ suggestions)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
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

# -----------------------------
# STATIC DATA
# -----------------------------
DAILY_TIPS = [
    "💧 Stay hydrated — water boosts your metabolism!",
    "🔥 Focus on consistency, not perfection.",
    "🧘 Don’t skip warm-ups — prevent injuries.",
    "😴 Sleep 7–9 hours daily for muscle recovery.",
    "💪 Discipline beats motivation every time!",
    "🏋️ Small progress every day leads to big results.",
    "🥗 Eat clean 80% of the time, enjoy 20% guilt-free.",
    "🚀 Fitness is a lifestyle, not a phase.",
    "🤸 Stretching improves recovery and flexibility.",
    "🍎 Fuel your body with whole, nutrient-dense foods.",
]

# goal-based questions
GOAL_BASED_FAQS = {
    "Weight loss": [
        ("🔥 Fat-Burning Cardio", "Suggest a 30-minute fat-burning cardio routine."),
        ("🍽️ Calorie Deficit", "How do I maintain a healthy calorie deficit?"),
        ("🥗 Low-Cal Diet", "Give me a sample low-calorie meal plan."),
        ("💧 Water Intake", "How much water should I drink daily for fat loss?"),
        ("🧘 Rest Days", "How many rest days should I take during weight loss?"),
        ("⚡ HIIT", "Give me a quick 15-minute HIIT plan for fat burn.")
    ],
    "Muscle gain": [
        ("🏋️ Strength Split", "Give me a 4-day muscle-building workout plan."),
        ("🍗 Protein Diet", "What should I eat to gain lean muscle?"),
        ("🥤 Supplements", "Should I take protein shakes or creatine for muscle gain?"),
        ("🛌 Recovery", "How many rest days do I need for muscle recovery?"),
        ("🥩 Calorie Surplus", "How can I safely increase calorie intake for growth?"),
        ("🧠 Focus", "How can I stay consistent with muscle gain training?")
    ],
    "Endurance": [
        ("🏃 Endurance Plan", "Give me a weekly running and HIIT plan."),
        ("🥦 Energy Diet", "What foods improve stamina and endurance?"),
        ("💨 Breathing", "How can I improve breathing during cardio workouts?"),
        ("🚴 Cycling Routine", "What are good cycling routines for stamina?"),
        ("🏊 Swimming", "Is swimming effective for endurance?"),
        ("🕐 Schedule", "What’s an ideal 5-day endurance training routine?")
    ],
    "General fitness": [
        ("💪 Balanced Routine", "Suggest a balanced weekly workout plan."),
        ("🥗 Healthy Eating", "What should a general fitness diet include?"),
        ("🧘 Mind & Body", "How can I include yoga for better overall health?"),
        ("⚖️ Lifestyle", "Give me daily tips to stay fit and active."),
        ("🚶 Walking", "Is walking every day enough for fitness?"),
        ("🥑 Nutrition Basics", "What are essential nutrients for general health?")
    ]
}

# -----------------------------
# SESSION STATE
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
if "faq_display" not in st.session_state:
    st.session_state.faq_display = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def build_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def create_llm_chain():
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_KEY, temperature=0.25)
    template = """
You are FitBot, a professional and friendly AI fitness coach.
Use the user's profile data to give personalized responses.
Be motivating, friendly, and clear.

User profile: {profile}
Chat history: {chat_history}
Context: {context}
Question: {question}
Answer:
"""
    return LLMChain(llm=llm, prompt=PromptTemplate(template=template, input_variables=["profile","chat_history","context","question"]))

def retrieve_context(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    profile_str = ", ".join(f"{k}: {v}" for k,v in profile.items() if v)
    chat_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-6:]])
    return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)

# -----------------------------
# PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state.profile["weight"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        submitted = st.form_submit_button("Start FitBot")

    if submitted:
        st.session_state.profile.update({"name": name, "goal": goal, "age": age, "weight": weight, "gender": gender})
        st.session_state.profile_submitted = True
        st.session_state.faq_display = random.sample(GOAL_BASED_FAQS[goal], 4)
        st.success("✅ Profile saved! Starting FitBot...")
        time.sleep(0.8)
        st.rerun()

# -----------------------------
# CHAT PAGE
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile"):
            st.session_state.profile_submitted = False
            st.rerun()

    # right sidebar: chat history
    st.sidebar.header("📜 Chat History")
    for turn in reversed(st.session_state.history):
        with st.sidebar.expander(f"Q: {turn['user'][:35]}..."):
            st.markdown(f"**Q:** {turn['user']}")
            st.markdown(f"**A:** {turn['assistant']}")
    if st.sidebar.button("🧹 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("### ⚡ Ask about workouts, diet, or recovery")

    # prepare AI
    kb_text = "Fitness knowledge base loaded."
    vectorstore = build_vectorstore(kb_text)
    chain = create_llm_chain()

    def handle_query(q_text):
        placeholder = st.empty()
        # 🌈 animated motivational tips
        tip_html = f"""
        <div style='text-align:center; color:#00a896; font-size:18px; font-weight:600; transition:opacity 1s ease-in-out;' id='tip_box'>
            💭 {random.choice(DAILY_TIPS)}
        </div>
        <script>
        const tips = {DAILY_TIPS};
        let idx = 0;
        const box = document.getElementById('tip_box');
        function changeTip() {{
            box.style.opacity = 0;
            setTimeout(()=>{{
                box.innerText = "💭 " + tips[idx];
                box.style.opacity = 1;
                idx = (idx+1) % tips.length;
            }}, 500);
        }}
        setInterval(changeTip, 3000);
        </script>
        """
        placeholder.markdown(tip_html, unsafe_allow_html=True)

        with st.spinner("Generating your personalized response..."):
            start = time.time()
            ans = generate_answer(chain, vectorstore, q_text, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        placeholder.empty()
        st.session_state.history.append({"user": q_text, "assistant": ans, "time": latency})
        # 🎯 refresh new goal-based questions dynamically
        goal = st.session_state.profile["goal"]
        st.session_state.faq_display = random.sample(GOAL_BASED_FAQS[goal], 4)
        st.success(ans)

    # show FAQ buttons (auto-refresh on every click)
    st.markdown("#### 💡 Recommended Quick Questions")
    cols = st.columns(4)
    for i, (label, q) in enumerate(st.session_state.faq_display):
        if cols[i].button(label, key=f"faq_{i}"):
            handle_query(q)

    user_query = st.chat_input("Ask FitBot your question:")
    if user_query:
        handle_query(user_query)

# -----------------------------
# CONTROL FLOW
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Smart AI Fitness Coach | Capstone Project")
