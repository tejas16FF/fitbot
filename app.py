# app.py — Final Stable Version (FAQ Fix + Loading Animation + Knowledge Base)
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
# DATA + DYNAMIC CONTENT
# -----------------------------
FALLBACK_KB = """
Fitness tips: Regular exercise improves cardiovascular health and builds strength.
Eat balanced meals and stay hydrated for recovery.
"""

DAILY_TIPS = [
    "💪 Stay consistent — small progress every day adds up!",
    "🔥 Hydration is key — drink enough water to fuel your muscles.",
    "🧘 Focus on your form, not the weight.",
    "🏃 20 minutes of daily movement is better than 0 minutes of perfection.",
    "🥗 Nutrition is 70% of fitness. Eat smart, not less.",
    "💤 Sleep well — muscles grow while you rest.",
    "🚶 Walk after meals to aid digestion.",
    "🧠 Don’t chase motivation. Build discipline instead.",
    "⚡ Track your progress weekly, not daily.",
    "❤️ Consistency beats intensity every single time.",
]

FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner workout plan.",
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
# SESSION STATE INIT
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

# -----------------------------
# HELPERS
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else FALLBACK_KB

@st.cache_resource(show_spinner=False, ttl=600)
def build_vectorstore(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    if not api_key:
        return None, None
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.25)
    template = """
You are FitBot, a professional AI fitness coach.
Use the user's profile data to give personalized answers.
Be friendly, supportive, and clear.
Never mention documents or internal data.

User Profile: {profile}
Conversation History: {chat_history}
Context: {context}
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
        return "⚠️ Sorry, something went wrong while generating the answer."

# -----------------------------
# PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your fitness journey 👇")

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
        st.success("✅ Profile saved successfully!")
        time.sleep(1)
        st.rerun()

# -----------------------------
# MAIN CHAT PAGE
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Coach")

    # Sidebar — Profile Info
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # Sidebar — Chat History
    st.sidebar.header("📜 Chat History")
    if not st.session_state.history:
        st.sidebar.info("No chats yet. Start asking below 👇")
    else:
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.sidebar.expander(f"🧩 {turn['user'][:40]}..."):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
                st.caption(f"⏱️ {turn['time']:.2f}s")
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Load knowledge base and chain
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    # -----------------------------
    # ✅ FIXED FAQ BUTTONS SECTION
    # -----------------------------
    st.markdown("### ⚡ Quick Fitness Queries")
    display_faqs = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(4)

    def faq_click_handler(q_text: str):
        st.session_state["faq_query"] = q_text
        st.session_state["faq_loading"] = True

    for i, (label, q) in enumerate(display_faqs):
        key = f"faq_btn_{i}_{random.randint(1000, 9999)}"
        cols[i].button(label, key=key, on_click=faq_click_handler, args=(q,))

    if st.session_state.get("faq_loading", False):
        query = st.session_state.pop("faq_query", None)
        if query:
            with st.spinner("🤔 Thinking with Gemini..."):
                placeholder = st.empty()
                tip_html = f"""
                <style>
                #load_box {{
                    text-align:center;
                    padding:12px;
                    font-size:17px;
                    font-weight:600;
                    color:#009B77;
                    transition: opacity 0.6s ease-in-out;
                }}
                </style>
                <div id="load_box">💭 {random.choice(DAILY_TIPS)}</div>
                <script>
                const tips = {DAILY_TIPS};
                let idx = 0;
                const box = document.getElementById('load_box');
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
                st.markdown(tip_html, unsafe_allow_html=True)

                start = time.time()
                answer = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
                latency = time.time() - start
                st.session_state.history.append({"user": query, "assistant": answer, "time": latency})
                st.session_state["faq_loading"] = False
                st.success(answer)

    # User Query Input
    user_query = st.chat_input("Ask FitBot your question:")
    if user_query:
        with st.spinner("🤔 Thinking with Gemini..."):
            start = time.time()
            answer = generate_answer(chain, vectorstore, user_query, st.session_state.profile, st.session_state.history)
            latency = time.time() - start
            st.session_state.history.append({"user": user_query, "assistant": answer, "time": latency})
            st.success(answer)

# -----------------------------
# CONTROL FLOW
# -----------------------------
if st.session_state.profile_submitted:
    page_chat()
else:
    page_profile()

st.markdown("---")
st.caption("FitBot — Personalized AI Fitness Coach | Built with Gemini + FAISS + LangChain")
