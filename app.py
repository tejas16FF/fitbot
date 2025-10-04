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
from streamlit.components.v1 import html

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(".env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# Fallback data
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

DAILY_TIPS = [
    "💡 Stay hydrated — your muscles need water to perform well!",
    "🔥 Small progress each day adds up to big results!",
    "🧘 Breathe deep and focus — fitness is a journey, not a race.",
    "💪 Consistency is the real secret — keep going!",
    "🥗 Fuel your body — nutrition is half the battle.",
    "🏋️ Remember, rest days are part of the plan!",
]

FAQ_QUERIES = {
    "💪 Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-workout Meal": "What’s a good meal after exercise?",
    "🔥 Motivation Tips": "Share ways to stay consistent.",
    "🧘 Yoga Routine": "Give me a 10-minute morning yoga stretch plan.",
    "💧 Hydration": "How much water should I drink per day?",
    "⏱️ Sleep": "Why is sleep important for muscle recovery?",
}

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": "",
        "weight": "",
        "goal": "General fitness",
        "level": "Beginner",
        "gender": "Prefer not to say",
        "diet": "No preference",
        "workout_time": "Morning"
    }

if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

# -----------------------------
# HELPERS
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
You are FitBot, a professional AI fitness coach.
Use the user's profile data to give *personalized* responses.
Be motivating, friendly, and clear.
Never mention AI or internal details.

User Profile: {profile}
Conversation so far: {chat_history}
Context: {context}
User Question: {question}

Provide detailed, helpful, and encouraging answers.
"""
    prompt = PromptTemplate(
        input_variables=["profile", "chat_history", "context", "question"],
        template=template,
    )
    return llm, LLMChain(llm=llm, prompt=prompt)

def retrieve_context(vectorstore, query: str, k=3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)

def format_history(history: List[Dict[str, Any]], limit=6) -> str:
    recent = history[-limit:]
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return "⚠️ Sorry, I'm having trouble answering right now. Please try again later."

# -----------------------------
# PAGE: PROFILE SETUP
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your experience 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        age = st.number_input("Age", min_value=10, max_value=80, value=25)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
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
        st.success("✅ Profile saved! Starting FitBot...")
        time.sleep(1)
        st.rerun()

# -----------------------------
# PAGE: MAIN CHAT
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # --- LEFT SIDEBAR: PROFILE ---
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # --- RIGHT SIDEBAR: HISTORY ---
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

    # --- CENTER: CHAT AREA ---
    st.markdown("### 💡 Ask me about workouts, diet, or motivation")

    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    # --- Randomize FAQs each reload ---
    faq_items = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(len(faq_items))
    for i, (label, query) in enumerate(faq_items):
        if cols[i].button(label):
            st.session_state["last_quick"] = query
            st.rerun()

    user_query = st.chat_input("Ask FitBot your question:")
    if user_query or "last_quick" in st.session_state:
        query = user_query or st.session_state.pop("last_quick")

        # --- Animated motivational loader ---
        tip_script = """
        <script>
        const tips = %s;
        let idx = 0;
        const tipBox = document.getElementById("tip");
        function changeTip() {
            tipBox.style.opacity = 0;
            setTimeout(() => {
                tipBox.innerText = tips[idx];
                tipBox.style.opacity = 1;
                idx = (idx + 1) %% tips.length;
            }, 500);
        }
        setInterval(changeTip, 3000);
        </script>
        """ % DAILY_TIPS

        html(f"""
        <div style='text-align:center; font-size:20px; padding:10px; transition:opacity 1s;' id='tip'>
            💭 Getting your personalized advice...
        </div>
        {tip_script}
        """, height=100)

        with st.spinner(""):
            start = time.time()
            answer = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        st.session_state.history.append({"user": query, "assistant": answer, "time": latency})
        st.success(answer)

# -----------------------------
# PAGE CONTROL
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Smart AI Fitness Coach | Personalized | Motivational | Safe Advice")
