# app.py — FitBot (FAQ + loading-tip fixes)
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
# DYNAMIC DATA
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

DAILY_TIPS = [
    "Stay hydrated — water boosts your metabolism!",
    "Focus on consistency, not perfection.",
    "Don’t skip warm-ups — prevent injuries.",
    "Sleep 7–9 hours daily for muscle recovery.",
    "Discipline beats motivation every time!",
    "Small progress every day leads to big results.",
    "Eat clean 80% of the time, enjoy 20% guilt-free.",
    "Remember, fitness is a lifestyle, not a phase.",
    "Stretching improves recovery and flexibility.",
    "Fuel your body with whole, nutrient-dense foods.",
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
if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = None
# session id to help produce unique keys
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1, 10**9)

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
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.25)
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

def generate_answer(chain: LLMChain, vectorstore, query: str, profile: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        # return friendly fallback
        st.error(f"LLM error: {e}")
        return "Sorry — I couldn't generate an answer right now. Please try again."

# -----------------------------
# UI: PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your experience 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile.get("name", ""))
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile.get("age", 25))
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state.profile.get("weight", 70))
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"],
                              index=["Male", "Female", "Other", "Prefer not to say"].index(st.session_state.profile.get("gender","Prefer not to say")))
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"],
                            index=["Weight loss","Muscle gain","Endurance","General fitness"].index(st.session_state.profile.get("goal","General fitness")))
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"],
                             index=["Beginner","Intermediate","Advanced"].index(st.session_state.profile.get("level","Beginner")))
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"],
                            index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile.get("diet","No preference")))
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"],
                                    index=["Morning","Afternoon","Evening"].index(st.session_state.profile.get("workout_time","Morning")))
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
        st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
        # refresh faq_display for this session (so FAQ changes on login)
        st.session_state.faq_display = random.sample(list(FAQ_QUERIES.items()), min(6, len(FAQ_QUERIES)))
        st.success("Profile saved — loading FitBot...")
        time.sleep(0.8)
        st.rerun()


# -----------------------------
# UI: MAIN CHAT PAGE
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # Left sidebar: Profile (keeps profile editable)
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()


    # Right sidebar: History (sectioned)
    st.sidebar.header("📜 Chat History")
    if not st.session_state.history:
        st.sidebar.info("No chats yet. Start asking below 👇")
    else:
        for i, turn in enumerate(reversed(st.session_state.history)):
            # show as expander so user can click and view each Q/A
            with st.sidebar.expander(f"Q: {turn['user'][:40]}...", expanded=False):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
                if "time" in turn:
                    st.caption(f"⏱️ {turn['time']:.2f}s")
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


    # center area: main chat + FAQs
    st.markdown("### 💡 Ask me about workouts, diet, recovery or motivation")

    # build vectorstore & chain (cached)
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)
    if chain is None:
        st.error("LLM not initialized — make sure GOOGLE_API_KEY is set in .env")
        return

    # Tip of the Day (show once after profile submit)
    if st.session_state.tip_of_the_day:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")
        st.session_state.tip_of_the_day = None

    # Prepare FAQ display: pick 4 from session's faq_display (which was set on login)
    if "faq_display" not in st.session_state:
        st.session_state.faq_display = random.sample(list(FAQ_QUERIES.items()), min(6, len(FAQ_QUERIES)))
    display_faqs = st.session_state.faq_display[:4]

    st.markdown("#### ⚡ Quick Fitness Queries")
    # render FAQ buttons in responsive columns
    cols = st.columns(len(display_faqs))
    def handle_query(q_text: str):
        """Centralized query handler: shows loading tips, runs chain, appends history and displays answer."""
        # show rotating motivational tips in a placeholder (removed after answer)
        placeholder = st.empty()
        tip_html = """
        <style>
        #tip_box {
            text-align:center;
            padding:10px;
            border-radius:8px;
            transition: opacity 0.6s ease-in-out;
            font-weight:600;
            font-size:16px;
        }
        @media (prefers-color-scheme: dark) {
            #tip_box { background: rgba(255,255,255,0.03); color: #00BFA6; }
        }
        @media (prefers-color-scheme: light) {
            #tip_box { background: rgba(0,0,0,0.03); color: #006d5b; }
        }
        </style>
        <div id="tip_box">💭 Loading…</div>
        <script>
        const tips = %s;
        let idx = 0;
        const box = document.getElementById('tip_box');
        function changeTip(){ box.style.opacity = 0; setTimeout(()=>{ box.innerText = '💭 '+tips[idx]; box.style.opacity = 1; idx = (idx+1)%%tips.length; }, 300); }
        changeTip();
        let timer = setInterval(changeTip, 3000);
        // stop rotating after 12s
        setTimeout(()=>{ clearInterval(timer); }, 12000);
        </script>
        """ % DAILY_TIPS
        placeholder.markdown(tip_html, unsafe_allow_html=True)

        # run the model with spinner
        with st.spinner("Generating your personalized response..."):
            start = time.time()
            ans = generate_answer(chain, vectorstore, q_text, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        # clear tip placeholder and show answer
        placeholder.empty()
        st.session_state.history.append({"user": q_text, "assistant": ans, "time": latency})
        st.success(ans)

    # render FAQ buttons — handle immediately on click (no rerun)
    for i, (label, q) in enumerate(display_faqs):
        # unique key per session + index avoids collisions
        if cols[i].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            handle_query(q)

    # Chat input (user typed queries)
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
