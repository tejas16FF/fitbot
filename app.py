# app.py — FitBot (Single Sidebar, fixed FAQ buttons, gamification, inline history)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain / embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Gamification helper (make sure gamification.py is present)
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    render_progress_sidebar,
)

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

FALLBACK_KB = "Knowledge base not found. Add data.txt to project root."

# Motivational tips and FAQ pool
DAILY_TIPS = [
    "🏋️ Stay consistent — results come with patience.",
    "💧 Drink enough water daily to stay energized.",
    "🧠 Train your mind as much as your body.",
    "🥗 Fuel your body, don’t starve it.",
    "🔥 Progress is progress — even small steps count!",
    "🧘 Take deep breaths; stress kills gains.",
    "💪 Your only competition is your past self.",
    "🏃 Move more today than you did yesterday.",
    "😴 Recovery is part of training. Sleep well!",
    "💥 The body achieves what the mind believes.",
    "✨ Small improvements compound over time.",
    "🔁 Consistency > Intensity.",
]

FAQ_QUERIES = {
    "🏋️ Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-Workout Meal": "What should I eat after a workout for recovery?",
    "💪 Vegetarian Protein": "List vegetarian high-protein foods.",
    "🔥 Fat Loss Tips": "How do I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give me a 10-minute morning yoga routine.",
    "🚶 Warm-up": "Suggest dynamic warm-up exercises before workouts.",
    "😴 Sleep & Recovery": "How does sleep affect recovery?",
    "🏃 Cardio Plan": "Suggest a 20-minute fat-burning cardio session."
}

# -----------------------------
# SESSION STATE init
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

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
    st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)

# Used to store a query triggered by FAQ buttons or chat input
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# small flag so we can show spinner + rotating tips while processing
if "processing" not in st.session_state:
    st.session_state.processing = False

# unique session id for stable keys
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1, 10**9)

# -----------------------------
# Helpers: KB, embeddings, LLM
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
You are FitBot, a professional and friendly AI fitness coach.
Use the user's profile data to give personalized responses.
Be supportive, concise, and practical. Never mention internal mechanics.

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
    return "\n\n".join(d.page_content for d in docs)

def generate_answer(chain: LLMChain, vectorstore, query: str, profile: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    context = retrieve_context(vectorstore, query)
    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items())
    try:
        return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)
    except Exception as e:
        st.error(f"Model error: {e}")
        return "Sorry, I'm having trouble generating an answer right now."

# -----------------------------
# Gamification initialization wrapper
# -----------------------------
def init_gamification():
    initialize_gamification()
    # daily login check
    update_daily_login()

# -----------------------------
# UI: Profile page
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot")
    st.markdown("Let's personalize your experience.")

    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
        age = st.number_input("Age", min_value=10, max_value=80, value=int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=int(st.session_state.profile.get("weight", 70)))
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"],
                              index=["Male","Female","Other","Prefer not to say"].index(st.session_state.profile.get("gender", "Prefer not to say")))
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"],
                            index=["Weight loss","Muscle gain","Endurance","General fitness"].index(st.session_state.profile.get("goal", "General fitness")))
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"],
                             index=["Beginner","Intermediate","Advanced"].index(st.session_state.profile.get("level", "Beginner")))
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"],
                            index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile.get("diet","No preference")))
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"],
                                    index=["Morning","Afternoon","Evening"].index(st.session_state.profile.get("workout_time","Morning")))

        submitted = st.form_submit_button("Save & Continue")

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
        st.success("Profile saved — loading FitBot...")
        time.sleep(1)
        # initialize gamification after profile saved
        init_gamification()
        st.experimental_rerun()

# -----------------------------
# UI: Main Chat (single-sidebar layout)
# -----------------------------
def page_chat():
    # ensure gamification initialized
    init_gamification()

    # LEFT: Sidebar (profile + gamification)
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.experimental_rerun()

        # gamification sidebar component
        render_progress_sidebar()

    # CENTER: Chat, tips, FAQ, history (history displayed below chat)
    st.title("💬 FitBot — Your AI Fitness Assistant")
    st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")

    # load kb + model
    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)
    if not chain:
        st.error("❌ Gemini not initialized. Check .env GOOGLE_API_KEY.")
        return

    # --- FAQ buttons (use on_click handlers) ---
    st.markdown("### ⚡ Quick Fitness Queries")
    # choose 4 distinct items, if less available then take all
    faq_pool = list(FAQ_QUERIES.items())
    n_faq = min(4, len(faq_pool))
    faq_items = random.sample(faq_pool, n_faq)
    cols = st.columns(n_faq)

    def _on_faq_click(q_text):
        # set a pending query; actual processing occurs below the render
        st.session_state.pending_query = q_text

    for i, (label, qtext) in enumerate(faq_items):
        # stable key per session + index
        key = f"faq_{st.session_state.session_id}_{i}"
        cols[i].button(label, key=key, on_click=_on_faq_click, args=(qtext,))

    # --- Chat input (user) ---
    user_input = st.chat_input("Ask FitBot a question (press Enter):")
    if user_input:
        # store as pending query to be processed in unified handler
        st.session_state.pending_query = user_input

    # --- Process pending query (either from FAQ or chat input) ---
    if st.session_state.pending_query and not st.session_state.processing:
        # mark processing so button can't trigger duplicate
        st.session_state.processing = True
        query_to_run = st.session_state.pending_query
        st.session_state.pending_query = None  # consume the pending query

        # show spinner and rotating motivational tip
        tip_box = st.empty()
        spinner_text = random.choice(DAILY_TIPS)

        # HTML tip box with fade (works in many themes)
        tip_html = f"""
        <div style="padding:10px;border-radius:8px;text-align:center;font-weight:600;">
            💭 {spinner_text}
        </div>
        """
        tip_box.markdown(tip_html, unsafe_allow_html=True)

        with st.spinner("Thinking and retrieving relevant information..."):
            start = time.time()
            answer = generate_answer(chain, vectorstore, query_to_run, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        # reward, save history, show answer
        try:
            reward_for_chat()
        except Exception:
            # gamification might error in some envs; continue gracefully
            pass

        st.session_state.history.append({"user": query_to_run, "assistant": answer, "time": latency})

        # clear processing state & tip
        st.session_state.processing = False
        tip_box.empty()

        # show result
        st.success(answer)

    # --- History display (center, below chat) ---
    st.markdown("### 📚 Recent Conversations")
    if not st.session_state.history:
        st.info("No history yet. Ask a question or click a quick query above.")
    else:
        # show most recent first; collapsible expanders
        for i, turn in enumerate(reversed(st.session_state.history[-15:])):
            with st.expander(f"Q: {turn['user'][:80]}"):
                st.markdown(f"**Question:** {turn['user']}")
                st.markdown(f"**Answer:** {turn['assistant']}")
                st.caption(f"Response time: {turn.get('time', 0):.2f}s")

# -----------------------------
# CONTROL FLOW
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Personalized AI Fitness Coach | XP, Badges, Challenges")
