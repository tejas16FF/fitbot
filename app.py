# app.py — FitBot (Goal-based FAQ + Dynamic Tips)
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
# DATA: KNOWLEDGE + TIPS + FAQ
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

# Goal-based FAQ recommendations
GOAL_BASED_FAQS = {
    "Weight loss": [
        ("🔥 Fat-Burning Cardio", "Suggest a 30-minute fat-burning cardio routine."),
        ("🍽️ Calorie Deficit", "How do I maintain a healthy calorie deficit?"),
        ("🥗 Low-Cal Diet", "Give me a sample low-calorie meal plan."),
        ("💧 Water Intake", "How much water should I drink daily for fat loss?")
    ],
    "Muscle gain": [
        ("🏋️ Strength Split", "Give me a 4-day muscle-building workout plan."),
        ("🍗 Protein Diet", "What should I eat to gain lean muscle?"),
        ("🥤 Supplements", "Should I take protein shakes or creatine for muscle gain?"),
        ("🛌 Recovery", "How many rest days do I need for muscle recovery?")
    ],
    "Endurance": [
        ("🏃 Endurance Plan", "Give me a weekly running and HIIT plan."),
        ("🥦 Energy Diet", "What foods improve stamina and endurance?"),
        ("💨 Breathing", "How can I improve breathing during cardio workouts?"),
        ("🚴 Interval Training", "What are good cycling routines for stamina?")
    ],
    "General fitness": [
        ("💪 Balanced Routine", "Suggest a balanced weekly workout plan."),
        ("🥗 Healthy Eating", "What should a general fitness diet include?"),
        ("🧘 Mind & Body", "How can I include yoga for better overall health?"),
        ("⚖️ Lifestyle", "Give me daily tips to stay fit and active.")
    ]
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
        st.error(f"LLM error: {e}")
        return "Sorry — I couldn't generate an answer right now. Please try again."

# -----------------------------
# PAGE: PROFILE SETUP
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
            "name": name, "age": age, "weight": weight,
            "gender": gender, "goal": goal, "level": level,
            "diet": diet, "workout_time": workout_time,
        })
        st.session_state.profile_submitted = True
        st.session_state.faq_display = random.sample(GOAL_BASED_FAQS[goal], len(GOAL_BASED_FAQS[goal]))
        st.success("Profile saved — launching FitBot...")
        time.sleep(0.8)
        st.rerun()

# -----------------------------
# PAGE: MAIN CHAT
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # Sidebar: Profile
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # Sidebar: History
    st.sidebar.header("📜 Chat History")
    if not st.session_state.history:
        st.sidebar.info("No chats yet. Start below 👇")
    else:
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.sidebar.expander(f"Q: {turn['user'][:40]}..."):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("### ⚡ Ask about workouts, diet, or recovery")

    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)
    if chain is None:
        st.error("LLM not initialized — check GOOGLE_API_KEY")
        return

    def handle_query(q_text: str):
        placeholder = st.empty()
        tip_html = f"""
        <div style='text-align:center; color:#009688; font-size:18px; font-weight:600; transition:opacity 1s ease-in-out;' id='tip_box'>
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
        st.success(ans)
        st.session_state.faq_display = random.sample(GOAL_BASED_FAQS[st.session_state.profile["goal"]], len(GOAL_BASED_FAQS[st.session_state.profile["goal"]]))

    # Dynamic FAQ (changes after each interaction)
    st.markdown("#### 💡 Recommended Quick Questions")
    faq_items = st.session_state.faq_display[:4]
    cols = st.columns(len(faq_items))
    for i, (label, q) in enumerate(faq_items):
        if cols[i].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
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
st.caption("FitBot — Personalized AI Fitness Coach | Capstone Project")
