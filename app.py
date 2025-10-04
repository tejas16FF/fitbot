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

# -----------------------------
# KNOWLEDGE BASE & DYNAMIC DATA
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

# ✅ Extended Motivational Tips
DAILY_TIPS = [
    "💡 Stay hydrated — your muscles need water to perform well!",
    "🔥 Small progress each day adds up to big results!",
    "🧘 Focus on your form, not the weight. Perfect form builds strength safely!",
    "🏋️ You don’t have to be extreme, just consistent.",
    "🥗 Nutrition fuels your body — eat smart, not less.",
    "💪 Every rep counts — stay disciplined, not motivated.",
    "🧠 Rest days recharge your progress. Don’t skip recovery!",
    "🚶 Take a walk after meals to aid digestion.",
    "📈 Track your progress weekly — results build slowly, but surely.",
    "🕐 Time and patience beat intensity and shortcuts.",
]

# ✅ Extended FAQ Queries
FAQ_QUERIES = {
    "💪 Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-workout Meal": "What’s a good meal after exercise?",
    "🔥 Motivation Tips": "Share ways to stay consistent.",
    "🧘 Yoga Routine": "Give me a 10-minute morning yoga stretch plan.",
    "💧 Hydration": "How much water should I drink per day?",
    "⏱️ Sleep": "Why is sleep important for muscle recovery?",
    "🍽️ Calorie Intake": "How do I calculate my daily calorie needs?",
    "🏃 Cardio Routine": "Give me a 20-minute fat-burning cardio plan.",
    "🍳 Protein Sources": "List best vegetarian protein sources.",
    "🥤 Supplements": "Should I use protein shakes for weight loss?",
    "😴 Recovery Tips": "What are best recovery tips after intense workout?",
    "⚖️ Fat Loss vs Muscle Gain": "How can I lose fat without losing muscle?",
    "🏋️ Strength Plan": "Give me a 4-day strength training split.",
    "🥗 Balanced Diet": "What should a balanced diet include for daily fitness?",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
}

# -----------------------------
# SESSION STATE INITIALIZATION
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

# store Tip of the Day when profile is submitted
if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = None

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
        # set Tip of the Day to show once in main chat
        st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
        st.success("✅ Profile saved! Starting FitBot...")
        time.sleep(1)
        st.rerun()

# -----------------------------
# PAGE: MAIN CHAT
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — Your AI Fitness Assistant")

    # LEFT SIDEBAR — Profile
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.rerun()

    # RIGHT SIDEBAR — History (expanders)
    st.sidebar.header("📜 Chat History")
    if not st.session_state.history:
        st.sidebar.info("No chats yet. Start asking below 👇")
    else:
        # show each question as an expander (click to view Q/A)
        for i, turn in enumerate(reversed(st.session_state.history)):
            key = f"hist_exp_{i}"
            with st.sidebar.expander(f"🧩 {turn['user'][:40]}...", expanded=False):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
                if "time" in turn:
                    st.caption(f"⏱️ {turn['time']:.2f}s")
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # CENTER — Chat
    st.markdown("### 💡 Ask me about workouts, diet, or motivation")

    kb_text = read_knowledge_base("data.txt")
    vectorstore = build_vectorstore(kb_text)
    llm, chain = create_llm_chain(GOOGLE_KEY)

    # Show Tip of the Day once (after profile submit)
    if st.session_state.tip_of_the_day:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_of_the_day}")
        st.session_state.tip_of_the_day = None

    # Randomize FAQ Buttons each time and give unique keys
    faq_items = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(len(faq_items))
    for i, (label, query) in enumerate(faq_items):
        if cols[i].button(label, key=f"faq_{i}_{label}"):
            st.session_state["last_quick"] = query
            st.rerun()

    user_query = st.chat_input("Ask FitBot your question:")
    if user_query or "last_quick" in st.session_state:
        query = user_query or st.session_state.pop("last_quick")

        # ✅ Motivational Loading Tip (uses CSS var for auto dark/light color)
        tip_html = f"""
        <div id="tip_box" style="text-align:center; padding:10px; transition:opacity 1s;">
            <div style="font-size:18px; color: var(--text-color, #000);">💭 {random.choice(DAILY_TIPS)}</div>
        </div>
        <script>
        // rotate tips every 3s with fade-in/out
        const tips = {DAILY_TIPS};
        const tipBox = document.getElementById('tip_box');
        let idx = 0;
        function changeTip(){{
            tipBox.style.opacity = 0;
            setTimeout(() => {{
                tipBox.innerHTML = '<div style="font-size:18px; color: var(--text-color, #000);">💭 ' + tips[idx] + '</div>';
                tipBox.style.opacity = 1;
                idx = (idx + 1) % tips.length;
            }}, 500);
        }}
        setInterval(changeTip, 3000);
        </script>
        """
        html(tip_html, height=90)

        with st.spinner("Generating your personalized answer..."):
            start = time.time()
            answer = generate_answer(chain, vectorstore, query, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        # clear the tip area (the html component will remain but we proceed to show answer)
        # append to history
        st.session_state.history.append({"user": query, "assistant": answer, "time": latency})
        st.success(answer)

# -----------------------------
# CONTROL FLOW
# -----------------------------
if not st.session_state.profile_submitted:
    page_profile()
else:
    page_chat()

st.markdown("---")
st.caption("FitBot — Smart AI Fitness Coach | Personalized | Motivational | Safe Advice")
