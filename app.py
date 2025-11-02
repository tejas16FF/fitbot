# app.py — FitBot (Extended Quick Questions + Dynamic Refresh + Animated Tips)
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
# DATA
# -----------------------------
DAILY_TIPS = [
    # 🔥 Motivation & Consistency
    "💪 Progress takes time — be patient and consistent.",
    "🔥 You don’t have to be extreme, just consistent.",
    "🚀 Small daily habits lead to big lifelong changes.",
    "🎯 Show up for yourself — even 10 minutes counts.",
    "🧠 Focus on progress, not perfection.",
    "🏁 The hardest part is starting — the rest will follow.",
    "🌟 Every rep is a step toward a stronger you.",
    "💥 Motivation fades, discipline stays.",
    "⚡ You won’t always be motivated — but you can always be committed.",
    "🎧 A good playlist can turn any workout into therapy.",

    # 🏋️‍♂️ Strength & Workout
    "🏋️ Strength is built one rep at a time.",
    "🤸 Warm up before lifting — it prevents injuries.",
    "💪 Train smart — quality over quantity.",
    "⚙️ Focus on form, not just the weight.",
    "🔁 Consistency beats intensity over time.",
    "📈 Track your progress — small wins add up.",
    "🚶 A 20-minute walk is better than no movement at all.",
    "🎯 Compound exercises give the best muscle growth.",
    "🏆 Rest between sets — recovery makes you stronger.",
    "🧘 Stretch after workouts for flexibility and longevity.",

    # 🥗 Nutrition
    "🥗 Fuel your goals — nutrition is 80% of fitness.",
    "🍳 Start your day with protein to stay energized.",
    "🥤 Smoothies are great, but whole foods are better.",
    "🍠 Complex carbs give lasting energy — skip the sugar crash.",
    "🥦 Eat the rainbow — more colors mean more nutrients.",
    "🥑 Healthy fats = healthy hormones.",
    "🍽️ Don’t skip meals — balance your macros instead.",
    "🍓 Replace cravings with fruits, not processed snacks.",
    "🌿 Protein helps recovery — don’t forget post-workout meals.",
    "🍎 Real food > supplements — every single time.",

    # 💧 Hydration & Recovery
    "💧 Stay hydrated — muscles need water to perform.",
    "🕐 Drink water before you feel thirsty.",
    "🌊 Electrolytes matter after sweating a lot.",
    "🛌 Sleep is your body’s natural recovery system.",
    "😴 Aim for 7–9 hours of sleep every night.",
    "🧘 Rest days are part of progress, not weakness.",
    "🔥 Stretch and foam roll to prevent soreness.",
    "📵 Reduce screen time before bed for better recovery.",
    "🧊 A cold shower can help reduce post-workout inflammation.",
    "🕯️ Deep breathing helps reduce stress and aid muscle repair.",

    # 🧠 Mindset & Lifestyle
    "🌅 Morning sunlight boosts your mood and focus.",
    "📅 Plan your workouts like appointments — don’t skip them.",
    "🎯 Your only competition is who you were yesterday.",
    "🧍‍♂️ Good posture = more confidence and better breathing.",
    "🧩 Fitness is not punishment — it’s self-respect.",
    "❤️ Love your body enough to take care of it.",
    "🎉 Rest days count — celebrate them too.",
    "🚫 Don’t let one bad meal ruin your day — balance it out.",
    "🕺 Move more, sit less — your body was made to move.",
    "🌞 A short morning stretch can change your entire day."
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
# HELPERS
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
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=template,
            input_variables=["profile", "chat_history", "context", "question"],
        ),
    )

def retrieve_context(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

def generate_answer(chain, vectorstore, query, profile, history):
    context = retrieve_context(vectorstore, query)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    chat_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-6:]])
    return chain.predict(profile=profile_str, chat_history=chat_str, context=context, question=query)

# -----------------------------
# PROFILE PAGE
# -----------------------------
def page_profile():
    st.title("🏋️ Welcome to FitBot!")
    st.markdown("Let's personalize your experience 👇")

    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.profile["name"])
        goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile["age"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state.profile["weight"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])
        submitted = st.form_submit_button("Start FitBot")

    if submitted:
        st.session_state.profile.update({
            "name": name,
            "goal": goal,
            "age": age,
            "weight": weight,
            "gender": gender,
            "level": level,
            "diet": diet,
            "workout_time": workout_time,
        })
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

    # SIDEBARS
    with st.sidebar:
        st.header("👤 Profile")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("✏️ Edit Profile"):
            st.session_state.profile_submitted = False
            st.rerun()

    st.sidebar.header("📜 Chat History")
    for turn in reversed(st.session_state.history):
        with st.sidebar.expander(f"Q: {turn['user'][:35]}..."):
            st.markdown(f"**Q:** {turn['user']}")
            st.markdown(f"**A:** {turn['assistant']}")
    if st.sidebar.button("🧹 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("### ⚡ Ask about workouts, diet, or recovery")

    kb_text = "Fitness knowledge base loaded."
    vectorstore = build_vectorstore(kb_text)
    chain = create_llm_chain()

    def handle_query(q_text):
        placeholder = st.empty()

        # 🌈 Animated motivational tip
        tip_html = f"""
        <style>
        #tip_box {{
            text-align:center;
            color:#009688;
            font-size:18px;
            font-weight:600;
            opacity:1;
            transition:opacity 1s ease-in-out;
        }}
        </style>
        <div id="tip_box">💭 {random.choice(DAILY_TIPS)}</div>
        <script>
        const tips = {DAILY_TIPS};
        let idx = 0;
        const box = document.getElementById("tip_box");
        function fadeTip() {{
            box.style.opacity = 0;
            setTimeout(() => {{
                box.innerText = "💭 " + tips[idx];
                box.style.opacity = 1;
                idx = (idx + 1) % tips.length;
            }}, 600);
        }}
        setInterval(fadeTip, 3000);
        </script>
        """
        placeholder.markdown(tip_html, unsafe_allow_html=True)

        with st.spinner("Generating your personalized response..."):
            start = time.time()
            ans = generate_answer(chain, vectorstore, q_text, st.session_state.profile, st.session_state.history)
            latency = time.time() - start

        placeholder.empty()
        st.session_state.history.append({"user": q_text, "assistant": ans, "time": latency})

        # refresh FAQs dynamically
        goal = st.session_state.profile["goal"]
        st.session_state.faq_display = random.sample(GOAL_BASED_FAQS[goal], 4)
        st.success(ans)

    # SHOW FAQ BUTTONS
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
st.caption("FitBot — Smart AI Fitness Coach")
