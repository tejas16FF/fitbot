import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# -------- Import LangChain modules --------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------- Gamification system --------
from gamification import (
    get_current_level,
    get_user_xp,
    add_xp,
    complete_challenge,
    get_weekly_challenges,
    get_completed_challenges,
)

# -------- Load env --------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-pro")

# --------- Data Inputs ---------
FALLBACK_KB = open("data.txt", "r", encoding="utf-8").read()

DAILY_TIPS = [
    "Stay hydrated — water boosts your metabolism!",
    "Focus on consistency, not perfection.",
    "You don’t have to be extreme, just consistent.",
    "Stretching improves recovery and flexibility.",
    "Small progress every day = big results.",
    "Strength starts in the mind.",
    "Your body achieves what your mind believes.",
    "Eat protein with every meal.",
    "Don’t quit on yourself!",
    "Recovery is where growth happens."
]

FAQ_QUERIES = {
    "🏋 Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-workout Meal": "What is the best post-workout meal?",
    "💧 Hydration Tips": "How much water should I drink daily?",
    "🔥 Fat Loss": "Give me efficient fat-loss strategies.",
    "💪 Muscle Gain": "How to build muscle effectively?",
    "🧘 Recovery": "What are the best recovery tips?",
    "😴 Sleep": "Explain why sleep is important for fitness.",
}

# -------- Session State Setup --------
if "profile" not in st.session_state:
    st.session_state.profile = {}

if "history" not in st.session_state:
    st.session_state.history = []

if "active_screen" not in st.session_state:
    st.session_state.active_screen = "profile"

if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1000, 9999999)

if "last_faq_trigger" not in st.session_state:
    st.session_state.last_faq_trigger = None


# -------- Helpers --------
def build_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def create_llm_chain():
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_KEY,
        temperature=0.35
    )

    template = """
You are FitBot, an AI fitness coach with a professional and motivational tone.
Use User Profile for personalization.

User Profile: {profile}
Chat History: {chat_history}
Context: {context}
Question: {question}

Provide actionable steps, clear advice, and never mention context, knowledge base, or internals.
"""

    prompt = PromptTemplate(
        input_variables=["profile", "chat_history", "context", "question"],
        template=template
    )
    return llm, LLMChain(llm=llm, prompt=prompt)


def retrieve_context(store, query):
    docs = store.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])


def format_history():
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in st.session_state.history[-6:]])


def themed_caption(text):
    base = st.get_option("theme.base") or "light"
    color = "#00E1C1" if base == "dark" else "#006d5b"
    return f"<p style='text-align:center;color:{color};font-weight:600;margin-top:8px'>{text}</p>"


def show_progress_image(img_path, caption):
    with st.container():
        st.image(img_path, use_column_width=True)
        st.markdown(themed_caption(caption), unsafe_allow_html=True)
        st.write("")


# ---------- Pages ----------

def page_profile():
    st.title("🏋️ Welcome to FitBot")
    st.write("Let's personalize your fitness plan.")

    with st.form("profile_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=10, max_value=80)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        goal = st.selectbox("Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-Vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])
        submitted = st.form_submit_button("Start")

    if submitted:
        st.session_state.profile = {
            "name": name,
            "age": age,
            "weight": weight,
            "gender": gender,
            "goal": goal,
            "level": level,
            "diet": diet,
            "workout_time": workout_time
        }
        st.session_state.active_screen = "chat"
        st.rerun()


def page_chat():
    st.title("💬 FitBot — Your AI Fitness Assistant")
    st.markdown("Ask me anything about workouts, diet, motivation, or recovery.")

    # knowledge base
    store = build_vectorstore(FALLBACK_KB)
    llm, chain = create_llm_chain()

    # Quick questions
    st.subheader("⚡ Quick Questions")
    cols = st.columns(3)

    for idx, (label, query) in enumerate(list(FAQ_QUERIES.items())[:6]):
        if cols[idx % 3].button(label, key=f"faq_{st.session_state.session_id}_{idx}"):
            st.session_state.last_faq_trigger = query
            st.rerun()

    if st.session_state.last_faq_trigger:
        user_query = st.session_state.last_faq_trigger
        st.session_state.last_faq_trigger = None
    else:
        user_query = st.chat_input("Type your question:")

    if user_query:
        with st.spinner("💭 Thinking..."):
            context = retrieve_context(store, user_query)
            answer = chain.predict(
                profile=str(st.session_state.profile),
                chat_history=format_history(),
                context=context,
                question=user_query
            )
            st.session_state.history.append({"user": user_query, "assistant": answer})
        st.success(answer)


def page_challenges():
    st.title("🎯 Weekly Challenges")

    challenges = get_weekly_challenges()
    completed = get_completed_challenges()

    for cid, challenge in challenges.items():
        with st.expander(f"{challenge['title']} ({challenge['xp']} XP)"):
            st.write(challenge["description"])
            if cid in completed:
                st.success("✅ Completed")
            else:
                if st.button(f"Complete {challenge['title']}", key=f"chall_{cid}"):
                    add_xp(challenge["xp"])
                    complete_challenge(cid)
                    st.success(f"🎉 Challenge Completed! +{challenge['xp']} XP")
                    time.sleep(1)
                    st.rerun()


def page_progress():
    st.title("📈 Your Progress")

    level = get_current_level()
    xp = get_user_xp()

    st.write(f"### Level: {level}")
    st.write(f"### XP: {xp}")

    show_progress_image("assets/progress1.png", "Level 1 — Beginner")
    show_progress_image("assets/progress2.png", "Level 2 — Challenger")
    show_progress_image("assets/progress3.png", "Level 3 — Warrior")


# -------- Bottom Navigation --------
def render_bottom_nav():
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("💬 Chat", key=f"nav_chat_{st.session_state.session_id}"):
            st.session_state.active_screen = "chat"
            st.rerun()
        if c2.button("🎯 Challenges", key=f"nav_chal_{st.session_state.session_id}"):
            st.session_state.active_screen = "challenges"
            st.rerun()
        if c3.button("📈 Progress", key=f"nav_prog_{st.session_state.session_id}"):
            st.session_state.active_screen = "progress"
            st.rerun()
        if c4.button("👤 Profile", key=f"nav_profile_{st.session_state.session_id}"):
            st.session_state.active_screen = "profile"
            st.rerun()


# -------- MAIN --------
def main():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

    screen = st.session_state.active_screen

    if screen == "profile":
        page_profile()
    elif screen == "chat":
        page_chat()
        render_bottom_nav()
    elif screen == "challenges":
        page_challenges()
        render_bottom_nav()
    elif screen == "progress":
        page_progress()
        render_bottom_nav()


if __name__ == "__main__":
    main()
