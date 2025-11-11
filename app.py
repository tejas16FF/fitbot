# ==========================================================
#                         FitBot v4.0
#        Clean, stable, patched build with Chroma + Gemini
#                Navigation + History + FAQ working
# ==========================================================

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# ✅ Updated LangChain imports (latest compatible versions)
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ----------------------------------------------------------
#   GLOBAL PRESETS
# ----------------------------------------------------------
FALLBACK_KB = """
General Fitness Guidelines:
- Regular exercise improves cardiovascular health.
- Balanced diet supports protein, carbs, healthy fats.
- Hydration: 2–3 litres daily.
- Sleep: 7–9 hours.
- Warm-ups reduce injuries.
- Progressive overload builds muscle.
- Combine cardio + strength for weight loss.
"""

DAILY_TIPS = [
    "Stay hydrated — water boosts your metabolism!",
    "Warm up before intense workouts to prevent injury.",
    "Small progress every day leads to big results.",
    "Focus on form, not just weights.",
    "Fuel your body with clean foods.",
    "Consistency > Motivation.",
    "Stretch after workouts to improve flexibility.",
    "Take rest days seriously.",
    "Track your progress — it's motivating!",
    "Eat protein with every meal.",
]

FAQ_QUERIES = {
    "🏋️ 3-Day Beginner Plan": "Give me a 3-day full-body workout plan.",
    "🔥 Fat Burn Tips": "How can I burn fat effectively?",
    "🍎 Post Workout Meal": "What should I eat after a workout?",
    "💪 Muscle Gain Diet": "What to eat for muscle gain?",
    "🧘 Recovery Tips": "Tell me recovery tips after intense workout.",
    "⚡ Motivation": "How do I stay consistent?",
    "💤 Why Sleep Matters": "Why is sleep important for fitness?",
    "🚶 Warm-up Ideas": "Give dynamic warm-up exercises.",
    "🥗 Vegetarian Protein": "List vegetarian high-protein foods.",
    "🏃 Cardio Routine": "Give me a 20-minute fat-burning cardio plan.",
}

# ----------------------------------------------------------
#   SESSION SETUP
# ----------------------------------------------------------
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": "",
        "weight": "",
        "gender": "",
        "goal": "",
        "level": "",
        "diet": "",
        "workout_time": "",
    }

if "history" not in st.session_state:
    st.session_state.history = []

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Chat"

# ----------------------------------------------------------
#   BUILD VECTORSTORE
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeds, collection_name="fitbot_kb")


# ----------------------------------------------------------
#   BUILD LLM CHAIN
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_llm():
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY in .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    template = """
You are FitBot, a professional, motivating AI fitness coach.
Use user's profile data to give personalized advice.
Never mention 'vectorstore', 'context', 'documents'.

User Profile:
{profile}

Previous Conversation:
{chat_history}

Relevant Info:
{context}

User Question: {question}

Answer politely, clearly, and professionally.
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["profile", "chat_history", "context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


# ----------------------------------------------------------
#   RETRIEVAL
# ----------------------------------------------------------
def get_context(store, query):
    docs = store.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


def get_chat_history():
    recent = st.session_state.history[-6:]
    return "\n".join([
        f"User: {x['user']}\nAssistant: {x['assistant']}"
        for x in recent
    ])


# ----------------------------------------------------------
#   Generate answer
# ----------------------------------------------------------
def generate_answer(chain, store, query):
    profile_str = ", ".join(f"{k}: {v}" for k, v in st.session_state.profile.items())
    ctx = get_context(store, query)
    hist = get_chat_history()

    return chain.run(
        profile=profile_str,
        chat_history=hist,
        context=ctx,
        question=query
    )


# ----------------------------------------------------------
#   UI COMPONENTS
# ----------------------------------------------------------
def render_top_nav():
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💬 Chat", key="top_chat"):
            st.session_state.nav_page = "Chat"
    with c2:
        if st.button("⚡ Challenges", key="top_challenges"):
            st.session_state.nav_page = "Challenges"
    with c3:
        if st.button("📜 History", key="top_history"):
            st.session_state.nav_page = "History"


# ----------------------------------------------------------
#   Profile Page
# ----------------------------------------------------------
def page_profile():
    st.title("🏋️ FitBot — Profile Setup")
    st.markdown("### Let's personalize your experience!")

    with st.form("profile_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", 10, 80)
        weight = st.number_input("Weight (kg)", 30, 200)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        goal = st.selectbox("Fitness Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])

        submitted = st.form_submit_button("Save & Continue")

        if submitted:
            st.session_state.profile = {
                "name": name,
                "age": age,
                "weight": weight,
                "gender": gender,
                "goal": goal,
                "level": level,
                "diet": diet,
                "workout_time": workout_time,
            }
            st.session_state.profile_submitted = True
            st.session_state.nav_page = "Chat"
            st.success("Profile saved successfully!")
            st.rerun()


# ----------------------------------------------------------
#   Chat Page
# ----------------------------------------------------------
def page_chat():
    st.title("💬 FitBot – Your AI Fitness Assistant")
    render_top_nav()

    store = build_store(FALLBACK_KB)
    chain = build_llm()

    # FAQ Buttons
    st.markdown("### ⚡ Quick Questions")

    faq_list = random.sample(list(FAQ_QUERIES.items()), 4)
    cols = st.columns(4)

    for idx, (label, q) in enumerate(faq_list):
        if cols[idx].button(label, key=f"faq_{idx}_{random.randint(1,9999)}"):
            with st.spinner("💭 Thinking..."):
                answer = generate_answer(chain, store, q)
                st.session_state.history.append({"user": q, "assistant": answer})
                st.success(answer)

    # Chat Input
    user_query = st.chat_input("Ask your fitness question:")
    if user_query:
        with st.spinner("💭 Generating answer..."):
            answer = generate_answer(chain, store, user_query)
            st.session_state.history.append({"user": user_query, "assistant": answer})
            st.success(answer)


# ----------------------------------------------------------
#   History Page
# ----------------------------------------------------------
def page_history():
    st.title("📜 Chat History")
    render_top_nav()

    if not st.session_state.history:
        st.info("No chat history yet.")
        return

    for turn in st.session_state.history[::-1]:
        st.markdown(f"### ❓ {turn['user']}")
        st.markdown(f"✅ {turn['assistant']}")
        st.markdown("---")


# ----------------------------------------------------------
#   Challenges Page
# ----------------------------------------------------------
def page_challenges():
    st.title("⚡ Weekly Challenges")
    render_top_nav()

    st.info("Coming Soon — Achievement system under construction.")


# ----------------------------------------------------------
#   NAV ROUTER
# ----------------------------------------------------------
def main():
    if not st.session_state.profile_submitted:
        page_profile()
        return

    page = st.session_state.nav_page

    if page == "Chat":
        page_chat()
    elif page == "Challenges":
        page_challenges()
    elif page == "History":
        page_history()


main()
