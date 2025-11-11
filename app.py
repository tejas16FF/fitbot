import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# LangChain & VectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Local embedding model
from sentence_transformers import SentenceTransformer

# --------------------------
# CONFIG & ENV
# --------------------------
load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# --------------------------
# SESSION STATE
# --------------------------
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

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

if "history" not in st.session_state:
    st.session_state.history = []

# For generating unique keys
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(10000, 99999)


# --------------------------
# KNOWLEDGE BASE
# --------------------------
def read_kb():
    if os.path.exists("data.txt"):
        return open("data.txt", "r", encoding="utf-8").read()
    return "General fitness advice only."


# --------------------------
# LOCAL EMBEDDING + CHROMA
# --------------------------
@st.cache_resource(show_spinner=False)
def build_store(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    class LocalEmbedder:
        def embed_documents(self, texts):
            return model.encode(texts, show_progress_bar=False).tolist()

        def embed_query(self, text):
            return model.encode([text])[0].tolist()

    embeds = LocalEmbedder()

    store = Chroma.from_documents(
        docs,
        embedding=embeds,
        collection_name=f"fitbot_store_{st.session_state.session_id}",
    )

    return store


# --------------------------
# LLM CHAIN
# --------------------------
@st.cache_resource(show_spinner=False)
def build_chain(api_key):
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.25,
    )

    template = """
You are FitBot, a professional AI fitness coach.
Use the user's profile to give personalized, motivational answers.
Be professional, clear, encouraging.
Never mention "documents", "embeddings", "context", "RAG".

User Profile: {profile}
Conversation: {chat_history}
Relevant Info: {context}
User Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["profile", "chat_history", "context", "question"],
        template=template,
    )

    return LLMChain(llm=llm, prompt=prompt)


# --------------------------
# RAG PIPELINE
# --------------------------
def retrieve_context(store, query, k=3):
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def answer_query(chain, store, query):
    context = retrieve_context(store, query)
    history = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.history[-6:]
    )

    profile = ", ".join(f"{k}: {v}" for k, v in st.session_state.profile.items())

    return chain.predict(
        profile=profile,
        chat_history=history,
        context=context,
        question=query,
    )


# --------------------------
# UI: PROFILE PAGE
# --------------------------
def page_profile():
    st.title("🏋️ FitBot — Personalize your experience")

    name = st.text_input("Your Name", value=st.session_state.profile["name"])
    age = st.number_input("Age", 10, 80, value=st.session_state.profile["age"])
    weight = st.number_input("Weight (kg)", 30, 200, value=st.session_state.profile["weight"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
    goal = st.selectbox("Primary Goal", ["Weight loss", "Muscle gain", "Endurance", "General fitness"])
    level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
    workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"])

    if st.button("✅ Save and Continue"):
        st.session_state.profile.update(
            {
                "name": name,
                "age": age,
                "weight": weight,
                "gender": gender,
                "goal": goal,
                "level": level,
                "diet": diet,
                "workout_time": workout_time,
            }
        )

        st.session_state.profile_submitted = True
        st.session_state.history = []
        st.success("Profile saved!")

        st.rerun()


# --------------------------
# FAQ & Quick Query Logic
# --------------------------
ALL_FAQ = {
    "🏋️ Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🔥 HIIT Routine": "Give me a HIIT workout for fat loss.",
    "🧘 Yoga Routine": "Give me a 10-minute morning yoga stretch session.",
    "🍎 Diet Plan": "Give me a sample balanced diet plan.",
    "🍗 Protein Sources": "What are the best protein sources for muscle gain?",
    "🥗 Vegetarian Diet": "Give me a vegetarian high-protein meal plan.",
    "💤 Sleep": "Explain why sleep is important for fitness.",
    "🚶 Warm-up": "Suggest some dynamic warm-up exercises.",
}


def get_recommended_faq():
    goal = st.session_state.profile["goal"]

    if goal == "Muscle gain":
        return random.sample(
            [
                "🏋️ Beginner Plan",
                "🍗 Protein Sources",
                "🥗 Vegetarian Diet",
                "🔥 HIIT Routine",
            ],
            3,
        )
    elif goal == "Weight loss":
        return random.sample(
            [
                "🔥 HIIT Routine",
                "🍎 Diet Plan",
                "🚶 Warm-up",
                "💤 Sleep",
            ],
            3,
        )
    else:
        return random.sample(list(ALL_FAQ.keys()), 3)


# --------------------------
# UI: CHAT PAGE
# --------------------------
def page_chat():
    st.title("💬 FitBot Chat")

    # Navigation bar top
    cols = st.columns(3)
    if cols[0].button("🏠 Chat"):
        st.session_state.current_page = "Chat"
        st.rerun()
    if cols[1].button("👤 Profile"):
        st.session_state.profile_submitted = False
        st.rerun()
    if cols[2].button("🎯 Challenges"):
        st.session_state.current_page = "Challenges"
        st.rerun()

    kb_text = read_kb()
    store = build_store(kb_text)
    chain = build_chain(GOOGLE_KEY)

    # FAQ Buttons
    st.subheader("⚡ Quick Queries")
    faq_list = get_recommended_faq()
    faq_cols = st.columns(len(faq_list))

    for i, label in enumerate(faq_list):
        if faq_cols[i].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            query = ALL_FAQ[label]
            generate_and_display_answer(chain, store, query)

    # User input
    user_query = st.chat_input("Ask FitBot anything:")
    if user_query:
        generate_and_display_answer(chain, store, user_query)

    # Display last answer
    for turn in st.session_state.history[::-1]:
        st.chat_message("user").markdown(turn["user"])
        st.chat_message("assistant").markdown(turn["assistant"])
        break


# --------------------------
# Helper to handle answer
# --------------------------
def generate_and_display_answer(chain, store, query):
    with st.spinner("Thinking..."):
        start = time.time()
        answer = answer_query(chain, store, query)
        latency = round(time.time() - start, 2)

    st.session_state.history.append(
        {"user": query, "assistant": answer, "time": latency}
    )

    st.chat_message("assistant").markdown(answer)


# --------------------------
# MAIN Router
# --------------------------
def main():
    if st.session_state.profile_submitted:
        page_chat()
    else:
        page_profile()


main()
