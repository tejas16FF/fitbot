# app.py — FitBot (Gemini 2.5 Pro + LangChain Community + Emoji Top Nav + Gamification)

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# Modern LangChain community modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini integration
import google.generativeai as genai
from google.api_core.exceptions import NotFound, PermissionDenied, ResourceExhausted

# Gamification
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    render_progress_sidebar_full,
    render_weekly_challenge_section,
    save_all_state,
)

# -----------------------------------
# Streamlit setup
# -----------------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
load_dotenv(".env")

# Hide Streamlit menu/footer/sidebar
st.markdown("""
<style>
#MainMenu, header, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ACCENT = "#00C49A"

# -----------------------------------
# Session defaults
# -----------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": 25,
        "weight": 70,
        "goal": "General fitness",
        "level": "Beginner",
        "diet": "No preference",
    }
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Profile"

initialize_gamification()

# -----------------------------------
# Load Knowledge Base
# -----------------------------------
DATA_FILE = "data.txt"

@st.cache_data(show_spinner=False)
def load_kb_text():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Exercise regularly, eat balanced meals, and stay hydrated."

@st.cache_resource(show_spinner="📚 Loading fitness knowledge base...")
def build_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(docs, embeds)

KB_TEXT = load_kb_text()
VSTORE = build_store(KB_TEXT)

def get_kb_context(query):
    docs = VSTORE.similarity_search(query, k=2)
    return "\n".join(d.page_content for d in docs)

# -----------------------------------
# Gemini 2.5 Pro Integration
# -----------------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_gemini(query: str, context: str = "") -> str:
    """Ask Gemini 2.5 Pro safely with friendly fallback handling."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={
                "max_output_tokens": 900,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 40,
            },
            safety_settings="block_none"
        )

        prompt = f"""
You are FitBot — a certified AI fitness and nutrition assistant.
Provide concise, motivational, and scientifically safe answers.

Context:
{context}

User Question:
{query}
"""
        resp = model.generate_content(prompt)

        # ✅ Safe extraction
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    text = " ".join(
                        getattr(p, "text", "") for p in c.content.parts if hasattr(p, "text")
                    ).strip()
                    if text:
                        return text

            reason = getattr(resp.candidates[0], "finish_reason", "unknown")
            if reason == 2:
                return "⚠️ Gemini filtered this content for safety. Try rephrasing your question."

        return "⚠️ Gemini didn’t return a valid response. Please try again."

    except NotFound:
        return "❌ Gemini 2.5 Pro model not found. Please verify your API setup."
    except PermissionDenied:
        return "🔒 You don’t have access to Gemini 2.5 Pro. Check your billing or permissions."
    except ResourceExhausted:
        return "⚠️ Gemini quota exceeded — try again later."
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# -----------------------------------
# Top emoji navigation bar
# -----------------------------------
def render_top_nav():
    st.markdown(f"""
    <style>
    .top-nav {{
        position: fixed; top:0; left:0; right:0;
        background: rgba(255,255,255,0.9);
        border-bottom:1px solid rgba(0,0,0,0.1);
        backdrop-filter:blur(8px);
        display:flex; justify-content:center; gap:16px; padding:10px;
        z-index:1000;
    }}
    .nav-btn {{
        padding:8px 14px; border-radius:12px;
        border:1px solid rgba(0,0,0,0.08);
        background:white; font-weight:600; color:#111;
        text-decoration:none; transition:all .2s ease;
    }}
    .nav-btn:hover{{background:#f5f5f5;}}
    .nav-btn.active{{border-color:{ACCENT};color:{ACCENT};box-shadow:0 0 6px rgba(0,196,154,0.3);}}
    </style>
    <div class="top-nav">
      <form method="get">
        <button name="nav" value="Chat" class="nav-btn {'active' if st.session_state.page=='Chat' else ''}">💬 Chat</button>
        <button name="nav" value="History" class="nav-btn {'active' if st.session_state.page=='History' else ''}">📜 History</button>
        <button name="nav" value="Challenges" class="nav-btn {'active' if st.session_state.page=='Challenges' else ''}">🎯 Challenges</button>
        <button name="nav" value="Progress" class="nav-btn {'active' if st.session_state.page=='Progress' else ''}">🏆 Progress</button>
        <button name="nav" value="Profile" class="nav-btn {'active' if st.session_state.page=='Profile' else ''}">⚙️ Profile</button>
      </form>
    </div>
    <div style="height:55px;"></div>
    """, unsafe_allow_html=True)

    nav_target = st.query_params.get("nav")
    if nav_target:
        st.session_state.page = nav_target
        st.query_params.clear()
        st.rerun()

# -----------------------------------
# Page definitions
# -----------------------------------
def page_profile():
    st.title("⚙️ Create Your Fitness Profile")
    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile.get("name", ""))
        age = st.number_input("Age", 10, 80, int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", 30, 200, int(st.session_state.profile.get("weight", 70)))
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"])
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"])
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight,
            "goal": goal, "level": level, "diet": diet
        })
        update_daily_login(silent=True)
        save_all_state()
        st.success("✅ Profile saved! Redirecting to Chat...")
        time.sleep(1)
        st.session_state.page = "Chat"
        st.rerun()

def page_chat():
    st.title("💬 Chat with FitBot")
    user_q = st.chat_input("Ask anything about fitness, nutrition or workouts:")
    if user_q:
        with st.spinner("🤖 Thinking..."):
            context = get_kb_context(user_q)
            answer = ask_gemini(user_q, context)
            reward_for_chat()
            update_challenge_progress("chat")
            save_all_state()
            st.session_state.history.append({"user": user_q, "assistant": answer})
        st.success(answer)

    st.markdown("### Quick FAQs")
    faqs = [
        "Create a 3-day beginner workout plan.",
        "What should I eat after a workout?",
        "List high-protein vegetarian foods.",
        "Share healthy fat-loss tips.",
        "10-minute morning yoga stretch.",
        "Dynamic warm-up ideas before workout.",
    ]
    cols = st.columns(3)
    for i, q in enumerate(faqs):
        if cols[i % 3].button(q, key=f"faq_{i}"):
            with st.spinner("🧠 Generating answer..."):
                ctx = get_kb_context(q)
                ans = ask_gemini(q, ctx)
                reward_for_chat()
                update_challenge_progress("chat")
                save_all_state()
                st.session_state.history.append({"user": q, "assistant": ans})
            st.success(ans)

    st.markdown("### Recent Conversations")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {turn['user'][:70]}"):
                st.markdown(f"**A:** {turn['assistant']}")

def page_history():
    st.title("📜 Chat History")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for t in reversed(st.session_state.history[-50:]):
            with st.expander(f"Q: {t['user'][:70]}"):
                st.markdown(f"**A:** {t['assistant']}")

def page_challenges():
    st.title("🎯 Weekly Challenges")
    render_progress_sidebar_full()
    render_weekly_challenge_section()

def page_progress():
    st.title("🏆 Progress & Badges")
    render_progress_sidebar_full()
    badges = st.session_state.gamification.get("badges", [])
    if badges:
        st.success(", ".join(badges))
    else:
        st.info("No badges earned yet.")

# -----------------------------------
# Main
# -----------------------------------
def main():
    if st.session_state.page != "Profile":
        render_top_nav()

    page = st.session_state.page
    if page == "Profile": page_profile()
    elif page == "Chat": page_chat()
    elif page == "History": page_history()
    elif page == "Challenges": page_challenges()
    elif page == "Progress": page_progress()
    else: page_chat()

if __name__ == "__main__":
    main()
