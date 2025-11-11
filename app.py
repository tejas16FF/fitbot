# app.py — FitBot (Option B: memory-only, TF-IDF retriever) + Gemini + Gamification
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv

# LLM (LangChain wrapper for Gemini)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Lightweight TF-IDF retriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gamification layer
from gamification import (
    init_gamification, update_daily_streak, gain_xp,
    render_challenges_page, render_achievements_page,
    check_all_achievements, record_query_metrics
)

# ---------------- Config / Theme ----------------
load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")
ACCENT = "#00E1C1"

st.markdown(f"""
<style>
  html, body, .stApp {{
    background: radial-gradient(1200px 800px at 50% -10%, #0b1220 0%, #0b0f1a 40%, #0a0d17 100%) !important;
    color: #E6F5F2;
  }}
  .topbar {{
    position: sticky; top: 0; z-index: 1000; background: rgba(11,16,26,.72);
    backdrop-filter: blur(8px); border-bottom: 1px solid rgba(255,255,255,.06);
    padding: 8px 0; margin: -1rem -1rem 0 -1rem;
  }}
  .navwrap {{ display: flex; justify-content: center; gap: 10px; }}
  .navbtn {{
    background: #121a2a; color: #cfe; border: 1px solid rgba(255,255,255,.06);
    padding: 10px 14px; border-radius: 14px; font-weight: 700; cursor: pointer;
  }}
  .navbtn:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
  .bubble-user {{
    background:#1a2438; border:1px solid rgba(255,255,255,.06); padding:12px 14px; border-radius:14px;
  }}
  .bubble-bot {{
    background:#0f1626; border:1px solid rgba(255,255,255,.06); padding:12px 14px; border-radius:14px;
  }}
  .faqbtn {{
    background:#10182a; color:#cfe; border:1px solid rgba(255,255,255,.06); border-radius:12px; padding:10px;
    font-weight:700; width:100%;
  }}
  .faqbtn:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
</style>
""", unsafe_allow_html=True)

# ---------------- Session ----------------
if "page" not in st.session_state: st.session_state.page = "Chat"
if "session_id" not in st.session_state: st.session_state.session_id = random.randint(1_000_000, 9_999_999)

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "", "age": 25, "weight": 70,
        "goal": "General fitness", "level": "Beginner", "gender": "Prefer not to say",
        "diet": "No preference", "workout_time": "Morning"
    }
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize gamification once profile exists
init_gamification(profile=st.session_state.profile)

# ---------------- Knowledge Base ----------------
def read_kb():
    p = "data.txt"
    if os.path.exists(p):
        return open(p, "r", encoding="utf-8").read()
    return "General fitness knowledge only."

# ---------------- TF-IDF Retriever ----------------
class SimpleEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = None
        self.docs = []

    def fit(self, documents):
        self.docs = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def query(self, text, top_k=3):
        if not self.docs:
            return [], []
        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        return top_idx, scores[top_idx]

def split_into_chunks(text: str, chunk_size=600, overlap=100):
    """Simple fixed-size splitter to avoid extra deps."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(chunk_size - overlap, 1)
    return chunks if chunks else [text]

def build_store(text: str):
    chunks = [c.strip() for c in split_into_chunks(text, 180, 30) if c.strip()]
    embedder = SimpleEmbedder()
    embedder.fit(chunks)
    return chunks, embedder

def retrieve_context(chunks, embedder, query, k=3):
    idxs, _ = embedder.query(query, top_k=k)
    return "\n\n---\n\n".join([chunks[i] for i in idxs])

# ---------------- LLM ----------------
def build_chain(api_key):
    if not api_key:
        return None
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["profile","chat_history","context","question"],
        template=(
            "You are FitBot, a professional, supportive AI fitness coach.\n"
            "Use the user's profile to personalize, and never mention documents or knowledge base.\n\n"
            "Profile: {profile}\n"
            "Conversation so far:\n{chat_history}\n\n"
            "Relevant info:\n{context}\n\n"
            "User question: {question}\n\n"
            "Answer:"
        ),
    )
    return LLMChain(llm=llm, prompt=prompt)

# ---------------- Top Navbar ----------------
def top_nav():
    st.markdown('<div class="topbar"><div class="navwrap">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("🏠 Chat", key=f"nav_chat_{st.session_state.session_id}"):
        st.session_state.page = "Chat"; st.rerun()
    if c2.button("🧾 History", key=f"nav_hist_{st.session_state.session_id}"):
        st.session_state.page = "History"; st.rerun()
    if c3.button("🎯 Challenges", key=f"nav_chal_{st.session_state.session_id}"):
        st.session_state.page = "Challenges"; st.rerun()
    if c4.button("🏆 Achievements", key=f"nav_ach_{st.session_state.session_id}"):
        st.session_state.page = "Achievements"; st.rerun()
    if c5.button("👤 Profile", key=f"nav_prof_{st.session_state.session_id}"):
        st.session_state.page = "Profile"; st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

# ---------------- FAQ ----------------
ALL_FAQ = {
    "🏋️ 3-Day Beginner Plan": "Give me a 3-day beginner full-body workout plan.",
    "🔥 HIIT Routine": "Give me a 20-minute HIIT routine for fat loss.",
    "🧘 Mobility Flow": "Give me a 10-minute morning mobility/stretching routine.",
    "🍎 Balanced Diet": "Give me a sample balanced diet plan for general fitness.",
    "🍗 Protein Sources": "What are the best protein sources for muscle gain?",
    "🥗 Veg High-Protein": "Create a vegetarian high-protein meal plan.",
    "💤 Sleep & Recovery": "Why is sleep important for muscle recovery?",
    "🚶 Dynamic Warm-up": "Give me a dynamic warm-up before a leg workout.",
}
GOAL_FAQ = {
    "Muscle gain": ["🏋️ 3-Day Beginner Plan","🍗 Protein Sources","🚶 Dynamic Warm-up"],
    "Weight loss": ["🔥 HIIT Routine","🍎 Balanced Diet","🧘 Mobility Flow"],
    "Endurance": ["🧘 Mobility Flow","🔥 HIIT Routine","🍎 Balanced Diet"],
    "General fitness": ["🏋️ 3-Day Beginner Plan","🍎 Balanced Diet","🧘 Mobility Flow"],
}

def goal_rec_faq():
    goal = st.session_state.profile.get("goal","General fitness")
    base = ["💤 Sleep & Recovery"]
    pool = GOAL_FAQ.get(goal, GOAL_FAQ["General fitness"])
    items = base + pool
    return random.sample(items, k=min(4, len(items)))

# ---------------- Loading Tips ----------------
TIPS = [
    "Consistency beats intensity — show up today.",
    "Hydration powers performance. Sip water.",
    "Form first, then weight. Protect your joints.",
    "Recovery fuels growth. Sleep 7–9 hours.",
    "Tiny habits → big results. Keep going.",
]
def loading_tips():
    html = f"""
    <div id="motibox" style="text-align:center;margin:8px 0;padding:10px;border-radius:10px;
         color:{ACCENT};background:rgba(0,225,193,.08);font-weight:700;transition:opacity .45s;">💭 {random.choice(TIPS)}</div>
    <script>
      const tips = {TIPS};
      let i=0; const box = document.getElementById('motibox');
      function nxt(){{ box.style.opacity=0; setTimeout(()=>{{ box.innerText='💭 '+tips[i]; box.style.opacity=1; i=(i+1)%tips.length; }}, 350); }}
      const t = setInterval(nxt, 2600);
      setTimeout(()=>{{ clearInterval(t); }}, 8300);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Pages ----------------
def page_profile():
    st.header("👤 Your Profile")
    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile["name"])
        age = st.number_input("Age", 10, 80, int(st.session_state.profile["age"]))
        weight = st.number_input("Weight (kg)", 30, 200, int(st.session_state.profile["weight"]))
        goal = st.selectbox("Goal", ["General fitness","Weight loss","Muscle gain","Endurance"],
                            index=["General fitness","Weight loss","Muscle gain","Endurance"].index(st.session_state.profile["goal"]))
        level = st.selectbox("Experience", ["Beginner","Intermediate","Advanced"],
                             index=["Beginner","Intermediate","Advanced"].index(st.session_state.profile["level"]))
        gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"],
                              index=["Male","Female","Other","Prefer not to say"].index(st.session_state.profile["gender"]))
        diet = st.selectbox("Diet", ["No preference","Vegetarian","Vegan","Non-vegetarian"],
                            index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile["diet"]))
        workout_time = st.selectbox("Preferred Workout Time", ["Morning","Afternoon","Evening"],
                            index=["Morning","Afternoon","Evening"].index(st.session_state.profile["workout_time"]))
        submitted = st.form_submit_button("Save Profile")
    if submitted:
        st.session_state.profile.update({"name":name,"age":age,"weight":weight,"goal":goal,"level":level,"gender":gender,"diet":diet,"workout_time":workout_time})
        update_daily_streak()
        check_all_achievements(st.session_state.profile)
        st.success("✅ Profile saved!")
        time.sleep(0.4)
        st.session_state.page = "Chat"; st.rerun()

def run_query(q: str, chunks, embedder, chain):
    loading_tips()
    with st.spinner("Generating your personalized response..."):
        start = time.time()
        context = retrieve_context(chunks, embedder, q, k=3)
        history = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in st.session_state.history[-6:]])
        profile = ", ".join([f"{k}: {v}" for k,v in st.session_state.profile.items()])
        try:
            ans = chain.predict(profile=profile, chat_history=history, context=context, question=q)
        except Exception:
            ans = "Sorry, I had trouble answering just now. Please try again."
        latency = round(time.time() - start, 2)

    st.session_state.history.append({"user": q, "assistant": ans, "time": latency})
    gain_xp(5, profile=st.session_state.profile)
    record_query_metrics(q)
    check_all_achievements(st.session_state.profile)

    st.chat_message("assistant").markdown(ans)

def page_chat():
    st.header("💬 FitBot — Your AI Fitness Coach")
    kb_text = read_kb()
    chunks, embedder = build_store(kb_text)
    chain = build_chain(GOOGLE_KEY)
    if not chain:
        st.error("❌ GOOGLE_API_KEY not found in .env")
        return

    st.subheader("⚡ Quick Queries")
    faq_list = goal_rec_faq()
    cols = st.columns(len(faq_list))
    for i, label in enumerate(faq_list):
        if cols[i].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            q = ALL_FAQ[label]
            run_query(q, chunks, embedder, chain)

    user_q = st.chat_input("Ask anything about workouts, diet, recovery, motivation:")
    if user_q:
        run_query(user_q, chunks, embedder, chain)

    if st.session_state.history:
        st.markdown("### Recent Q&A")
        for t in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {t['user'][:72]}"):
                st.markdown(f"<div class='bubble-user'><b>Q:</b> {t['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble-bot' style='margin-top:8px'><b>A:</b> {t['assistant']}</div>", unsafe_allow_html=True)

def page_history():
    st.header("🧾 History")
    if not st.session_state.history:
        st.info("No chats yet.")
        return
    for i, t in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{i+1}. {t['user'][:90]}"):
            st.markdown(f"<div class='bubble-user'><b>Q:</b> {t['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble-bot' style='margin-top:8px'><b>A:</b> {t['assistant']}</div>", unsafe_allow_html=True)
            st.caption(f"⏱️ {t.get('time',0)}s")

def page_challenges():
    render_challenges_page(profile=st.session_state.profile)

def page_achievements():
    render_achievements_page(profile=st.session_state.profile)

# ---------------- Router ----------------
def top_nav():
    st.markdown('<div class="topbar"><div class="navwrap">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("🏠 Chat", key=f"nav_chat_{st.session_state.session_id}"):
        st.session_state.page = "Chat"; st.rerun()
    if c2.button("🧾 History", key=f"nav_hist_{st.session_state.session_id}"):
        st.session_state.page = "History"; st.rerun()
    if c3.button("🎯 Challenges", key=f"nav_chal_{st.session_state.session_id}"):
        st.session_state.page = "Challenges"; st.rerun()
    if c4.button("🏆 Achievements", key=f"nav_ach_{st.session_state.session_id}"):
        st.session_state.page = "Achievements"; st.rerun()
    if c5.button("👤 Profile", key=f"nav_prof_{st.session_state.session_id}"):
        st.session_state.page = "Profile"; st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_page():
    if st.session_state.page == "Chat":
        page_chat()
    elif st.session_state.page == "History":
        page_history()
    elif st.session_state.page == "Challenges":
        page_challenges()
    elif st.session_state.page == "Achievements":
        page_achievements()
    elif st.session_state.page == "Profile":
        page_profile()
    else:
        st.session_state.page = "Chat"; page_chat()

def main():
    # First-time users go to Profile until they save
    if not st.session_state.profile.get("name"):
        st.session_state.page = "Profile"
    top_nav()
    render_page()

if __name__ == "__main__":
    main()
