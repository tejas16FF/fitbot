# app.py — FitBot (Professional UI + Persistent Gamification + Robust fallbacks)
import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# load .env (optional)
load_dotenv(".env")

# try to import heavy optional libs, fall back gracefully if missing
LLM_AVAILABLE = True
EMBEDDING_AVAILABLE = True
GEMINI_AVAILABLE = True

try:
    from langchain.text_splitter import CharacterTextSplitter
except Exception:
    CharacterTextSplitter = None
    LLM_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None
    EMBEDDING_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
    GEMINI_AVAILABLE = False

# gamification (must exist)
from gamification import (
    initialize_gamification,
    update_daily_login,
    reward_for_chat,
    render_progress_sidebar,
    save_all_state,
)

# -----------------------------
# App config & constants
# -----------------------------
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

DATA_FILE = "data.txt"
FALLBACK_KB = """
Regular exercise improves cardiovascular health and builds muscle strength.
A balanced diet should include proteins, carbohydrates, healthy fats, vitamins, and minerals.
"""

ACCENT_COLOR = "#0FB38B"  # visible on dark & light

# extended tips & FAQs
DAILY_TIPS = [
    "Stay hydrated — water boosts your performance.",
    "Small consistent steps lead to big results.",
    "Warm up well to avoid injuries.",
    "Sleep matters — aim for 7–9 hours.",
    "Track progress; celebrate small wins.",
    "Recovery is part of training — respect rest days.",
    "Form > weight — prioritize technique.",
    "Eat enough protein for muscle repair.",
    "Consistency beats intensity with no plan.",
    "A balanced diet supports performance and recovery.",
]

FAQ_QUERIES = {
    "🏋️ 3-Day Plan": "Give me a 3-day beginner full-body workout plan.",
    "🥗 Post-Workout Meal": "What should I eat after my workout for recovery?",
    "💪 Vegetarian Protein": "List high-protein vegetarian foods.",
    "🔥 Fat Loss Tips": "How can I lose fat safely and sustainably?",
    "🧘 Quick Yoga": "Give a 10-minute morning yoga stretch routine.",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
    "💤 Sleep Importance": "Why is sleep crucial for fitness?",
    "🍽️ Calorie Intake": "How to calculate daily calorie needs?",
    "🏃 Cardio Plan": "Suggest a 20-minute fat-burning cardio plan.",
    "⚖️ Maintenance vs Deficit": "How to maintain weight vs create a calorie deficit?"
}

# -----------------------------
# Session-state defaults
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
if "current_page" not in st.session_state:
    st.session_state.current_page = "Profile"
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(1000, 9999)
if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)

# initialize gamification (loads saved state quietly)
initialize_gamification()

# -----------------------------
# Knowledge base loader
# -----------------------------
def read_knowledge_base(path: str = DATA_FILE) -> str:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return FALLBACK_KB
    return FALLBACK_KB

KB_TEXT = read_knowledge_base(DATA_FILE)

# -----------------------------
# Simple local lookup fallback
# -----------------------------
def local_lookup_answer(query: str, kb_text: str) -> str:
    q = query.lower()
    paragraphs = [p.strip() for p in kb_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return "I couldn't find information right now. Try another question."
    scores = []
    qwords = [w for w in q.split() if len(w) > 2]
    for p in paragraphs:
        p_l = p.lower()
        score = sum(1 for w in qwords if w in p_l)
        scores.append((score, p))
    scores.sort(reverse=True, key=lambda x: x[0])
    if scores and scores[0][0] > 0:
        top = [p for s, p in scores if s == scores[0][0]]
        return "\n\n".join(top[:2])
    return "I couldn't find an exact match. General tip: focus on progressive training, balanced diet, hydration, and consistent recovery."

# -----------------------------
# LLM wrapper (graceful)
# -----------------------------
def create_llm_chain_if_possible():
    """Return a callable chain-like object or None."""
    if GEMINI_AVAILABLE and ChatGoogleGenerativeAI is not None and GOOGLE_KEY:
        try:
            # minimal wrapper with predict method to match our usage
            model = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_KEY, temperature=0.25)
            class Chain:
                def __init__(self, model):
                    self.model = model
                def predict(self, **kwargs):
                    # Use model._call or model.generate depending on installed version
                    try:
                        # prefer high-level call if available
                        return self.model._call(kwargs)
                    except Exception:
                        try:
                            out = self.model.generate(kwargs)
                            # try to extract text safely
                            return getattr(out, "generations", [[{"text": str(out)}]])[0][0].get("text", str(out))
                        except Exception:
                            return str(out)
            return Chain(model)
        except Exception:
            return None
    return None

_chain = create_llm_chain_if_possible()

def answer_via_llm_or_local(query: str) -> str:
    if _chain is not None:
        # build profile & history
        profile_str = ", ".join(f"{k}: {v}" for k, v in st.session_state.profile.items() if v)
        chat_hist = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in st.session_state.history[-6:]])
        try:
            return _chain.predict(profile=profile_str, chat_history=chat_hist, context="", question=query)
        except Exception as e:
            return local_lookup_answer(query, KB_TEXT) + f"\n\n(LLM error: {e})"
    else:
        return local_lookup_answer(query, KB_TEXT)

# -----------------------------
# UI helpers
# -----------------------------
def nav_sidebar():
    st.sidebar.title("FitBot")
    st.sidebar.markdown("---")
    st.sidebar.header("Profile")
    p = st.session_state.profile
    st.sidebar.markdown(f"**Name:** {p.get('name','') or '—'}")
    st.sidebar.markdown(f"**Goal:** {p.get('goal','')}")
    st.sidebar.markdown(f"**Level:** {p.get('level','')}")
    st.sidebar.markdown("---")
    # Navigation buttons
    if st.sidebar.button("🏠 Chat"):
        st.session_state.current_page = "Chat"
    if st.sidebar.button("🧾 History"):
        st.session_state.current_page = "History"
    if st.sidebar.button("🎯 Challenges"):
        st.session_state.current_page = "Challenges"
    if st.sidebar.button("🏆 Progress"):
        st.session_state.current_page = "Progress"
    if st.sidebar.button("⚙️ Profile"):
        st.session_state.current_page = "Profile"
    st.sidebar.markdown("---")
    # Gamification snapshot in sidebar (small)
    if st.session_state.current_page != "Profile":
        render_progress_sidebar()
    st.sidebar.caption("FitBot — Capstone Project")

# motivational tip HTML (used only inside loading placeholder)
def motivational_tip_html(tip_text: str) -> str:
    safe_tip = tip_text.replace("'", "\\'")
    html = f"""
    <style>
    #tip_box {{
        text-align:center;
        padding:8px 10px;
        border-radius:10px;
        font-weight:600;
        transition: opacity 0.6s ease-in-out;
        color: {ACCENT_COLOR};
        background: rgba(15,179,139,0.04);
    }}
    @media (prefers-color-scheme: dark) {{
        #tip_box {{ background: rgba(255,255,255,0.02); }}
    }}
    </style>
    <div id="tip_box">💭 {safe_tip}</div>
    <script>
    // rotate tips a few times for pleasant effect
    const tips = {DAILY_TIPS};
    let idx = 0;
    const box = document.getElementById('tip_box');
    function changeTip(){{
        box.style.opacity = 0;
        setTimeout(()=>{{ box.innerText = '💭 '+tips[idx]; box.style.opacity = 1; idx=(idx+1)%tips.length; }}, 400);
    }}
    let t = setInterval(changeTip, 3000);
    // stop after 10 seconds
    setTimeout(()=>{{ clearInterval(t); }}, 10000);
    </script>
    """
    return html

# deterministic button key helper
def faq_button_key(i: int, label: str) -> str:
    safe_label = label.replace(" ", "_").replace(":", "").replace("/", "_")
    return f"faq_{st.session_state.session_id}_{i}_{safe_label}"

# -----------------------------
# Pages
# -----------------------------
def page_profile():
    st.title("🏋️ Profile")
    st.markdown("Enter a few details so FitBot can give tailored answers.")
    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
        age = st.number_input("Age", min_value=10, max_value=90, value=int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=int(st.session_state.profile.get("weight", 70)))
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], index=0)
        goal = st.selectbox("Primary Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"], index=0)
        level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"], index=0)
        diet = st.selectbox("Diet Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"], index=0)
        workout_time = st.selectbox("Preferred Workout Time", ["Morning", "Afternoon", "Evening"], index=0)
        submitted = st.form_submit_button("Save & Continue")
    if submitted:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight,
            "gender": gender, "goal": goal, "level": level,
            "diet": diet, "workout_time": workout_time
        })
        # do not show gamification popups here; update login silently
        update_daily_login(silent=True)
        save_all_state()
        st.success("Profile saved — launching Chat.")
        time.sleep(0.6)
        st.session_state.current_page = "Chat"
        st.rerun()

def page_chat():
    st.title("💬 Chat — FitBot")
    st.markdown(f"**Tip:** {st.session_state.tip_of_the_day}")
    # left-most sidebar already shows profile + small progress

    # Build a small local vectorstore only if available (cached)
    vectorstore = None
    if EMBEDDING_AVAILABLE and FAISS is not None and CharacterTextSplitter is not None:
        try:
            @st.cache_resource(show_spinner=False)
            def _build_vs(kb_text):
                splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                docs = splitter.create_documents([kb_text])
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return FAISS.from_documents(docs, embeddings)
            vectorstore = _build_vs(KB_TEXT)
        except Exception:
            vectorstore = None

    # FAQ selection influenced by profile goal (simple heuristic)
    goal = st.session_state.profile.get("goal", "").lower()
    faqs = list(FAQ_QUERIES.items())
    prioritized = []
    if "muscle" in goal:
        for k, v in faqs:
            if "protein" in v.lower() or "strength" in v.lower() or "plan" in v.lower():
                prioritized.append((k, v))
    elif "weight" in goal:
        for k, v in faqs:
            if "fat" in v.lower() or "cardio" in v.lower():
                prioritized.append((k, v))
    if len(prioritized) < 4:
        # fill rest
        remaining = [item for item in faqs if item not in prioritized]
        prioritized += random.sample(remaining, min(4 - len(prioritized), len(remaining)))
    display_faqs = prioritized[:4] if prioritized else random.sample(faqs, min(4, len(faqs)))

    st.markdown("#### ⚡ Quick Fitness Queries")
    cols = st.columns(len(display_faqs))

    def handle_query(query_text: str):
        # show motivational loading tip (HTML) only during processing
        placeholder = st.empty()
        placeholder.markdown(motivational_tip_html(random.choice(DAILY_TIPS)), unsafe_allow_html=True)
        with st.spinner("Generating personalized answer..."):
            start = time.time()
            try:
                answer = answer_via_llm_or_local(query_text)
            except Exception:
                answer = local_lookup_answer(query_text, KB_TEXT)
            latency = time.time() - start
        # clear tip & append history, reward & persist
        placeholder.empty()
        st.session_state.history.append({"user": query_text, "assistant": answer, "time": latency})
        # reward (quietly)
        try:
            reward_for_chat(show_msg=False)
            # update weekly challenge progress
            from gamification import update_challenge_progress  # safe to import here
            update_challenge_progress("chat")
            save_all_state()
        except Exception:
            pass
        st.success(answer)

    # render FAQ buttons with deterministic keys
    for i, (label, qtext) in enumerate(display_faqs):
        key = faq_button_key(i, label)
        if cols[i].button(label, key=key):
            handle_query(qtext)

    # chat input
    user_q = st.chat_input("Ask FitBot your question:")
    if user_q:
        handle_query(user_q)

    # show recent history
    st.markdown("### 📚 Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet — ask something above!")
    else:
        for idx, turn in enumerate(reversed(st.session_state.history[-20:])):
            with st.expander(f"Q: {turn['user'][:80]}"):
                st.markdown(f"**Q:** {turn['user']}")
                st.markdown(f"**A:** {turn['assistant']}")
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(turn.get("time", 0)))
                st.caption(f"Answered: {ts}")

def page_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No history recorded yet.")
        return
    for i, t in enumerate(reversed(st.session_state.history[-100:])):
        with st.expander(f"Q{i+1}: {t['user'][:80]}"):
            st.markdown(f"**Q:** {t['user']}")
            st.markdown(f"**A:** {t['assistant']}")
            st.caption(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t.get('time', 0)))}")

def page_challenges():
    st.title("🎯 Challenges")
    gam = st.session_state.gamification
    check_and = None
    try:
        # show weekly challenge details via gamification helper
        st.markdown("### Current Weekly Challenge")
        ch = gam.get("weekly_challenge", {})
        if ch:
            st.markdown(f"**{ch.get('desc','-')}**")
            st.markdown(f"Progress: {gam.get('challenge_progress',0)}/{ch.get('target',0)}")
            if gam.get("challenge_completed"):
                st.success("✅ Completed — reward awarded")
        else:
            st.info("No active challenge.")
        st.markdown("---")
        st.markdown("Manual actions (log activity to progress challenges):")
        if st.button("Log: Completed a workout (manual)"):
            # manual action increments challenge progress if type matches 'chat' etc.
            from gamification import update_challenge_progress
            update_challenge_progress("streak")
            save_all_state()
            st.success("Logged workout (manual).")
    except Exception as e:
        st.error(f"Could not load challenges: {e}")

def page_progress():
    st.title("🏆 Progress")
    render_progress_sidebar()
    st.markdown("---")
    st.markdown("### Badges & Achievements")
    badges = st.session_state.gamification.get("badges", [])
    if not badges:
        st.info("No badges yet — keep interacting to unlock.")
    else:
        cols = st.columns(min(4, len(badges)))
        for i, b in enumerate(badges):
            cols[i % len(cols)].metric(label=b, value="Unlocked")

def page_stats():
    st.title("📊 Statistics")
    gam = st.session_state.gamification
    st.markdown(f"**XP:** {gam.get('xp',0)}")
    st.markdown(f"**Level:** {gam.get('level',1)}")
    st.markdown(f"**Streak:** {gam.get('streak',0)} days")
    st.markdown(f"**History length:** {len(st.session_state.history)}")

# -----------------------------
# Answer helpers re-used in pages
# -----------------------------
def answer_via_llm_or_local(query: str) -> str:
    """Wrapper for LLM/local lookup above (keeps same name used earlier)."""
    return answer_via_llm_or_local.__wrapped__(query) if hasattr(answer_via_llm_or_local, "__wrapped__") else local_lookup_answer(query, KB_TEXT)

# ensure the wrapper references our defined function above
# (we create a proper binding to the implementation above)
def _impl_answer(query: str) -> str:
    if _chain is not None:
        try:
            return _chain.predict(profile=", ".join(f"{k}:{v}" for k,v in st.session_state.profile.items()),
                                  chat_history="\n".join([f"User:{h['user']}\nAssistant:{h['assistant']}" for h in st.session_state.history[-6:]]),
                                  context="",
                                  question=query)
        except Exception:
            return local_lookup_answer(query, KB_TEXT)
    else:
        return local_lookup_answer(query, KB_TEXT)

# patch function object
answer_via_llm_or_local.__wrapped__ = _impl_answer
def answer_via_llm_or_local(query: str) -> str:
    return _impl_answer(query)

# -----------------------------
# Main control flow
# -----------------------------
def main():
    nav_sidebar()
    page = st.session_state.current_page or "Profile"
    if page == "Profile":
        page_profile()
    elif page == "Chat":
        page_chat()
    elif page == "History":
        page_history()
    elif page == "Challenges":
        page_challenges()
    elif page == "Progress":
        page_progress()
    elif page == "Statistics":
        page_stats()
    else:
        page_chat()

if __name__ == "__main__":
    main()
