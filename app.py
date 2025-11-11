# app.py — FitBot (Gemini LLM + HF embeddings, no LangChain; all queries go to LLM)
import os
import json
import time
import random
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Optional gamification hooks (will be no-ops if file not present)
try:
    from gamification import (
        initialize_gamification,
        update_daily_login,
        reward_for_chat,
        update_challenge_progress,
        render_progress_sidebar_full,
        render_weekly_challenge_section,
        show_challenge_popup,
        save_all_state,
        reset_progress_file,
    )
    GAMIFY = True
except Exception:
    GAMIFY = False

# ---- Gemini (google-generativeai) ----
import google.generativeai as genai

# ---- HF embeddings (no faiss/chroma) ----
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-pro")     # "gemini-1.5-pro" also works
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ACCENT = "#0FB38B"
DATA_FILE = "data.txt"
INDEX_EMB = "kb_embeddings.npy"
INDEX_JSON = "kb_chunks.json"

if not GOOGLE_KEY:
    st.error("❌ GOOGLE_API_KEY missing in .env")
    st.stop()

genai.configure(api_key=GOOGLE_KEY)

# -----------------------------
# SESSION DEFAULTS
# -----------------------------
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
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(10**6, 10**9)
if "tip_after_profile" not in st.session_state:
    st.session_state.tip_after_profile = None

if GAMIFY:
    initialize_gamification()

# -----------------------------
# UTIL: read kb
# -----------------------------
def read_kb() -> str:
    if os.path.exists(DATA_FILE):
        return open(DATA_FILE, "r", encoding="utf-8").read()
    return """Regular exercise improves cardiovascular health and builds muscle strength.
A balanced diet should include proteins, carbohydrates, healthy fats, vitamins, and minerals.
"""

# -----------------------------
# CHUNK + EMBED + CACHE
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)

def chunk_text(text: str, size=450, overlap=90):
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p in paras:
        start = 0
        while start < len(p):
            end = start + size
            chunks.append(p[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(p):
                break
    # Fallback if nothing split
    if not chunks:
        chunks = [text[:size]]
    return chunks

def build_or_load_index(kb_text: str):
    """Persist small numpy index to disk; rebuild if text changed."""
    # hash by length+first/last part
    sig = {"len": len(kb_text), "head": kb_text[:200], "tail": kb_text[-200:]}
    need_build = not (os.path.exists(INDEX_EMB) and os.path.exists(INDEX_JSON))
    idx_meta = None
    if not need_build:
        try:
            idx_meta = json.load(open(INDEX_JSON, "r", encoding="utf-8"))
            if idx_meta.get("sig") != sig:
                need_build = True
        except Exception:
            need_build = True

    if need_build:
        chunks = chunk_text(kb_text)
        model = get_embedder()
        embs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        json.dump({"sig": sig, "chunks": chunks}, open(INDEX_JSON, "w", encoding="utf-8"))
        np.save(INDEX_EMB, embs)
        return chunks, embs
    else:
        chunks = idx_meta["chunks"]
        embs = np.load(INDEX_EMB)
        return chunks, embs

def top_k_context(query: str, chunks, embs, k=4):
    model = get_embedder()
    q = model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
    # cosine since normalized
    sims = embs @ q
    top_idx = np.argsort(-sims)[:k]
    ctx = "\n\n".join(chunks[i] for i in top_idx)
    return ctx

# -----------------------------
# LLM (Gemini) — single place
# -----------------------------
def ask_gemini(system_msg: str, user_msg: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)  # 'gemini-pro' or 'gemini-1.5-pro'
    prompt = f"{system_msg.strip()}\n\n{user_msg.strip()}"
    try:
        resp = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        return "I couldn’t generate a response at the moment. Please try again."
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# -----------------------------
# PROMPT
# -----------------------------
def build_user_prompt(context: str, profile: dict, question: str) -> str:
    p = profile or {}
    prof = ", ".join(f"{k}: {v}" for k, v in p.items() if str(v).strip())
    return f"""You are FitBot, a professional, helpful fitness coach. Use the provided context only as helpful background.
Always give clear, practical steps. Never mention ‘knowledge base’ or ‘documents’. If something is unknown, give the
best safe advice.

User Profile: {prof}

Relevant notes from my fitness docs:
{context}

Question: {question}

Answer in a supportive, professional tone:
"""

SYSTEM_MSG = "You are a trusted AI fitness coach. Be concise, accurate, and encouraging."

# -----------------------------
# NAV BAR (top centered)
# -----------------------------
def render_top_nav():
    active = st.session_state.page
    def btn(label, page):
        cls = "active" if active == page else ""
        return f'<button name="nav" value="{page}" class="nav-btn {cls}">{label}</button>'

    st.markdown(f"""
    <style>
    .top-wrap {{
        position: sticky; top: 0; z-index: 999;
        background: rgba(0,0,0,0.02);
        backdrop-filter: blur(6px);
        border-bottom: 1px solid rgba(0,0,0,0.08);
        padding: 10px 0 6px 0;
        margin-bottom: 12px;
    }}
    .nav-bar {{ display:flex; gap:10px; justify-content:center; }}
    .nav-btn {{
        border: 1px solid rgba(0,0,0,.12);
        background: white;
        padding: 8px 14px; border-radius: 10px; font-weight: 700; cursor: pointer;
    }}
    .nav-btn.active {{ border-color: {ACCENT}; color: {ACCENT}; }}
    @media (prefers-color-scheme: dark) {{
        .nav-btn {{ background: #111; color: #EAEAEA; border: 1px solid #333; }}
        .nav-btn.active {{ color: {ACCENT}; border-color: {ACCENT}; }}
        .top-wrap {{ background: rgba(255,255,255,0.03); border-bottom: 1px solid #333; }}
    }}
    </style>
    <div class="top-wrap">
      <form method="get">
        <div class="nav-bar">
          {btn("🏠 Chat", "Chat")}
          {btn("📜 History", "History")}
          {btn("🎯 Challenges", "Challenges")}
          {btn("🏆 Progress", "Progress")}
          {btn("⚙️ Profile", "Profile")}
        </div>
      </form>
    </div>
    """, unsafe_allow_html=True)

    nav = st.query_params.get("nav")
    if nav:
        st.session_state.page = nav
        st.query_params.clear()
        st.rerun()

# -----------------------------
# PAGES
# -----------------------------
FAQS = [
    ("🏋️ 3-Day Plan", "Give me a 3-day beginner full-body workout plan."),
    ("🥗 Post-Workout Meal", "What should I eat after my workout for better recovery?"),
    ("💪 Vegetarian Protein", "List high-protein vegetarian foods and sample meals."),
    ("🔥 Fat Loss Tips", "How can I lose body fat safely and consistently?"),
    ("🧘 Quick Yoga", "Give me a 10-minute morning yoga stretch routine."),
    ("🚶 Warm-up Ideas", "Suggest dynamic warm-up exercises before a workout."),
]

def page_profile():
    st.title("🏋️ Create Your Fitness Profile")
    st.markdown("_We’ll personalize your coaching from this info._")

    with st.form("profile"):
        name = st.text_input("Name", st.session_state.profile.get("name",""))
        age = st.number_input("Age", 10, 80, int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", 30, 200, int(st.session_state.profile.get("weight", 70)))
        goal = st.selectbox("Primary Goal", ["General fitness","Weight loss","Muscle gain","Endurance"],
                            index=["General fitness","Weight loss","Muscle gain","Endurance"].index(st.session_state.profile.get("goal","General fitness")))
        level = st.selectbox("Experience Level", ["Beginner","Intermediate","Advanced"],
                            index=["Beginner","Intermediate","Advanced"].index(st.session_state.profile.get("level","Beginner")))
        diet = st.selectbox("Diet", ["No preference","Vegetarian","Vegan","Non-vegetarian"],
                            index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile.get("diet","No preference")))
        ok = st.form_submit_button("Save & Continue")

    if ok:
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight, "goal": goal, "level": level, "diet": diet
        })
        if GAMIFY:
            update_daily_login(silent=True); save_all_state()
        st.session_state.tip_after_profile = random.choice([
            "Consistency beats intensity — train smart and steady.",
            "Fuel your body well and your workouts will follow.",
            "Recovery is training — sleep, hydrate, stretch.",
        ])
        st.session_state.page = "Chat"
        st.success("✅ Profile saved. Opening chat…")
        time.sleep(0.6)
        st.rerun()

def do_answer(user_query: str):
    # loading animation with rotating tips
    tips = [
        "Small steps daily lead to big wins.",
        "Hydration powers your performance.",
        "Form first, then intensity.",
        "Recovery fuels growth.",
        "Discipline > motivation. Show up.",
    ]
    ph = st.empty()
    ph.markdown(
        f"""
        <div id="motibox" style="text-align:center; margin:10px 0; padding:10px; border-radius:10px;
        color:{ACCENT}; background:rgba(15,179,139,.06); font-weight:700; transition:opacity .5s;">
        💭 {random.choice(tips)}
        </div>
        <script>
        const tips = {tips};
        let idx=0; const box=document.getElementById('motibox');
        function nxt(){ box.style.opacity=0; setTimeout(()=>{{ box.innerText='💭 '+tips[idx]; box.style.opacity=1; idx=(idx+1)%tips.length; }},350); }
        let tm=setInterval(nxt,3000); setTimeout(()=>{{ clearInterval(tm); }}, 9000);
        </script>
        """,
        unsafe_allow_html=True
    )

    # Retrieval
    kb_text = read_kb()
    chunks, embs = build_or_load_index(kb_text)
    context = top_k_context(user_query, chunks, embs, k=4)

    # LLM
    user_prompt = build_user_prompt(context, st.session_state.profile, user_query)
    start = time.time()
    answer = ask_gemini(SYSTEM_MSG, user_prompt)
    latency = round(time.time() - start, 2)

    ph.empty()
    st.session_state.history.append({"user": user_query, "assistant": answer, "time": latency})
    if GAMIFY:
        try:
            reward_for_chat(show_msg=False)
            update_challenge_progress("chat")
            save_all_state()
        except Exception:
            pass
    st.success(answer)

def page_chat():
    st.title("💬 FitBot — Chat")

    if st.session_state.tip_after_profile:
        st.info(f"💡 Tip of the Day: {st.session_state.tip_after_profile}")
        st.session_state.tip_after_profile = None

    st.caption("Ask me anything about workouts, diet, recovery, or motivation.")

    # FAQ buttons → ALL go to LLM
    cols = st.columns(3)
    for i, (label, q) in enumerate(FAQS):
        if cols[i % 3].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            do_answer(q)

    # Manual input → also LLM
    user_q = st.chat_input("Your question…")
    if user_q:
        do_answer(user_q)

    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for turn in reversed(st.session_state.history[-6:]):
            with st.expander(f"Q: {turn['user'][:72]}"):
                st.markdown(f"**A:** {turn['assistant']}")
                st.caption(f"⏱️ {turn.get('time', '-') } s")

def page_history():
    st.title("📜 History")
    if not st.session_state.history:
        st.info("No chats yet.")
        return
    for i, t in enumerate(reversed(st.session_state.history[-50:])):
        with st.expander(f"Q {i+1}: {t['user'][:80]}"):
            st.markdown(f"**Q:** {t['user']}")
            st.markdown(f"**A:** {t['assistant']}")
            st.caption(f"⏱️ {t.get('time','-')} s")

def page_challenges():
    st.title("🎯 Challenges")
    if GAMIFY:
        render_progress_sidebar_full()
        render_weekly_challenge_section()
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        if c1.button("Log workout (manual)"):
            update_challenge_progress("manual"); save_all_state()
            st.success("Logged!")
        if c2.button("Log check-in (manual)"):
            update_challenge_progress("login"); save_all_state()
            st.success("Logged!")
        if c3.button("Claim weekly reward"):
            st.info("✅ If completed, reward was auto-claimed.")
    else:
        st.info("Gamification is not active in this environment.")

def page_progress():
    st.title("🏆 Progress")
    if GAMIFY:
        render_progress_sidebar_full()
        badges = st.session_state.gamification.get("badges", [])
        st.markdown("### Badges")
        st.write(", ".join(badges) if badges else "No badges yet.")
    else:
        st.info("Gamification is not active in this environment.")

# -----------------------------
# ROUTER
# -----------------------------
def main():
    if st.session_state.page != "Profile":
        render_top_nav()

    page = st.session_state.page
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
    else:
        page_chat()

if __name__ == "__main__":
    main()
