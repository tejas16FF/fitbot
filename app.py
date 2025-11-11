# app.py — FitBot (Fitness-theme UI, top centered nav, ripple, fade loading, goal-aware FAQ)
import os, time, random
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain stack (versions pinned in your requirements.txt)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Gamification
from gamification import (
    init_gamification, save_all_state, mark_daily_login,
    reward_for_chat, render_progress_block, get_weekly, complete_challenge
)

# ---------- Config ----------
load_dotenv(".env")
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")
ACCENT = "#00E1C1"

# ---------- CSS (Fitness app style + ripple) ----------
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
    position: relative; overflow: hidden; transition: transform .08s ease, border-color .2s;
  }}
  .navbtn:hover {{ border-color: {ACCENT}; }}
  .navbtn:active {{ transform: scale(.98); }}
  .navbtn.active {{ color: {ACCENT}; border-color: {ACCENT}; }}
  /* ripple */
  .navbtn::after {{
    content: ""; position: absolute; width: 12px; height: 12px; background: {ACCENT};
    border-radius: 50%; opacity: .25; transform: scale(1); transition: transform .5s, opacity .7s;
    pointer-events: none;
  }}
  .navbtn.ripple::after {{ transform: scale(18); opacity: 0; }}
  /* Chat bubbles */
  .bubble-user {{
    background:#1a2438; border:1px solid rgba(255,255,255,.06); padding:12px 14px; border-radius:14px;
  }}
  .bubble-bot {{
    background:#0f1626; border:1px solid rgba(255,255,255,.06); padding:12px 14px; border-radius:14px;
  }}
  /* FAQ button ripple */
  .faqbtn {{
    background:#10182a; color:#cfe; border:1px solid rgba(255,255,255,.06); border-radius:12px; padding:10px;
    font-weight:700; width:100%; position:relative; overflow:hidden;
  }}
  .faqbtn:hover {{ border-color:{ACCENT}; }}
  .faqbtn::after {{
    content:""; position:absolute; left:50%; top:50%; width:8px; height:8px; background:{ACCENT};
    transform: translate(-50%,-50%) scale(1); opacity:.3; border-radius:50%; transition: opacity .6s, transform .5s;
  }}
  .faqbtn.ripple::after {{ transform: translate(-50%,-50%) scale(20); opacity:0; }}
</style>
""", unsafe_allow_html=True)

# ---------- Session ----------
if "page" not in st.session_state: st.session_state.page = "Chat"
if "profile" not in st.session_state:
    st.session_state.profile = {"name":"","age":25,"weight":70,"goal":"General fitness","level":"Beginner","diet":"No preference"}
if "history" not in st.session_state: st.session_state.history = []
if "session_id" not in st.session_state: st.session_state.session_id = random.randint(1_000_000,9_999_999)
init_gamification()

# ---------- Data / KB ----------
def load_kb_text():
    p = "data.txt"
    if os.path.exists(p): 
        with open(p,"r",encoding="utf-8") as f: return f.read()
    return "Train consistently, eat balanced meals, hydrate, and prioritize sleep."

@st.cache_resource(show_spinner=False)
def build_store(text: str):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeds)

@st.cache_resource(show_spinner=False)
def build_chain():
    if not GOOGLE_KEY:
        return None, None
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_KEY, temperature=0.3)
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
    return llm, LLMChain(llm=llm, prompt=prompt)

def retrieve(store, q: str, k=3):
    docs = store.as_retriever(search_kwargs={"k":k}).get_relevant_documents(q)
    return "\n\n---\n\n".join(d.page_content for d in docs)

# ---------- Navbar ----------
def top_nav():
    st.markdown('<div class="topbar"><div class="navwrap">', unsafe_allow_html=True)
    cols = st.columns([1,1,1,1,1,1,1,1])  # spacing around center
    c3,c4,c5,c6 = cols[2], cols[3], cols[4], cols[5]
    def btn(col, label, page):
        active = "active" if st.session_state.page == page else ""
        b = col.button(label, key=f"nav_{page}_{st.session_state.session_id}",
                       help=page, type="secondary")
        st.markdown(f"<script>document.querySelector('button[kind=\"secondary\"][data-testid=\"stWidgetKey-{f'nav_{page}_{st.session_state.session_id}'}\"] )?.classList.add('navbtn','{active}');</script>", unsafe_allow_html=True)
        if b:
            st.session_state.page = page
            st.rerun()
    btn(c3, "🏠 Home", "Chat")
    btn(c4, "🧾 History", "History")
    btn(c5, "🎯 Challenges", "Challenges")
    btn(c6, "👤 Profile", "Profile")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ---------- FAQ (goal-aware) ----------
BASE_FAQ = [
    ("💧 Hydration", "How much water should I drink per day?"),
    ("🧘 Recovery", "What are the best post-workout recovery tips?"),
    ("⏱️ Sleep", "Why is sleep important for muscle recovery?"),
    ("🚶 Warm-up", "Suggest a quick dynamic warm-up before strength training."),
]
GOAL_FAQ = {
    "Muscle gain": [
        ("💪 Hypertrophy Split", "Give me a 4-day muscle gain split."),
        ("🍽️ Protein Targets", "How much protein per day for muscle gain?"),
    ],
    "Weight loss": [
        ("🔥 Fat Loss Plan", "Give me a 5-day weight loss plan with meals."),
        ("🥗 Calorie Deficit", "How to create a sustainable calorie deficit?"),
    ],
    "Endurance": [
        ("🏃 5K Plan", "Create a simple 3-week beginner 5K plan."),
        ("⚡ HIIT", "Give a 20-minute HIIT routine for endurance."),
    ],
    "General fitness": [
        ("🏋️ 3-Day Plan", "Give me a 3-day beginner full-body workout plan."),
        ("🥗 Balanced Diet", "What should a balanced diet include daily?"),
    ],
}

def get_faq_for_profile():
    goal = st.session_state.profile.get("goal","General fitness")
    items = BASE_FAQ + GOAL_FAQ.get(goal, GOAL_FAQ["General fitness"])
    # rotate randomly each render
    return random.sample(items, k=min(6,len(items)))

# ---------- Loading tip (fade in/out) ----------
TIPS = [
    "Consistency beats intensity — show up today.",
    "Hydration powers performance. Sip water.",
    "Form first, then weight. Protect your joints.",
    "Recovery fuels growth. Sleep 7–9 hours.",
    "Tiny habits → big results. Keep going.",
]
def loading_tips():
    html = f"""
    <div id="motibox" style="text-align:center;margin:6px 0;padding:10px;border-radius:10px;
         color:{ACCENT};background:rgba(0,225,193,.08);font-weight:700;transition:opacity .45s;">💭 {random.choice(TIPS)}</div>
    <script>
      const tips = {TIPS};
      let i=0; const box = document.getElementById('motibox');
      function nxt(){{ box.style.opacity=0; setTimeout(()=>{{ box.innerText='💭 '+tips[i]; box.style.opacity=1; i=(i+1)%tips.length; }}, 350); }}
      const t = setInterval(nxt, 2800);
      setTimeout(()=>{{ clearInterval(t); }}, 9000);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------- Pages ----------
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
        diet = st.selectbox("Diet", ["No preference","Vegetarian","Vegan","Non-vegetarian"],
                            index=["No preference","Vegetarian","Vegan","Non-vegetarian"].index(st.session_state.profile["diet"]))
        submitted = st.form_submit_button("Save Profile")
    if submitted:
        st.session_state.profile.update({"name":name,"age":age,"weight":weight,"goal":goal,"level":level,"diet":diet})
        mark_daily_login()
        save_all_state()
        st.success("✅ Profile saved!")
        time.sleep(0.5)
        st.session_state.page = "Chat"; st.rerun()

def page_chat():
    st.header("💬 FitBot — Your AI Fitness Coach")
    kb_text = load_kb_text()
    store = build_store(kb_text)
    llm, chain = build_chain()
    if not llm:
        st.error("❌ GOOGLE_API_KEY not found in .env"); return

    st.subheader("⚡ Quick Questions")
    faq_items = get_faq_for_profile()
    cols = st.columns(3)
    for i,(label,q) in enumerate(faq_items):
        if cols[i%3].button(label, key=f"faq_{st.session_state.session_id}_{i}"):
            run_query(q, store, chain)

    user_q = st.chat_input("Ask anything about workouts, diet, recovery, motivation:")
    if user_q:
        run_query(user_q, store, chain)

    st.markdown("### Recent Q&A")
    if not st.session_state.history:
        st.info("No questions yet.")
    else:
        for t in reversed(st.session_state.history[-8:]):
            with st.expander(f"Q: {t['user'][:72]}"):
                st.markdown(f"<div class='bubble-user'><b>Q:</b> {t['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble-bot' style='margin-top:8px'><b>A:</b> {t['assistant']}</div>", unsafe_allow_html=True)

def run_query(q: str, store, chain):
    placeholder = st.empty()
    loading_tips()  # show tips while working
    with st.spinner("Generating your personalized response..."):
        start = time.time()
        context = retrieve(store, q, k=3)
        history = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in st.session_state.history[-6:]])
        profile = ", ".join([f"{k}: {v}" for k,v in st.session_state.profile.items()])
        try:
            ans = chain.predict(profile=profile, chat_history=history, context=context, question=q)
        except Exception as e:
            ans = "Sorry, I had trouble answering just now. Please try again."
        latency = time.time() - start
    placeholder.empty()
    st.session_state.history.append({"user": q, "assistant": ans, "time": round(latency,2)})
    reward_for_chat()
    save_all_state()
    st.success(ans)

def page_history():
    st.header("🧾 History")
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for i,t in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{i+1}. {t['user'][:90]}"):
                st.markdown(f"<div class='bubble-user'><b>Q:</b> {t['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble-bot' style='margin-top:8px'><b>A:</b> {t['assistant']}</div>", unsafe_allow_html=True)
                st.caption(f"⏱️ {t.get('time',0)}s")

def page_challenges():
    st.header("🎯 Challenges")
    render_progress_block()
    st.markdown("---")
    wk = get_weekly()
    st.subheader(f"Current Weekly Challenge: {wk['title']}")
    st.caption(wk["desc"])
    if st.button("Mark Weekly Challenge Completed", key=f"wk_done_{st.session_state.session_id}"):
        complete_challenge(wk["id"]); save_all_state(); st.rerun()

# ---------- Router ----------
def render_page():
    page = st.session_state.page
    if page == "Chat": page_chat()
    elif page == "History": page_history()
    elif page == "Challenges": page_challenges()
    elif page == "Profile": page_profile()
    else:
        st.session_state.page = "Chat"; page_chat()

# ---------- App ----------
def main():
    top_nav()               # centered, sticky, ripple
    render_page()           # page content

if __name__ == "__main__":
    main()
