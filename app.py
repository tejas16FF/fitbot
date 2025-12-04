# app.py — FitBot Final (Horizontal FAQ, polished sidebar, permanent avatar with change/delete,
# toasts that auto-fade, removed theme toggle)
# Paste into: C:\Users\tejas\OneDrive\Desktop\fitbot\app.py

import os
import io
import time
import random
import textwrap
from datetime import datetime, date
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageDraw, ImageOps

import pandas as pd
import plotly.express as px

# gamification utilities (must be present in your project)
from gamification import (
    initialize_gamification,
    save_all_state,
    update_daily_login,
    reward_for_chat,
    update_challenge_progress,
    render_progress_sidebar_full,
    render_weekly_challenge_section,
    show_challenge_popup,
    get_gamification_state,
)

# -------------------------
# Env & client
# -------------------------
load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(page_title="FitBot Pro", page_icon="💪", layout="wide")

# -------------------------
# Colors & constants (fixed dark theme)
# -------------------------
ACCENT = "#F9C74F"      # gold accent
BG = "#0E1117"
CARD = "#15171A"
TEXT = "#E8ECEF"
MUTED = "#A7B1B6"
BORDER = "#1F2427"
SUCCESS = "#26DFA6"

AVATAR_DIR = os.path.join(os.getcwd(), "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)

# -------------------------
# Initialize session_state (all keys to avoid AttributeError)
# -------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": 25,
        "weight": 70,
        "goal": "General fitness",
        "level": "Beginner",
        "diet": "No preference",
        "avatar_path": None,
    }

if "profile_completed" not in st.session_state:
    st.session_state.profile_completed = False

if "page" not in st.session_state:
    st.session_state.page = "Profile"

if "faq_set" not in st.session_state:
    st.session_state.faq_set = None

if "workouts" not in st.session_state:
    st.session_state.workouts = []

if "meals" not in st.session_state:
    st.session_state.meals = []

if "water_daily" not in st.session_state:
    st.session_state.water_daily = {}

if "weight_history" not in st.session_state:
    st.session_state.weight_history = []

if "toast_queue" not in st.session_state:
    st.session_state.toast_queue = []

if "history" not in st.session_state:
    st.session_state.history = []

# initialize gamification (reads user_progress.json)
initialize_gamification()

# -------------------------
# CSS + theme injection (fixed dark style)
# -------------------------
def inject_css():
    css = f"""
    <style>
    html, body, .stApp {{
      background: {BG} !important;
      color: {TEXT} !important;
      font-family: Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }}
    .card {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .muted {{ color: {MUTED}; }}
    .accent-btn .stButton>button {{
      background: linear-gradient(90deg, {ACCENT}, #e9c85a) !important;
      color: #071014 !important;
      border-radius: 10px;
      font-weight:700;
    }}
    .small-btn .stButton>button {{
      background: transparent !important;
      color: {MUTED} !important;
      border: 1px solid {BORDER} !important;
    }}
    .user-bubble {{
      background: #182022;
      color: {TEXT};
      padding:10px 12px;
      border-radius:12px;
      max-width:80%;
      display:inline-block;
    }}
    .bot-bubble {{
      background: #0f1516;
      color: {TEXT};
      padding:12px;
      border-radius:12px;
      max-width:90%;
      display:inline-block;
    }}
    /* Toast area */
    #fitbot-toasts {{
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 9999;
      display:flex;
      flex-direction:column;
      gap:10px;
      align-items:flex-end;
    }}
    .fitbot-toast {{
      min-width: 220px;
      padding:10px 14px;
      border-radius:10px;
      font-weight:700;
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    }}
    .fitbot-toast.xp {{ background: {SUCCESS}; color: #04100d; }}
    .fitbot-toast.badge {{ background: {ACCENT}; color: #04100d; }}
    .fitbot-toast.info {{ background: #2B7FFF; color: #04102b; }}
    .fitbot-toast.warn {{ background: #F9C74F; color:#04100d; }}
    /* FAQ horizontal layout */
    .faq-row {{
      display:flex;
      gap:10px;
      margin-bottom:12px;
      flex-wrap:wrap;
    }}
    .faq-btn {{
      padding:8px 12px;
      border-radius:8px;
      border:1px solid {BORDER};
      background:transparent;
      color:{TEXT};
      font-weight:600;
    }}
    /* FAQ answer row: label + answer horizontally */
    .faq-answer-row {{
      display:flex;
      gap:12px;
      margin-top:10px;
      align-items:flex-start;
      width:100%;
    }}
    .faq-label {{
      min-width:90px;
      font-weight:700;
      color:{ACCENT};
      flex:0 0 90px;
    }}
    .faq-answer {{
      flex:1 1 auto;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# -------------------------
# Toasts: queue + render (fades out)
# -------------------------
def render_toasts():
    if not st.session_state.toast_queue:
        return
    html = "<div id='fitbot-toasts'>"
    for t in st.session_state.toast_queue:
        kind = t.get("kind", "info")
        cls = "info"
        if kind in ("xp","success"):
            cls = "xp"
        elif kind == "badge":
            cls = "badge"
        elif kind == "warn":
            cls = "warn"
        html += f"<div class='fitbot-toast {cls}'>{t.get('msg')}</div>"
    html += "</div>"
    # auto remove after 2.8s
    js = """
    <script>
    setTimeout(function(){ 
      let el = document.getElementById('fitbot-toasts'); 
      if(!el) return;
      el.style.transition = "opacity 600ms";
      el.style.opacity = "0";
      setTimeout(()=> el.remove(), 700);
    }, 2800);
    </script>
    """
    st.markdown(html + js, unsafe_allow_html=True)
    st.session_state.toast_queue = []

def app_show_toast(msg: str, kind: str = "info"):
    st.session_state.toast_queue.append({"msg": msg, "kind": kind})

# -------------------------
# Avatar utilities (circle crop + gold border) - style B chosen
# -------------------------
def crop_to_circle_with_border(pil_img: Image.Image, size: int = 240, border_px: int = 6, border_color: str = ACCENT) -> Image.Image:
    img = pil_img.convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)

    border_size = size + 2 * border_px
    bg = Image.new("RGBA", (border_size, border_size), (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(bg)
    bc = tuple(int(border_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    draw_bg.ellipse((0, 0, border_size, border_size), fill=bc + (255,))
    bg.paste(out, (border_px, border_px), out)
    return bg

def save_avatar_file(pil_img: Image.Image, username: str = None) -> str:
    if username and username.strip():
        safe = "".join(c for c in username if c.isalnum() or c in ("_", "-")).strip()
        filename = f"{safe}.png" if safe else f"avatar_{random.randint(1000,9999)}.png"
    else:
        filename = f"avatar_{random.randint(1000,9999)}.png"
    path = os.path.join(AVATAR_DIR, filename)
    pil_img.save(path, format="PNG")
    return path

def remove_avatar_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def load_avatar_bytes(path: str):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

# -------------------------
# KB and GROQ wrapper
# -------------------------
DATA_FILE = "data.txt"
def load_kb():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Exercise, nutrition, hydration, and sleep."

KB_TEXT = load_kb()

def simple_local_retriever(query: str, kb_text: str, top_n: int = 3) -> str:
    q = [w for w in query.lower().split() if w.strip()]
    paras = [p.strip() for p in kb_text.split("\n\n") if p.strip()]
    scored = []
    for p in paras:
        score = sum(1 for w in q if w in p.lower())
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    selected = [p for s, p in scored if s > 0][:top_n]
    if not selected:
        return "General fitness guidance: progressive overload, protein, hydration, rest."
    return "\n\n---\n\n".join(selected)

def ask_llm(prompt_query: str, context: str = ""):
    if client is None:
        return "❌ GROQ API key missing. Add GROQ_API_KEY to .env"
    prompt = textwrap.dedent(f"""
    You are FitBot, a professional fitness & nutrition coach.
    Use the context when relevant. Be concise and practical.
    Context:
    {context}
    User question:
    {prompt_query}
    """)
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=600,
            temperature=0.32,
            top_p=0.9
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GROQ error: {str(e)}"

# -------------------------
# Sidebar (professional arrangement)
# -------------------------
def render_sidebar():
    st.sidebar.markdown(f"<div style='padding:8px 6px; font-weight:700; color:{ACCENT};'>💪 FitBot Pro</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='card'>", unsafe_allow_html=True)

    # Avatar area — upload (Change Photo) + Delete
    st.sidebar.markdown("<div style='text-align:center; margin-bottom:6px;'>", unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader("Change Photo", type=["png", "jpg", "jpeg"], key="avatar_uploader")
    if uploaded is not None:
        try:
            pil = Image.open(uploaded)
            cropped = crop_to_circle_with_border(pil, size=240, border_px=6, border_color=ACCENT)
            username = st.session_state.profile.get("name") or None
            path = save_avatar_file(cropped, username=username)
            # Remove old avatar if exists & different
            old = st.session_state.profile.get("avatar_path")
            if old and old != path:
                remove_avatar_file(old)
            st.session_state.profile["avatar_path"] = path
            save_all_state()
            app_show_toast("Avatar updated", kind="info")
        except Exception:
            app_show_toast("Avatar update failed", kind="warn")

    apath = st.session_state.profile.get("avatar_path")
    if apath and os.path.exists(apath):
        st.sidebar.image(apath, width=120)
        if st.sidebar.button("Delete Photo"):
            remove_avatar_file(apath)
            st.session_state.profile["avatar_path"] = None
            save_all_state()
            app_show_toast("Avatar removed", kind="warn")
    else:
        # placeholder circle
        placeholder = Image.new("RGBA", (240,240), (28,28,28,255))
        buf = io.BytesIO(); placeholder.save(buf, format="PNG"); buf.seek(0)
        st.sidebar.image(buf.getvalue(), width=120)

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Profile name & quick stats
    st.sidebar.markdown(f"<div style='font-weight:700; padding-top:6px'>{st.session_state.profile.get('name','Guest')}</div>", unsafe_allow_html=True)
    gam = get_gamification_state()
    xp = int(gam.get("xp", 0)); lvl = gam.get("level", 1); streak = gam.get("streak", 0)
    st.sidebar.markdown(f"<div style='margin-top:8px'><strong style='color:{ACCENT};'>Level {lvl}</strong></div>", unsafe_allow_html=True)
    pct = min(1.0, (xp % 200) / 200 if xp is not None else 0.0)
    st.sidebar.progress(pct)
    st.sidebar.markdown(f"<div class='muted'>XP: {xp}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='muted'>Streak: {streak} days</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    # Navigation
    if st.sidebar.button("⚙️ Profile"):
        st.session_state.page = "Profile"
    if st.sidebar.button("💬 Chat"):
        if st.session_state.profile_completed:
            st.session_state.page = "Chat"
        else:
            app_show_toast("Complete profile first", "warn")
            st.session_state.page = "Profile"
    if st.sidebar.button("📊 Dashboard"):
        if st.session_state.profile_completed:
            st.session_state.page = "Dashboard"
        else:
            app_show_toast("Complete profile first", "warn")
            st.session_state.page = "Profile"
    if st.sidebar.button("🏋️ Workouts"):
        if st.session_state.profile_completed:
            st.session_state.page = "Workouts"
        else:
            app_show_toast("Complete profile first", "warn")
            st.session_state.page = "Profile"
    if st.sidebar.button("🥗 Diet"):
        if st.session_state.profile_completed:
            st.session_state.page = "Diet"
        else:
            app_show_toast("Complete profile first", "warn")
            st.session_state.page = "Profile"
    if st.sidebar.button("🏆 Progress"):
        if st.session_state.profile_completed:
            st.session_state.page = "Progress"
        else:
            app_show_toast("Complete profile first", "warn")
            st.session_state.page = "Profile"

    # <-- ADDED HISTORY BUTTON -->
    if st.sidebar.button("📜 History"):
        st.session_state.page = "History"
    # <-- end added -->

    # badges list
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-weight:700; margin-bottom:6px;'>Badges</div>", unsafe_allow_html=True)
    badges = gam.get("badges", [])
    if badges:
        for b in badges[:8]:
            st.sidebar.markdown(f"<div class='muted'>• {b}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div class='muted'>No badges yet</div>", unsafe_allow_html=True)

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Pages
# -------------------------
def todays_date_str():
    return date.today().isoformat()

def page_profile():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("⚙️ Profile")
    # Show avatar + delete here as well
    apath = st.session_state.profile.get("avatar_path")
    if apath and os.path.exists(apath):
        st.image(apath, width=160)
        if st.button("Delete Photo (Profile)"):
            remove_avatar_file(apath)
            st.session_state.profile["avatar_path"] = None
            save_all_state()
            app_show_toast("Avatar removed", kind="warn")

    with st.form("profile_form"):
        name = st.text_input("Name", st.session_state.profile.get("name", ""))
        age = st.number_input("Age", 10, 100, int(st.session_state.profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", 20, 300, int(st.session_state.profile.get("weight", 70)))
        goal = st.selectbox("Goal", ["General fitness", "Weight loss", "Muscle gain", "Endurance"], index=0)
        level = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], index=0)
        diet = st.selectbox("Diet", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"], index=0)
        submitted = st.form_submit_button("Save Profile")
    if submitted:
        st.session_state.profile.update({"name": name, "age": age, "weight": weight, "goal": goal, "level": level, "diet": diet})
        st.session_state.profile_completed = True
        st.session_state.weight_history.append({"date": todays_date_str(), "weight": weight})
        update_daily_login(silent=True)
        save_all_state()
        show_challenge_popup("Profile completed — welcome!", kind="info")
        time.sleep(0.35)
        st.session_state.page = "Chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def page_chat():
    st.title("💬 Chat")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Horizontal FAQ row (above chat input)
    faqs = [
        ("🏋️ Plan", "Create a 3-day beginner workout plan."),
        ("🥗 Meal", "What to eat after a workout?"),
        ("💪 Protein", "List high-protein vegetarian foods."),
        ("🔥 Fat Loss", "Give safe tips for fat loss."),
    ]
    # Render as columns (horizontal). If narrow screen, they will wrap.
    cols = st.columns(len(faqs))
    for i, (label, qtext) in enumerate(faqs):
        with cols[i]:
            if st.button(label, key=f"faq_{i}"):
                ph = st.empty(); ph.info("Thinking...")
                ctx = simple_local_retriever(qtext, KB_TEXT)
                ans = ask_llm(qtext, ctx)
                ph.empty()
                # record into history (same shape used by History page)
                st.session_state.history.append({"user": qtext, "assistant": ans, "time": time.time()})
                reward_for_chat()
                update_challenge_progress("chat")
                show_challenge_popup("Answered — +10 XP", kind="xp")
                save_all_state()
                # render horizontally: label left, answer right (full width)
                html = f"""
                <div class='faq-answer-row'>
                  <div class='faq-label'>{label}</div>
                  <div class='faq-answer'><div class='bot-bubble'>{ans}</div></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

    # Chat input below FAQ row
    user_q = st.chat_input("Ask FitBot anything...")
    if user_q:
        ph = st.empty(); ph.info("Thinking...")
        ctx = simple_local_retriever(user_q, KB_TEXT)
        ans = ask_llm(user_q, ctx)
        ph.empty()
        st.session_state.history.append({"user": user_q, "assistant": ans, "time": time.time()})
        reward_for_chat()
        update_challenge_progress("chat")
        show_challenge_popup("Nice question — +10 XP", kind="xp")
        save_all_state()
        st.markdown(f"<div class='bot-bubble'>{ans}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def page_dashboard():
    st.title("📊 Dashboard")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    gam = get_gamification_state()
    xp = int(gam.get("xp", 0)); lvl = gam.get("level", 1); streak = gam.get("streak", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Level", lvl)
    c2.metric("XP", xp)
    c3.metric("Streak", streak)

    wh = pd.DataFrame(st.session_state.weight_history)
    if not wh.empty:
        wh['date'] = pd.to_datetime(wh['date'])
        fig = px.line(wh, x='date', y='weight', markers=True, title="Weight Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weight history yet.")

    if st.session_state.workouts:
        df = pd.DataFrame(st.session_state.workouts)
        counts = df['type'].value_counts().reset_index()
        counts.columns = ['type', 'count']
        fig2 = px.bar(counts, x='type', y='count', title="Workouts by Type")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No workouts logged.")

    st.markdown("</div>", unsafe_allow_html=True)

def page_workouts():
    st.title("🏋️ Workouts")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("log_workout"):
        wtype = st.selectbox("Workout Type", ["Strength", "Cardio", "Yoga", "HIIT", "Other"])
        dur = st.number_input("Duration (min)", min_value=5, max_value=300, value=30)
        cal = st.number_input("Calories", min_value=0, max_value=5000, value=200)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Log")
    if submitted:
        entry = {"date": datetime.now().isoformat(), "type": wtype, "duration": dur, "calories": cal, "notes": notes}
        st.session_state.workouts.append(entry)
        save_all_state()
        update_challenge_progress("manual")
        show_challenge_popup(f"Workout logged — +{20 + (dur//30)*10} XP", kind="xp")
        st.success("Logged")
    if st.session_state.workouts:
        st.markdown("### Recent")
        for w in reversed(st.session_state.workouts[-6:]):
            st.write(f"{w['date'][:19]} — {w['type']} • {w['duration']}m • {w['calories']} kcal")
    st.markdown("</div>", unsafe_allow_html=True)

def page_diet():
    st.title("🥗 Diet")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("log_meal"):
        mtype = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner", "Snack"])
        kcal = st.number_input("Calories", min_value=0, max_value=10000, value=400)
        protein = st.number_input("Protein (g)", min_value=0, max_value=1000, value=25)
        carbs = st.number_input("Carbs (g)", min_value=0, max_value=1000, value=40)
        fat = st.number_input("Fat (g)", min_value=0, max_value=500, value=12)
        submitted = st.form_submit_button("Log Meal")
    if submitted:
        meal = {"date": datetime.now().isoformat(), "meal_type": mtype, "calories": kcal, "protein": protein, "carbs": carbs, "fat": fat}
        st.session_state.meals.append(meal)
        save_all_state()
        show_challenge_popup("Meal logged — +5 XP", kind="xp")
        st.success("Meal saved")

    today = todays_date_str()
    current = st.session_state.water_daily.get(today, 0)
    add = st.number_input("Add glasses", min_value=1, max_value=20, value=1, step=1)
    if st.button("Log Water"):
        st.session_state.water_daily[today] = current + add
        save_all_state()
        if st.session_state.water_daily[today] >= 8 and current < 8:
            show_challenge_popup("Hydration goal achieved — +15 XP", kind="success")
        else:
            show_challenge_popup(f"Logged {add} glass(es). Total today: {st.session_state.water_daily[today]}", kind="info")
    st.markdown("</div>", unsafe_allow_html=True)

def page_history():
    st.title("📜 History")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No chats yet.")
    else:
        for i, t in enumerate(reversed(st.session_state.history[-200:]), 1):
            with st.expander(f"{i}. {t['user'][:80]}"):
                st.write(f"Q: {t['user']}")
                st.write(f"A: {t['assistant']}")
                st.caption(datetime.fromtimestamp(t.get("time", time.time())).isoformat())
    st.markdown("</div>", unsafe_allow_html=True)

def page_progress():
    st.title("🏆 Progress")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_progress_sidebar_full()
    render_weekly_challenge_section()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
def todays_date_str():
    return date.today().isoformat()

# -------------------------
# Layout: render sidebar (no toggle) + toasts
# -------------------------
def render_layout():
    # top-left brand
    c1, c2, c3 = st.columns([2, 6, 2])
    with c1:
        st.markdown(f"<div style='font-weight:700; color:{ACCENT};'>💪 FitBot Pro</div>", unsafe_allow_html=True)
    # right column left intentionally empty to keep layout clean

    # sidebar
    render_sidebar()
    # render toasts each run
    render_toasts()

# -------------------------
# MAIN
# -------------------------
def main():
    render_layout()
    page = st.session_state.page
    if page == "Profile":
        page_profile()
    elif page == "Chat":
        page_chat()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Workouts":
        page_workouts()
    elif page == "Diet":
        page_diet()
    elif page == "History":
        page_history()
    elif page == "Progress":
        page_progress()
    else:
        page_chat()

if __name__ == "__main__":
    main()
