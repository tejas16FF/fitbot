# gamification.py — Pro, minimal API used by app.py
import json, os, random, time
import streamlit as st
from datetime import date, datetime, timedelta

STATE_FILE = "gamification_state.json"

# ---------------- Core state ----------------
def _default():
    return {
        "xp": 0,
        "streak": 0,
        "last_login": None,
        "level": 1,
        "chat_count": 0,
        "completed": [],
        "weekly": {
            "id": None, "title": None, "desc": None, "xp": 100, "started_on": None
        }
    }

def _load():
    if not os.path.exists(STATE_FILE): return _default()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f: data = json.load(f)
        base = _default(); base.update(data)
        if "weekly" not in base or not isinstance(base["weekly"], dict):
            base["weekly"] = _default()["weekly"]
        return base
    except Exception:
        return _default()

def _save(s): 
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(s, f, indent=2)
    except Exception: pass

def init_gamification():
    if "gstate" not in st.session_state:
        st.session_state.gstate = _load()
        _ensure_weekly()

def save_all_state():
    if "gstate" in st.session_state: _save(st.session_state.gstate)

# ---------------- Level / XP ----------------
def _recalc_level():
    s = st.session_state.gstate
    lvl = 1 + s["xp"] // 200
    if lvl > s["level"]:
        s["level"] = lvl
        show_toast(f"🎉 Level up! You’re now Level {lvl}.")

def add_xp(amount: int, reason: str = ""):
    s = st.session_state.gstate
    s["xp"] += int(amount)
    _recalc_level()
    save_all_state()

def reward_for_chat(xp: int = 10):
    s = st.session_state.gstate
    s["chat_count"] += 1
    add_xp(xp, "Chat")
    save_all_state()

# ---------------- Streak (login) ------------
def mark_daily_login():
    s = st.session_state.gstate
    today = date.today()
    last = s.get("last_login")
    if last:
        try:
            last_d = datetime.fromisoformat(last).date()
            if (today - last_d) == timedelta(days=1):
                s["streak"] += 1
            elif (today - last_d).days >= 1:
                s["streak"] = 1
        except Exception:
            s["streak"] = 1
    else:
        s["streak"] = 1
    s["last_login"] = today.isoformat()
    add_xp(50, "Daily login")
    save_all_state()

def get_xp(): return int(st.session_state.gstate["xp"])
def get_level(): return int(st.session_state.gstate["level"])
def get_streak(): return int(st.session_state.gstate["streak"])
def get_chat_count(): return int(st.session_state.gstate["chat_count"])

# ---------------- Weekly challenge ----------
POOL = [
    {"id":"hydra_7","title":"Hydration Hero","desc":"Drink water 7 times today.","xp":100},
    {"id":"steps_5k","title":"5K Steps","desc":"Walk at least 5,000 steps.","xp":120},
    {"id":"stretch_10","title":"Stretch 10","desc":"Stretch for 10 minutes.","xp":90},
    {"id":"sleep_8","title":"Sleep 8h","desc":"Sleep 8 hours tonight.","xp":110},
    {"id":"push_50","title":"50 Push-ups","desc":"Accumulate 50 push-ups.","xp":140},
    {"id":"cardio_20","title":"Cardio 20","desc":"20 minutes of cardio.","xp":130},
]

def _ensure_weekly():
    s = st.session_state.gstate
    wk = s["weekly"]
    today = date.today()
    if not wk["id"] or not wk["started_on"]:
        _assign_new_weekly()
        return
    try:
        started = datetime.fromisoformat(wk["started_on"]).date()
        if (today - started).days >= 7:
            _assign_new_weekly()
    except Exception:
        _assign_new_weekly()

def _assign_new_weekly():
    s = st.session_state.gstate
    item = random.choice(POOL)
    s["weekly"] = {
        "id": item["id"], "title": item["title"], "desc": item["desc"],
        "xp": item["xp"], "started_on": date.today().isoformat()
    }
    save_all_state()

def get_weekly(): 
    _ensure_weekly()
    return st.session_state.gstate["weekly"]

def complete_challenge(ch_id: str):
    s = st.session_state.gstate
    if ch_id in s["completed"]: return False
    s["completed"].append(ch_id)
    add_xp(get_weekly()["xp"])
    show_toast("🎯 Weekly challenge completed! +XP")
    save_all_state()
    return True

# ---------------- UI helpers ----------------
def ring(progress: float, total: float, label: str, height=220):
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(progress),
            title={"text":label},
            gauge={
                "axis":{"range":[None, float(total)]},
                "bar":{"color":"#00E1C1"},
                "bgcolor":"white",
                "borderwidth":1, "bordercolor":"#ddd",
            },
        ))
        fig.update_layout(margin=dict(l=12,r=12,t=40,b=12), height=height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(label); st.progress(min(1.0, progress/max(total,1.0)))

def show_toast(message: str, duration_ms: int = 2800):
    uid = random.randint(100000,999999)
    safe = str(message).replace("'", "\\'")
    html = f"""
    <div id="toast_{uid}" style="
      position: fixed; top: 18px; right: -420px; z-index: 9999;
      background: #0FB38B; color: #fff; padding: 12px 16px;
      border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.25);
      font-weight: 700; min-width: 240px; transition: right .35s ease, opacity .35s ease;">
      {safe}
    </div>
    <script>
      const el = document.getElementById('toast_{uid}');
      el.style.right = '18px';
      setTimeout(()=>{{ el.style.opacity=0; el.style.right='-420px'; }}, {duration_ms});
      setTimeout(()=>{{ el.remove(); }}, {duration_ms}+450);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True); time.sleep(0.05)

def render_progress_block():
    st.subheader("🏆 Your Progress")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Level", get_level())
    with c2: st.metric("XP", get_xp())
    with c3: st.metric("🔥 Streak (days)", get_streak())
    st.markdown("---")
    wk = get_weekly()
    st.markdown(f"### 🎯 Weekly: **{wk['title']}**")
    st.caption(wk["desc"]); ring(progress=get_xp()%200, total=200, label="XP toward next level")

def reset_gamification():
    st.session_state.gstate = _default(); save_all_state()
