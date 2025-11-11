# gamification.py — lightweight gamification (no heavy deps)
import json, os, random, time
import streamlit as st
from datetime import date, datetime, timedelta

_STATE_PATH = "gamification_state.json"

def _defaults():
    return {
        "xp": 0,
        "level": 1,
        "streak": 0,
        "last_login": None,
        "chat_count": 0,
        "completed": [],
        "weekly": {"id": None, "title": None, "desc": None, "xp": 100, "started_on": None},
    }

def _load_state():
    if not os.path.exists(_STATE_PATH):
        return _defaults()
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _defaults()
        base.update(data)
        if "weekly" not in base or not isinstance(base["weekly"], dict):
            base["weekly"] = _defaults()["weekly"]
        return base
    except Exception:
        return _defaults()

def _save_state():
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.gam, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

def init_gamification():
    if "gam" not in st.session_state:
        st.session_state.gam = _load_state()
        _ensure_weekly()

def save_all_state():
    if "gam" in st.session_state:
        _save_state()

def _recalc_level():
    # Level every 200 XP
    lvl = 1 + st.session_state.gam["xp"] // 200
    if lvl > st.session_state.gam["level"]:
        st.session_state.gam["level"] = lvl
        show_toast(f"🎉 Level up! You’re now Level {lvl}.")

def add_xp(amount: int):
    st.session_state.gam["xp"] += int(amount)
    _recalc_level()
    _save_state()

def reward_for_chat(xp: int = 10):
    st.session_state.gam["chat_count"] += 1
    add_xp(xp)
    _save_state()

def mark_daily_login():
    today = date.today()
    last = st.session_state.gam.get("last_login")
    if last:
        try:
            last_d = datetime.fromisoformat(last).date()
            delta = (today - last_d).days
            if delta == 1:
                st.session_state.gam["streak"] += 1
            elif delta >= 1:
                st.session_state.gam["streak"] = 1
        except Exception:
            st.session_state.gam["streak"] = 1
    else:
        st.session_state.gam["streak"] = 1
    st.session_state.gam["last_login"] = today.isoformat()
    add_xp(50)
    _save_state()

def get_xp(): return int(st.session_state.gam["xp"])
def get_level(): return int(st.session_state.gam["level"])
def get_streak(): return int(st.session_state.gam["streak"])
def get_chat_count(): return int(st.session_state.gam["chat_count"])

_POOL = [
    {"id":"hydra_7","title":"Hydration Hero","desc":"Drink water 7 times today.","xp":100},
    {"id":"steps_5k","title":"5K Steps","desc":"Walk at least 5,000 steps.","xp":120},
    {"id":"stretch_10","title":"Stretch 10","desc":"Stretch for 10 minutes.","xp":90},
    {"id":"sleep_8","title":"Sleep 8h","desc":"Sleep 8 hours tonight.","xp":110},
    {"id":"push_50","title":"50 Push-ups","desc":"Accumulate 50 push-ups.","xp":140},
    {"id":"cardio_20","title":"Cardio 20","desc":"20 minutes of cardio.","xp":130},
]

def _assign_new_weekly():
    item = random.choice(_POOL)
    st.session_state.gam["weekly"] = {
        "id": item["id"], "title": item["title"], "desc": item["desc"],
        "xp": item["xp"], "started_on": date.today().isoformat()
    }
    _save_state()

def _ensure_weekly():
    wk = st.session_state.gam["weekly"]
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

def get_weekly():
    _ensure_weekly()
    return st.session_state.gam["weekly"]

def complete_challenge(ch_id: str):
    if ch_id in st.session_state.gam["completed"]:
        return False
    st.session_state.gam["completed"].append(ch_id)
    add_xp(get_weekly()["xp"])
    show_toast("🎯 Weekly challenge completed! +XP")
    _save_state()
    return True

def show_toast(message: str, duration_ms: int = 2600):
    uid = random.randint(100000, 999999)
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
    # Simple progress to next level (every 200 XP)
    nxt = 200
    cur = get_xp() % nxt
    st.caption("XP toward next level")
    st.progress(min(1.0, cur / nxt))
