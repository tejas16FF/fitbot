import os
import time
import streamlit as st
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_FOLDER_PATH = "."
# *** FINAL COLOR ***: Vibrant Amber/Gold (#FFC107) for high contrast and energy
UNIVERSAL_TIP_COLOR = "#FFC107" 

load_dotenv(".env")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") 
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# -----------------------------
# KNOWLEDGE BASE & DYNAMIC DATA
# -----------------------------
FALLBACK_KB = """
BEGIN FITNESS KB
Day 1: Squats (3x10), Push-ups (3x8), Plank (3x30s)
Day 2: Cardio: Running 30 mins, Jump rope 10 mins
Day 3: Pull-ups (3x5), Dumbbell curls (3x10), Shoulder press (3x8)
Nutrition: High protein diet, avoid processed sugar.
END FITNESS KB
"""

# ✅ Extended Motivational Tips
DAILY_TIPS = [
    "💡 Stay hydrated — your muscles need water to perform well!",
    "🔥 Small progress each day adds up to big results!",
    "🧘 Focus on your form, not the weight. Perfect form builds strength safely!",
    "🏋️ You don’t have to be extreme, just consistent.",
    "🥗 Nutrition fuels your body — eat smart, not less.",
    "💪 Every rep counts — stay disciplined, not motivated.",
    "🧠 Rest days recharge your progress. Don’t skip recovery!",
    "🚶 Take a walk after meals to aid digestion.",
    "📈 Track your progress weekly — results build slowly, but surely.",
    "🕐 Time and patience beat intensity and shortcuts.",
]

# ✅ Extended FAQ Queries
FAQ_QUERIES = {
    "💪 Beginner Plan": "Give me a 3-day beginner workout plan.",
    "🍎 Post-workout Meal": "What’s a good meal after exercise?",
    "🔥 Motivation Tips": "Share ways to stay consistent.",
    "🧘 Yoga Routine": "Give me a 10-minute morning yoga stretch plan.",
    "💧 Hydration": "How much water should I drink per day?",
    "⏱️ Sleep": "Why is sleep important for muscle recovery?",
    "🍽️ Calorie Intake": "How do I calculate my daily calorie needs?",
    "🏃 Cardio Routine": "Give me a 20-minute fat-burning cardio plan.",
    "🍳 Protein Sources": "List best vegetarian protein sources.",
    "🥤 Supplements": "Should I use protein shakes for weight loss?",
    "😴 Recovery Tips": "What are best recovery tips after intense workout?",
    "⚖️ Fat Loss vs Muscle Gain": "How can I lose fat without losing muscle?",
    "🏋️ Strength Plan": "Give me a 4-day strength training split.",
    "🥗 Balanced Diet": "What should a balanced diet include for daily fitness?",
    "🚶 Warm-up Ideas": "Suggest dynamic warm-up exercises before a workout.",
}

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "", "age": 25, "weight": 70, "goal": "Weight loss", "level": "Beginner", 
        "gender": "Prefer not to say", "diet": "No preference", "workout_time": "Morning"
    }

if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

if "tip_of_the_day" not in st.session_state:
    st.session_state.tip_of_the_day = None
    
if "selected_turn_index" not in st.session_state:
    st.session_state.selected_turn_index = -1 

# -----------------------------
# HELPERS
# -----------------------------
def read_knowledge_base(path="data.txt") -> str:
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else FALLBACK_KB

@st.cache_resource(show_spinner=False)
def build_vectorstore(text: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(FAISS_FOLDER_PATH) and any(f.startswith(FAISS_INDEX_PATH) for f in os.listdir(FAISS_FOLDER_PATH)):
        try:
            vectorstore = FAISS.load_local(FAISS_FOLDER_PATH, embeddings, FAISS_INDEX_PATH)
            return vectorstore
        except Exception:
            pass

    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_FOLDER_PATH, index_name=FAISS_INDEX_PATH)
    return vectorstore

@st.cache_resource(show_spinner=False)
def create_llm_chain(api_key: str):
    if not api_key:
        return None, None

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.3
    )

    template = """
You are FitBot, a professional and friendly AI fitness coach. Respond in a helpful, supportive, and safe manner. NEVER mention internal mechanics (like "knowledge base", "context", or "retrieved docs").
If very specific medical guidance is requested, give a general guideline and recommend consulting a professional.
If the question is completely out of scope, politely refuse and state your specialization: "I specialize in fitness and wellness."

User Profile: {profile}
Fitness Level: {level}
Gender: {gender}

Conversation so far: {chat_history}
Context: {context}
User Question: {question}

Provide detailed, helpful, and encouraging answers.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["profile", "level", "gender", "chat_history", "context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return llm, chain

def retrieve_context(vectorstore, query: str, k=3) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)

def format_history(history: List[Dict[str, Any]], limit=6) -> str:
    recent = history[-limit:]
    return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])

def generate_answer(chain, vectorstore, query, profile, history):
    context_query = f"{history[-1]['user'] if history else ''} {query}" 
    context = retrieve_context(vectorstore, context_query)
    
    if not context.strip():
        context = "General fitness knowledge is used if specific context is unavailable."

    chat_str = format_history(history)
    profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items() if v)
    
    try:
        return chain.predict(
            profile=profile_str, 
            level=profile.get('level', 'Beginner'), 
            gender=profile.get('gender', 'Prefer not to say'), 
            chat_history=chat_str, 
            context=context, 
            question=query
        )
    except Exception as e:
        st.error(f"Model Error: {e}")
        return "⚠️ Sorry, I'm having trouble answering right now. Please try again later."

# -----------------------------
# PAGE: PROFILE SETUP
# -----------------------------
def page_profile_setup():
    st.title("💪 Welcome to FitBot! Let's get started.")
    st.markdown("Please enter your details to get **personalized fitness advice**.")
    
    with st.form("profile_form"):
        st.subheader("Personal Details")
        name = st.text_input("Your Name", value=st.session_state.profile.get("name", ""))
        
        st.subheader("Fitness Metrics")
        age = st.text_input("Age (Years)", value=str(st.session_state.profile.get("age", 25)))
        weight = st.text_input("Weight (kg)", value=str(st.session_state.profile.get("weight", 70)))
        
        gender_options = ["Male", "Female", "Other", "Prefer not to say"]
        gender = st.selectbox("Gender", gender_options, index=gender_options.index(st.session_state.profile.get("gender", "Prefer not to say")))
        
        goal_options = ["Muscle gain", "Weight loss", "Endurance", "General health"]
        goal = st.selectbox("Primary Goal", goal_options, index=goal_options.index(st.session_state.profile.get("goal", "Weight loss")))
        
        level_options = ["Beginner", "Intermediate", "Advanced"]
        level = st.selectbox("Level", level_options, index=level_options.index(st.session_state.profile.get("level", "Beginner")))

        # Additional Profile Details
        diet_options = ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"]
        diet = st.selectbox("Diet Preference", diet_options, index=diet_options.index(st.session_state.profile.get("diet", "No preference")))

        time_options = ["Morning", "Afternoon", "Evening"]
        workout_time = st.selectbox("Preferred Workout Time", time_options, index=time_options.index(st.session_state.profile.get("workout_time", "Morning")))
        
        submitted = st.form_submit_button("Start FitBot")
    
    if submitted and all([name, age, weight]):
        st.session_state.profile.update({
            "name": name, "age": age, "weight": weight, "goal": goal, "level": level, 
            "gender": gender, "diet": diet, "workout_time": workout_time
        })
        st.session_state.user_api_key = GOOGLE_KEY 
        st.session_state.profile_submitted = True
        st.session_state.tip_of_the_day = random.choice(DAILY_TIPS)
        st.rerun()
    elif submitted:
        st.error("Please fill in your name, age, and weight to continue.")

# -----------------------------
# PAGE: MAIN CHAT
# -----------------------------
def page_chat():
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")
    st.title("💬 FitBot — AI Fitness Coach")
    st.caption(f"Welcome back, {st.session_state.profile['name']}! Your goal is **{st.session_state.profile['goal']}**.")

    # Load RAG components (cached)
    llm_key = st.session_state.get('user_api_key', GOOGLE_KEY)
    if not llm_key:
        st.error("Fatal Error: Developer GOOGLE_API_KEY is not set in the environment.")
        return
        
    with st.spinner(f"Preparing RAG components: loading knowledge base and FAISS index..."):
        kb_text = read_knowledge_base("data.txt")
        vectorstore = build_vectorstore(kb_text)
        llm, llm_chain = create_llm_chain(llm_key)
        
        if llm_chain is None:
            st.error("Setup Error: Failed to initialize LLM. Check developer's API key validity.")
            return

    # --- LEFT SIDEBAR (Native): PROFILE ---
    with st.sidebar:
        st.subheader("👤 Your Profile")
        st.markdown("---")
        for k, v in st.session_state.profile.items():
            st.markdown(f"**{k.capitalize()}**: {v}")
        st.markdown("---")
        if st.button("Change Profile", use_container_width=True):
            st.session_state.profile_submitted = False
            st.session_state.selected_turn_index = -1
            st.rerun()

    # --- Layout: Main Content (Active Chat) and Right Column (History Index) ---
    col_chat, col_history_index = st.columns([2, 1]) # 2/3 for chat, 1/3 for history index

    # --- RIGHT COLUMN (INDEX): INTERACTIVE HISTORY ---
    with col_history_index:
        st.subheader("📚 History Index")
        st.caption("Click a question to view the full response.")
        
        history_labels = [f"Q{i+1}: {turn['user'][:40]}..." for i, turn in enumerate(st.session_state.history)]
        history_labels.append("Ask New Question")
        
        default_index = len(st.session_state.history) if st.session_state.selected_turn_index == -1 else st.session_state.selected_turn_index
        
        selected_label = st.radio(
            "Past Questions",
            options=history_labels,
            index=default_index,
            key='history_radio',
            label_visibility='collapsed'
        )

        if selected_label == "Ask New Question":
            st.session_state.selected_turn_index = -1
        else:
            st.session_state.selected_turn_index = history_labels.index(selected_label)
            
        st.markdown("---")
        if st.button("Clear All History", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected_turn_index = -1
            st.rerun()

    # --- CENTER COLUMN (CHAT): ACTIVE CONVERSATION ---
    with col_chat:
        
        # Display Motivational Tip (FIXED COLOR: #FFC107 - Amber)
        tip_html = f"""
        <div style="text-align:center; padding:10px 0; font-size:16px; color: {UNIVERSAL_TIP_COLOR}; font-weight: bold;">
            {st.session_state.tip_of_the_day}
        </div>
        """
        st.markdown(tip_html, unsafe_allow_html=True)


        # --- QUICK START BUTTONS (FIXED LOGIC) ---
        st.markdown("---")
        st.markdown("**Quick Start Questions:**")
        button_cols = st.columns(5)
        faq_items = list(FAQ_QUERIES.items())
        
        for i in range(len(button_cols)):
            if i < 5 and i < len(faq_items):
                label, query = faq_items[i]
                # FIX: Use key to ensure all buttons work and store the query
                if button_cols[i].button(label, key=f"faq_{i}"):
                    st.session_state["last_quick"] = query
                    st.session_state.selected_turn_index = -1 
                    st.rerun() 
        st.markdown("---")

        if st.session_state.selected_turn_index == -1:
            # --- Display New Chat View ---
            st.subheader("Start a New Conversation")
            
            # Main Chat Input 
            initial_input = st.session_state.pop("last_quick", "") 
            user_query = st.chat_input("Ask FitBot your question (Press Enter to submit):", key="main_input")

            # Handle execution (Quick Button OR Enter Key)
            if user_query or initial_input:
                final_query = user_query if user_query else initial_input
                
                # Run pipeline
                with st.spinner(f"🤔 Thinking with Gemini, retrieving context..."):
                    start = time.time()
                    resp = generate_answer(llm_chain, vectorstore, final_query, st.session_state.profile, st.session_state.history)
                    latency = time.time() - start

                # Save history and select the new turn (switches view to show full response)
                st.session_state.history.append({"user": final_query, "assistant": resp, "time": latency})
                st.session_state.selected_turn_index = len(st.session_state.history) - 1 # Select the new turn
                st.rerun()
                
            # Show the latest active conversation only
            if st.session_state.history and st.session_state.selected_turn_index == len(st.session_state.history) - 1:
                latest_turn = st.session_state.history[-1]
                with st.chat_message("user"):
                    st.markdown(latest_turn['user'])
                with st.chat_message("assistant"):
                    st.markdown(latest_turn['assistant'])

        else:
            # --- Display Selected History Section ---
            selected_turn = st.session_state.history[st.session_state.selected_turn_index]
            
            st.subheader("Selected Query:")
            with st.chat_message("user"):
                st.markdown(selected_turn['user'])
            
            st.subheader("FitBot Full Response:")
            with st.chat_message("assistant"):
                st.markdown(selected_turn['assistant'])
            
            st.caption(f"⏱️ Response Time: {selected_turn.get('time', 0):.2f}s")
            
            if st.button("Ask a New Question", key="back_to_new_q"):
                st.session_state.selected_turn_index = -1
                st.rerun()

# -----------------------------
# CONTROL FLOW
# -----------------------------
if st.session_state.profile_submitted:
    page_chat()
else:
    st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")
    page_profile_setup()
    
# Footer (Display globally)
st.markdown("---")
st.caption("FitBot — Capstone Project (RAG, Memory, Personalization). Always consult a licensed professional for medical issues.")
