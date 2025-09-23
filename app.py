# app.py — FitBot (RAG with HuggingFace + Gemini + FAISS)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv("/workspaces/fitbot/.env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")

# -----------------------------
# Helper Functions
# -----------------------------
def load_data(filepath="data.txt"):
    """Load knowledge base from text file"""
    if not os.path.exists(filepath):
        st.error(f"❌ Knowledge base not found: {filepath}")
        st.stop()
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def build_vectorstore(text):
    """Create FAISS vector store from text"""
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def create_qa_chain(vectorstore):
    """Setup Gemini-powered QA chain with professional tone"""
    if not GOOGLE_KEY:
        st.error("❌ GOOGLE_API_KEY not found in .env. Please add your Gemini API key.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_KEY,
        temperature=0.7
    )

    # ✅ Must include {context} and {question}
    template = """
    You are FitBot, a professional AI fitness coach.
    Always provide clear, polite, and detailed answers
    about workouts, nutrition, recovery, health, and motivation.
    Never mention documents, context, or knowledge base.
    If exact details are missing, provide the best possible
    professional advice instead.
    Keep your tone supportive and encouraging.

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

def answer_query(qa, query):
    """Generate answer for user query"""
    with st.spinner("🤔 Thinking..."):
        return qa.run(query)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")

st.title("💪 FitBot - Your AI Fitness Assistant")
st.markdown(
    """
    Welcome to **FitBot** 👋  
    Ask me about **workouts, diet, recovery, or motivation** and I’ll give you 
    professional, supportive advice.
    """
)

# -----------------------------
# Main Logic
# -----------------------------
text_data = load_data("data.txt")
vectorstore = build_vectorstore(text_data)
qa_chain = create_qa_chain(vectorstore)

st.subheader("💬 Chat with FitBot")
query = st.text_input("Enter your fitness question:")

if query:
    answer = answer_query(qa_chain, query)
    st.success(answer)
else:
    st.info("Type a question above to get started!")
