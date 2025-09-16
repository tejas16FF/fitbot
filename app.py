# app.py — FitBot (RAG with HuggingFace + Gemini + FAISS)
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/workspaces/fitbot/.env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro")  # default Gemini model

if not GOOGLE_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env. Please add your Gemini API key.")
    st.stop()

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- Streamlit UI ----
st.set_page_config(page_title="FitBot", page_icon="💪", layout="wide")

st.title("💪 FitBot - Your AI Fitness Assistant")
st.markdown(
    """
    Welcome to **FitBot**!  
    Ask me anything about **workouts, nutrition, recovery, or motivation**.  
    Powered by **Hugging Face embeddings + FAISS vector search + Google Gemini LLM**.
    """
)

# ---- Load knowledge base ----
kb_path = "data.txt"
if not os.path.exists(kb_path):
    st.error(f"❌ Knowledge base not found: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    text = f.read()

# ---- Split text ----
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([text])

# ---- Embeddings (Hugging Face) ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---- Vectorstore (FAISS) ----
vectorstore = FAISS.from_documents(docs, embeddings)

# ---- Gemini LLM ----
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_KEY)

# ---- RAG Chain ----
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    chain_type="stuff"
)

# ---- Chat UI ----
st.subheader("💬 Chat with FitBot")
query = st.text_input("Ask a fitness question:")

if query:
    with st.spinner("🤔 Thinking..."):
        answer = qa.run(query)
    st.success(answer)
else:
    st.info("Type your question above to get started!")
