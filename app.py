# app.py  (Gemini + LangChain + FAISS RAG)
import os
import streamlit as st
from dotenv import load_dotenv

# load .env
load_dotenv("/workspaces/fitbot/.env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-pro")

# safety check
if not GOOGLE_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env. Add your Gemini API key as GOOGLE_API_KEY.")
    st.stop()

# LangChain Google imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Streamlit UI
st.set_page_config(page_title="FitBot (Gemini) - Week 2", page_icon="💪")
st.title("FitBot — Week 2 (Gemini + FAISS RAG)")
st.write("Using Google Gemini (Generative AI) for LLM and embeddings.")

# Load KB
kb_path = "data.txt"   # make sure this exists
if not os.path.exists(kb_path):
    st.error(f"Knowledge base missing: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    text = f.read()

# split documents
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([text])

# Create embeddings (Google Generative AI embeddings)
# We explicitly pass google_api_key so it works inside Codespaces
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GOOGLE_KEY
)

# Build FAISS (stores embeddings for retrieval)
vectorstore = FAISS.from_documents(docs, embeddings)

# Create Gemini chat LLM (LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=GOOGLE_KEY
)

# Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    chain_type="stuff"
)

# Chat UI
query = st.text_input("Ask FitBot a question about workouts, diet, or fitness:")

if query:
    with st.spinner("Thinking with Gemini..."):
        answer = qa.run(query)
    st.success(answer)
else:
    st.info("Type a question above to try a Gemini-powered RAG answer.")
