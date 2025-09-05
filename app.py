# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file (contains API credentials)
load_dotenv()

# Read environment variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT_ID")

# ---- Safety checks ----
if not OPENAI_KEY:
    st.error("❌ No API key found. Please set OPENAI_API_KEY in .env")
    st.stop()

# ---- Configure OpenAI client (for project keys) ----
import openai
openai.api_key = OPENAI_KEY
if OPENAI_ORG:
    openai.organization = OPENAI_ORG
if OPENAI_PROJECT:
    openai.project = OPENAI_PROJECT

# ---- LangChain & OpenAI imports ----
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ---- Streamlit UI ----
st.set_page_config(page_title="FitBot - Week 2", page_icon="💪")
st.title("💪 FitBot — Week 2 Progress")
st.write("This version uses **OpenAI (Project Key) + FAISS (RAG pipeline)** to give fitness advice.")

# ---- Load knowledge base ----
kb_path = "data.txt"   # your file path
if not os.path.exists(kb_path):
    st.error(f"Knowledge base file missing: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    fitness_text = f.read()

# ---- Split text into chunks ----
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([fitness_text])

# ---- Create FAISS vector store ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_KEY
)
vectorstore = FAISS.from_documents(docs, embeddings)

# ---- Create QA chain (RAG pipeline) ----
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_KEY
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# ---- Chat interface ----
query = st.text_input("Ask FitBot a question about workouts, diet, or fitness:")

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
    st.success(answer)
else:
    st.info("Type a fitness question above to try FitBot 🚀")
