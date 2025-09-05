# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT_ID")

if not OPENAI_KEY:
    st.error("❌ No API key found. Please set OPENAI_API_KEY in .env")
    st.stop()

# ---- Configure OpenAI client (project-aware) ----
import openai
openai.api_key = OPENAI_KEY
if OPENAI_ORG:
    openai.organization = OPENAI_ORG
if OPENAI_PROJECT:
    openai.project = OPENAI_PROJECT

# ---- LangChain imports ----
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ---- UI ----
st.set_page_config(page_title="FitBot - Week 2", page_icon="💪")
st.title("💪 FitBot — Week 2 Progress")
st.write("This version uses **OpenAI Project Key + FAISS (RAG)**")

# ---- Load knowledge base ----
kb_path = "data.txt"
if not os.path.exists(kb_path):
    st.error(f"Knowledge base file missing: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    fitness_text = f.read()

# ---- Split into chunks ----
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([fitness_text])

# ---- Embeddings + FAISS ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_KEY,
    organization=OPENAI_ORG
)

vectorstore = FAISS.from_documents(docs, embeddings)

# ---- LLM ----
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_KEY,
    organization=OPENAI_ORG
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# ---- Chat ----
query = st.text_input("Ask FitBot a fitness question:")

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
    st.success(answer)
else:
    st.info("Type a fitness question above to try FitBot 🚀")
