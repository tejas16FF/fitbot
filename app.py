# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file (for OPENAI_API_KEY)
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Safety check
if not OPENAI_KEY:
    st.error("❌ OpenAI API key not found. Please set it in .env as OPENAI_API_KEY")
    st.stop()

# ---- LangChain & OpenAI imports ----
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ---- Streamlit UI ----
st.set_page_config(page_title="FitBot - Week 2", page_icon="💪")
st.title("💪 FitBot — Week 2 Progress")
st.write("This version uses **OpenAI + FAISS (RAG pipeline)** to give fitness advice.")

# ---- Initialize OpenAI models ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_KEY
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_KEY
)

# ---- Load fitness knowledge base ----
kb_path = "data.txt"
if not os.path.exists(kb_path):
    st.error(f"Knowledge base file missing: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    fitness_text = f.read()

# ---- Split text into chunks ----
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([fitness_text])

# ---- Create FAISS vector store ----
vectorstore = FAISS.from_documents(docs, embeddings)

# ---- Create QA chain (RAG pipeline) ----
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
