# app.py
import os
from dotenv import load_dotenv
load_dotenv()  # load .env into environment

# Short debug print (optional) — remove after verifying
# print("DEBUG: OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))

import streamlit as st

# import langchain OpenAI wrappers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# get key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OpenAI API key not found. Add it to .env as OPENAI_API_KEY and restart.")
    st.stop()

# UI
st.set_page_config(page_title="FitBot - Week 2", page_icon="💪")
st.title("FitBot — Week 2 (RAG demo)")

# initialize embeddings and llm with explicit api key
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_KEY)

# load KB
kb_path = "data.txt"
if not os.path.exists(kb_path):
    st.error(f"Knowledge base missing: {kb_path}")
    st.stop()

with open(kb_path, "r", encoding="utf-8") as f:
    text = f.read()

# split and build FAISS
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([text])
vectorstore = FAISS.from_documents(docs, embeddings)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

query = st.text_input("Ask FitBot a fitness question:")
if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
    st.write(answer)
else:
    st.info("Enter a question above to try RAG-based answers.")
