import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UI setup
st.set_page_config(page_title="FitBot - Week 2", page_icon="💪")
st.title("💪 FitBot - AI Powered Fitness Chatbot (Week 2 Progress)")
st.write("Now using OpenAI + FAISS (RAG pipeline)!")

# Initialize OpenAI models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Load fitness knowledge base
with open("data.txt", "r") as f:
    fitness_text = f.read()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.create_documents([fitness_text])

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Create QA chain (RAG pipeline)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Chat input
user_input = st.text_input("Ask me about workouts, diet, or fitness tips:")

if user_input:
    response = qa.run(user_input)
    st.success(response)
else:
    st.info("Enter a question above to get a personalized fitness response.")
