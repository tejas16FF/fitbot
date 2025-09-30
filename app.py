import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# --- NEW IMPORTS FOR MEMORY AND CONVERSATIONAL RAG ---
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv("/workspaces/fitbot/.env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# Using a model suitable for chat history
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-pro") 

# -----------------------------
# Helper Functions
# -----------------------------
def load_data(filepath="data.txt"):
    """Load knowledge base from text file"""
    if not os.path.exists(filepath):
        # Professional error handling for academic review
        st.error(f"❌ Knowledge base file not found: {filepath}. Please ensure 'data.txt' is in the root directory.")
        st.stop()
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def build_vectorstore(text):
    """Create FAISS vector store from text"""
    # Splitting text into manageable, overlapping chunks for effective retrieval (RAG)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    
    # Using HuggingFace embeddings as decided during project development
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Creating the FAISS index for fast similarity search
    return FAISS.from_documents(docs, embeddings)

def create_qa_chain(vectorstore):
    """Setup Gemini-powered Conversational QA chain with professional tone and memory"""
    if not GOOGLE_KEY:
        st.error("❌ GOOGLE_API_KEY not found in .env. Please add your Gemini API key.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_KEY,
        temperature=0.7 # Balancing creativity (fluency) with factuality (RAG context)
    )
    
    # --- 1. CONVERSATIONAL MEMORY SETUP ---
    # Stores conversation history to maintain context
    memory = ConversationBufferMemory(
        memory_key="chat_history", # Variable name used in the prompt template
        return_messages=True, 
        output_key='answer' # Ensures the output is saved correctly for the chain
    )

    # --- 2. UPDATED PROMPT TEMPLATE ---
    # Now includes {chat_history} variable
    template = """
    You are FitBot, a professional AI fitness coach. 
    Always provide clear, polite, supportive, and detailed answers 
    about workouts, nutrition, recovery, health, and motivation.
    Never mention documents, context, or knowledge base. 
    If exact details are missing in the provided context, combine the history and context 
    to provide the best possible professional advice instead.
    Keep your tone supportive and encouraging.

    Chat History:
    {chat_history}
    
    Context: {context}
    
    Question: {question}
    
    Professional Answer:
    """
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["chat_history", "context", "question"]
    )

    # --- 3. CONVERSATIONAL RETRIEVAL CHAIN ---
    # This chain automatically manages memory and retrieval
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
        # Pass the customized prompt to the chain that combines documents and history
        combine_docs_chain_kwargs={"prompt": prompt}, 
        return_source_documents=False # Hide source documents for a cleaner, professional UX
    )

def answer_query(qa_chain, query):
    """Generate answer for user query"""
    # The ConversationalRetrievalChain expects the query key 'question'
    return qa_chain.run({"question": query})

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FitBot", page_icon="💪", layout="centered")

st.title("💪 FitBot - Your AI Fitness Assistant")
st.markdown(
    """
    Welcome to **FitBot** 👋. **I remember our conversation!** Ask me about **workouts, diet, recovery, or motivation** and I’ll give you 
    professional, supportive advice based on our past chat.
    """
)

# --- INITIALIZE CHAT HISTORY IN SESSION STATE ---
# Streamlit's mechanism to store data across script reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am FitBot, your AI fitness coach. How can I support your fitness goals today?"}
    ]

# -----------------------------
# Main Logic (Initialization)
# -----------------------------
# Initialize RAG components only once
if 'qa_chain' not in st.session_state:
    text_data = load_data("data.txt")
    vectorstore = build_vectorstore(text_data)
    st.session_state['qa_chain'] = create_qa_chain(vectorstore)

qa_chain = st.session_state['qa_chain']

# --- DISPLAY ALL PAST MESSAGES ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- HANDLE USER INPUT AND RESPONSE ---
query = st.chat_input("Enter your fitness question:")

if query:
    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking with Gemini..."):
            # Pass the query to the C-RAG chain
            answer = answer_query(qa_chain, query)
            
            # Store the new assistant message in session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

# End of File Generation
