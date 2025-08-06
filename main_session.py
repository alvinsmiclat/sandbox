import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# --- NEW: Import necessary components for conversational memory ---
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# --- END NEW ---
from langchain.prompts import PromptTemplate

# --- 1. CORE LOGIC (Unchanged) ---

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = "kb_store"

# Fuzzy date handling
def parse_date_keywords(query):
    today = datetime.today()
    if "next week" in query.lower():
        next_week = today + timedelta(days=7)
        query += f" (around {next_week.strftime('%B %d')})"
    elif "this week" in query.lower():
        query += f" (around {today.strftime('%B %d')})"
    elif "next month" in query.lower():
        next_month = today + timedelta(days=30)
        query += f" (around {next_month.strftime('%B')})"
    return query

# --- 2. STREAMLIT UI/UX CONFIGURATION (Unchanged) ---

st.set_page_config(page_title="PDF Q&A Chat", page_icon="📚", layout="centered")

# Custom styling for a professional look
st.markdown("""
    <style>
        /* General body styling */
        body {
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background-color: #f0f2f6;
        }
        /* Main container styling */
        .block-container {
            padding: 2rem 1rem;
            max-width: 800px;
            margin: auto;
        }
        /* Header styling */
        .header {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5rem;
            color: #1e3a8a; /* A deep blue color */
            margin: 0;
        }
        .header p {
            font-size: 1.1rem;
            color: #555;
            margin-top: 0.5rem;
        }
        /* Chat message styling */
        .stChatMessage {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        .stChatMessage[data-testid="stChatMessageContent"] {
             font-size: 1.05rem;
        }
        /* Style for AI messages */
        [data-testid="chat-message-container"]:has([data-testid="chat-avatar-assistant"]) .stChatMessage {
             background-color: #e3f2fd; /* Light blue background for AI */
        }
        /* Input bar styling */
        .stChatInputContainer {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 -4px 8px rgba(0,0,0,0.1);
            border: none;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- 3. APPLICATION LOGIC AND UI FLOW (Updated for Memory) ---

# Header Section
with st.container():
    st.markdown('<div class="header"><h1>📚 Bina Bangsa School SandBox V1.0</h1><p>Chat with BBS Documents — powered by AI, Made by AB</p></div>', unsafe_allow_html=True)

# Sidebar with Clear Chat Button
st.sidebar.title("Controls")
if st.sidebar.button("Clear Chat History", use_container_width=True):
    # Reset the chat history and memory
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.rerun()

# Main App Logic: Check for API key and vector store
if not openai_api_key:
    st.error("🚫 API key not found. Please add it to your .env file (OPENAI_API_KEY=sk-...).")
elif not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    st.error("🚫 Knowledge base not found. Please run the `builder.py` script first to process your PDFs.")
else:
    # --- NEW: Caching the retriever for efficiency ---
    @st.cache_resource
    def get_retriever():
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # --- NEW: Defining the prompt for the final answering step ---
    # This prompt ensures the AI gives answers based on the retrieved documents.
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a highly knowledgeable and friendly assistant. Use the following context extracted from the user's PDF documents to answer the question accurately and clearly. If you don't know the answer from the context, say that you cannot find the information in the documents.

Context:
{context}

Question: {question}
Answer:"""
    )

    # Initialize session state for messages and memory
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
    if "memory" not in st.session_state:
        # The memory object stores the conversation history
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- NEW: Create the conversational chain ---
    # This chain has memory and uses the retriever to find relevant info.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.1),
        retriever=get_retriever(),
        memory=st.session_state.memory, # Link the chain to the session's memory
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about our BBS documents..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query using the conversational chain
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                enriched_query = parse_date_keywords(prompt)
                # The chain is called with the query. It automatically uses memory.
                result = qa_chain({"question": enriched_query})
                response = result["answer"]
                st.markdown(response)

        # Add AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})