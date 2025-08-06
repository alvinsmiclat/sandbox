import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit as st

# --- Updated imports for latest LangChain ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = "kb_store"

# --- Fuzzy date handling ---
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

# --- Streamlit page config ---
st.set_page_config(page_title="PDF Q&A Chat", page_icon="📚", layout="centered")

# --- Styling ---
st.markdown("""
<style>
/* Styling omitted for brevity; keep your original CSS here */
</style>
""", unsafe_allow_html=True)

# --- Header ---
with st.container():
    st.markdown('<div class="header"><h1>📚 Bina Bangsa School SandBox V1.0</h1><p>Chat with BBS Documents — powered by AI, Made by AB</p></div>', unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.title("Controls")
if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.rerun()

# --- API Key & Vector Store checks ---
if not openai_api_key:
    st.error("🚫 API key not found. Please add it to your .env file (OPENAI_API_KEY=sk-...).")
elif not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    st.error("🚫 Knowledge base not found. Please run the `builder.py` script first to process your PDFs.")
else:
    # --- Cache retriever ---
    @st.cache_resource
    def get_retriever():
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # --- Custom prompt ---
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a highly knowledgeable and friendly assistant. Use the following context extracted from the user's PDF documents to answer the question accurately and clearly. If you don't know the answer from the context, say that you cannot find the information in the documents.

Context:
{context}

Question: {question}
Answer:"""
    )

    # --- Session state init ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- Conversational Retrieval Chain ---
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=openai_api_key),
        retriever=get_retriever(),
        memory=st.session_state.memory,
        qa_prompt=custom_prompt,
        return_source_documents=True
    )

    # --- Display chat history ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask a question about our BBS documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- AI response ---
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                enriched_query = parse_date_keywords(prompt)
                result = qa_chain({"question": enriched_query})

                # Handle both old & new LangChain return formats
                if "answer" in result:
                    response = result["answer"]
                elif "result" in result:
                    response = result["result"]
                else:
                    response = "I couldn't find an answer in the documents."

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
