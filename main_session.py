import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = "kb_store"

# Fuzzy date handling
def parse_date_keywords(query):
    today = datetime.today()
    if "next week" in query.lower():
        query += f" (around {(today + timedelta(days=7)).strftime('%B %d')})"
    elif "this week" in query.lower():
        query += f" (around {today.strftime('%B %d')})"
    elif "next month" in query.lower():
        query += f" (around {(today + timedelta(days=30)).strftime('%B')})"
    return query

# Streamlit page setup
st.set_page_config(
    page_title="Bina Bangsa School Sandbox",
    page_icon="📘",  # or use favicon.png if you have one
    layout="centered"
)


from PIL import Image


st.markdown("## **Bina Bangsa School Sandbox**")
st.caption("BBS AI Application By Academic Board")


# Load knowledge base
if not openai_api_key:
    st.error("🚫 API key not found. Add it to your `.env` file.")
elif not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    st.error("🚫 No knowledge base found. Run `builder.py` first.")
else:
    with st.spinner("🔍 Loading knowledge base..."):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a highly knowledgeable assistant. Use the following context extracted from PDFs to answer the question accurately and clearly. Mention relevant chunks or sources when helpful.

Context:
{context}

Question: {question}
Answer:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt}
        )

        # Chat session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Clear history button
        with st.sidebar:
            # Center logo using st.image inside a container
            st.markdown(
                """
                <style>
                    .centered-logo {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        margin-top: 1rem;
                        margin-bottom: 1rem;
                    }
                    .centered-logo h3 {
                        margin: 0.5rem 0 0.2rem 0;
                        font-size: 1.2rem;
                        text-align: center;
                    }
                    .centered-logo p {
                        font-size: 0.85rem;
                        color: gray;
                        text-align: center;
                        margin: 0;
                    }
                </style>
                <div class="centered-logo">
                """,
                unsafe_allow_html=True
            )

            st.image("logo.png", width=100)

            st.markdown(
                """
                    <h3>BBS SandBox</h3>
                    <p>AI-powered Q&A</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown("## 🧹 Options")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


        # User input form
        with st.form(key="query_form", clear_on_submit=True):
            query = st.text_input("💬 Ask something:", placeholder="Ask BBS Knowledge Base", label_visibility="collapsed")
            submitted = st.form_submit_button("Ask")

        if submitted and query:
            enriched_query = parse_date_keywords(query)

            # Include limited recent history in prompt
            chat_context = ""
            for i in range(max(0, len(st.session_state.chat_history) - 6), len(st.session_state.chat_history), 2):
                u = st.session_state.chat_history[i][1]
                a = st.session_state.chat_history[i + 1][1] if i + 1 < len(st.session_state.chat_history) else ""
                chat_context += f"\nUser: {u}\nAI: {a}"

            full_query = f"{chat_context}\nUser: {query}\nAI:"

            with st.spinner("🤔 Thinking..."):
                response = qa_chain.run(full_query)

            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", response))

        # Display chat
        for role, msg in st.session_state.chat_history:
            klass = "user" if role == "user" else "bot"
            label = "You" if role == "user" else "AI"
            st.markdown(f'<div class="message {klass}"><strong>{label}:</strong><br>{msg}</div>', unsafe_allow_html=True)
