import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
#load_dotenv()

#For Local deployment
#openai_api_key = os.getenv("OPENAI_API_KEY")

#For streamlit deployment
openai_key_key = os.environ["OPENAI_API_KEY"]
VECTOR_DIR = "kb_store"

# Fuzzy date query handling
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

# Streamlit UI setup
st.set_page_config(page_title="PDF Q&A Reader", layout="wide")
st.title("📚 Ask Your PDF Knowledge Base")

# Load vector store
if not openai_api_key:
    st.error("🚫 API key not found. Add it to your .env file as OPENAI_API_KEY=sk-...")
elif not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    st.error("🚫 No knowledge base found. Run builder.py first to process your PDFs.")
else:
    with st.spinner("🔍 Loading knowledge base..."):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Prompt Template
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a highly knowledgeable assistant. Use the following context extracted from PDFs to answer the question accurately and clearly. Mention relevant chunks or sources when helpful.

Context:
{context}

Question: {question}
Answer:"""
        )

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt}
        )

        # User query input
        query = st.text_input("💬 Ask something about your PDF knowledge base:")
        if query:
            enriched_query = parse_date_keywords(query)
            with st.spinner("🤔 Thinking..."):
                response = qa_chain.run(enriched_query)
                st.success(response)
