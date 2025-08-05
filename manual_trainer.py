import os
import pickle
from dotenv import load_dotenv
import streamlit as st

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Config
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = "kb_store"
MANUAL_FILE = os.path.join(VECTOR_DIR, "manual_entries.pkl")

# Streamlit UI
st.set_page_config(page_title="🧠 Manual Knowledge Trainer", layout="wide")
st.title("🧠 Train Knowledge Base Manually")

# Load manual entries if available
if os.path.exists(MANUAL_FILE):
    with open(MANUAL_FILE, "rb") as f:
        manual_entries = pickle.load(f)
else:
    manual_entries = []

# --- Add New Manual Input ---
st.subheader("✍️ Add New Knowledge")
text_input = st.text_area("Type or paste information you want to teach the system:", height=200)

if st.button("Add to Knowledge Base"):
    if not openai_api_key:
        st.error("API key not found in .env file.")
    elif not text_input.strip():
        st.warning("Please enter some text before submitting.")
    else:
        with st.spinner("🔄 Updating knowledge base..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = splitter.split_text(text_input.strip())

            new_docs = []
            for chunk in chunks:
                entry_id = f"manual-{len(manual_entries)+1}-{hash(chunk)}"
                manual_entries.append({"id": entry_id, "text": chunk})
                new_docs.append(Document(page_content=chunk, metadata={"source": entry_id}))

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Load or create vectorstore
            if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
                vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
                vectordb.add_documents(new_docs)
            else:
                vectordb = FAISS.from_documents(new_docs, embedding=embeddings)

            vectordb.save_local(VECTOR_DIR)

            # Save entry tracking
            with open(MANUAL_FILE, "wb") as f:
                pickle.dump(manual_entries, f)

        st.success("✅ Manual input added to the knowledge base!")

# --- View & Delete Manual Entries ---
st.subheader("📋 Existing Manual Entries")

if not manual_entries:
    st.info("No manual entries yet.")
else:
    for i, entry in enumerate(manual_entries):
        with st.expander(f"Entry {i+1}"):
            st.code(entry['text'])
            if st.button(f"🗑️ Delete Entry {i+1}", key=f"delete_{i}"):
                manual_entries.pop(i)
                # Rebuild vector DB after deletion
                with st.spinner("Rebuilding vector store..."):
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    if manual_entries:
                        docs = [
                            Document(page_content=e["text"], metadata={"source": e["id"]})
                            for e in manual_entries
                        ]
                        vectordb = FAISS.from_documents(docs, embedding=embeddings)
                        vectordb.save_local(VECTOR_DIR)
                    else:
                        # Delete existing vectorstore files
                        for f in ["index.faiss", "index.pkl"]:
                            try:
                                os.remove(os.path.join(VECTOR_DIR, f))
                            except FileNotFoundError:
                                pass

                    # Save updated manual entries
                    with open(MANUAL_FILE, "wb") as f:
                        pickle.dump(manual_entries, f)

                st.success("✅ Entry deleted and vector store updated.")
                st.experimental_rerun()
