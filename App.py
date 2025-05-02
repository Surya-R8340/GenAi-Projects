import os
import tempfile
from datetime import time

import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import requests
from urllib3.exceptions import ReadTimeoutError
from huggingface_hub import InferenceClient

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_OPTIONS = {
    "TinyLlama 1.1B (Fast)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Mistral 7B (Better)": "mistralai/Mistral-7B-Instruct-v0.1",
    "Llama 3 8B (Best)": "meta-llama/Meta-Llama-3-8B-Instruct"
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
NUM_TOPICS = 3
RETRIEVE_K = 3


# -------------------------------------------------------------------
# Custom URL Setup (Run this FIRST in terminal before starting app)
# -------------------------------------------------------------------
# streamlit run app.py --server.port=1234 --server.address=0.0.0.0
# Then access via http://your-machine-name:1234

# -------------------------------------------------------------------
# Document Processor
# -------------------------------------------------------------------
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

    def load_and_split(self, pdf_paths: List[str]) -> List[Dict]:
        docs = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                docs.extend({
                                "text": chunk.page_content,
                                "metadata": chunk.metadata,
                                "source": os.path.basename(path),
                                "embedding": self.embedder.encode(chunk.page_content)
                            } for chunk in chunks)
            except Exception as e:
                st.warning(f"Skipped {path}: {str(e)}")
        return docs

    def generate_topics(self, docs: List[Dict]) -> Tuple[List[str], List[str]]:
        texts = [doc["text"] for doc in docs]
        if len(texts) < 2:
            return ["General"] * len(texts), ["General Topic"]

        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(texts)
        n_topics = min(NUM_TOPICS, len(texts) // 2 or 1)

        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        kmeans.fit(X)

        terms = vectorizer.get_feature_names_out()
        topics = [
            ", ".join(terms[ind] for ind in kmeans.cluster_centers_[i].argsort()[-3:])
            for i in range(n_topics)
        ]
        return [topics[label] for label in kmeans.labels_], topics


# -------------------------------------------------------------------
# RAG System
# -------------------------------------------------------------------
class RAGSystem:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.docs = []

    def ingest(self, pdf_paths: List[str]):
        self.docs = self.processor.load_and_split(pdf_paths)
        doc_topics, all_topics = self.processor.generate_topics(self.docs)
        for doc, topic in zip(self.docs, doc_topics):
            doc["metadata"]["topic"] = topic
        return all_topics

    def retrieve(self, query: str) -> List[str]:
        query_embed = self.processor.embedder.encode(query)
        scores = [
            (doc, np.dot(query_embed, doc["embedding"]))
            for doc in self.docs
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc[0]["text"] for doc in scores[:RETRIEVE_K]]

    def query(self, question: str, model_name: str, hf_token: str) -> str:
        context = "\n\n".join(self.retrieve(question))
        prompt = f"""Answer using this context:
        {context}

        Question: {question}
        Answer (2-3 sentences):"""

        client = InferenceClient(token=hf_token)
        return client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    # Custom page config
    st.set_page_config(
        page_title="Multi-Document RAG Model for Q&A and Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("Multi-Document RAG Model for Q&A and Chat")
    st.markdown("""
    <style>
    .stApp { max-width: 1000px; margin: 0 auto; }
    .model-selector { background: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # Model selection sidebar
    with st.sidebar:
        st.header("Configuration")
        hf_token = st.text_input("Hugging Face Token", type="password")
        selected_model = st.selectbox(
            "Choose LLM Model",
            list(MODEL_OPTIONS.keys()),
            index=0
        )
        st.caption(f"Selected: {MODEL_OPTIONS[selected_model]}")

    rag = RAGSystem()

    # Document upload
    with st.container():
        st.subheader("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload research PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for file in uploaded_files:
                    path = os.path.join(tmpdir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    paths.append(path)

                with st.spinner("Analyzing documents..."):
                    topics = rag.ingest(paths)
                    st.success(f"Processed {len(uploaded_files)} files!")

                    with st.expander("📊 Detected Topics"):
                        for i, topic in enumerate(topics):
                            st.write(f"{i + 1}. {topic}")

    # Question answering
    with st.container():
        st.subheader("💬 Ask Questions")
        question = st.text_input("Enter your research question")

        if question and hf_token:
            with st.spinner("Generating answer..."):
                try:
                    answer = rag.query(
                        question,
                        MODEL_OPTIONS[selected_model],
                        hf_token
                    )
                    st.subheader("📝 Answer")
                    st.write(answer)

                    with st.expander("🔍 See sources"):
                        for i, context in enumerate(rag.retrieve(question)):
                            st.markdown(f"**Source {i + 1}**")
                            st.caption(context[:500] + "...")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
