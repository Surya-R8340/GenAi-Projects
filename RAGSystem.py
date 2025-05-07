import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "tinyllama"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128




class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def process_documents(self, pdf_paths):
        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            docs.extend(chunks)
        return docs


class RAGSystem:
    def __init__(self, model_name="tinyllama"):
        self.processor = DocumentProcessor()
        self.vectorstore = None
        self.llm = Ollama(model=model_name)
        self.setup_chain()


    def setup_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "Answer using this context:\n{context}\n\nQuestion: {question}"
        )
        self.chain = (
                {"context": self.retrieve, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def retrieve(self, query):
        if not self.vectorstore:
            return "No documents loaded"
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join(d.page_content for d in docs)

    def ingest(self, pdf_paths):
        docs = self.processor.process_documents(pdf_paths)
        self.vectorstore = FAISS.from_documents(
            docs,
            self.processor.embedding_model
        )


def main():
    st.title("RAG System with Selectable Models")

    # Step 1: Let user select the LLM model
    model_choice = st.selectbox(
        "Choose a model:",
        options=["tinyllama", "mistral", "2llama", "gemma3:1b", "nomic-embed-text:latest"]
    )

    # Step 2: Pass the selected model to RAG system
    rag = RAGSystem(model_name=model_choice)

    # File upload and ingestion
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = []
            for file in uploaded_files:
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(path)

            with st.spinner("Processing..."):
                rag.ingest(file_paths)
            st.success("Documents loaded!")

    # Query input
    if query := st.text_input("Ask a question"):
        with st.spinner("Thinking..."):
            answer = rag.chain.invoke(query)
        st.write(answer)


if __name__ == "__main__":
    main()
