import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tracker import log_document_upload
from theme import apply_theme

apply_theme()

st.set_page_config(page_title="Ingestion", page_icon="📄", layout="wide")

st.title("📄 Document Ingestion Pipeline")
st.markdown("### Upload documents or paste a URL")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# File Upload
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=False
)

if uploaded_file:
    file_path = os.path.join(".", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document"):
        with st.spinner("Reading, chunking, and embedding..."):
            if uploaded_file.name.endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif uploaded_file.name.endswith(".txt"):
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            elif uploaded_file.name.endswith(".docx"):
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif uploaded_file.name.endswith(".csv"):
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path)
                documents = loader.load()
            else:
                st.error("Unsupported file type.")
                st.stop()

            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from settings import load_settings as load_user_settings
            user_settings = load_user_settings()
            chunk_size_value = user_settings.get("chunk_size", 1000)
            chunk_overlap_value = user_settings.get("chunk_overlap", 200)

            text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size_value,
                    chunk_overlap=chunk_overlap_value,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
            chunks = text_splitter.split_documents(documents)

            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Create a safe collection name from filename
            collection_name = uploaded_file.name.replace(".", "_").replace(" ", "_").replace("-", "_")

            # Delete old collection if exists, then create new
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.get_or_create_collection(name=collection_name)

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{"source": uploaded_file.name, "page": chunk.metadata.get("page", 0)} for chunk in chunks]
            ids = [f"chunk_{i}" for i in range(len(texts))]
            embeddings = embedder.encode(texts).tolist()

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            log_document_upload(uploaded_file.name, len(chunks), "file")

            st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
            st.markdown(f"**Collection Name:** `{collection_name}`")

# URL Ingestion
st.markdown("---")
st.markdown("### 🌐 Or Paste a URL")

url_input = st.text_input("Paste a live URL:", placeholder="https://en.wikipedia.org/wiki/...")

if url_input and st.button("Process URL"):
    with st.spinner("Scraping and processing..."):
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(url_input, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            from langchain_core.documents import Document
            documents = [Document(page_content=clean_text, metadata={"source": url_input, "page": 0})]
            chunks = text_splitter.split_documents(documents)

            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Create collection name from URL
            collection_name = url_input.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")[:50]

            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.get_or_create_collection(name=collection_name)

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{"source": url_input, "page": 0} for _ in chunks]
            ids = [f"url_chunk_{i}" for i in range(len(texts))]
            embeddings = embedder.encode(texts).tolist()

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            log_document_upload(url_input, len(chunks), "url")

            st.success(f"✅ Processed {len(chunks)} chunks from URL")
            st.markdown(f"**Collection Name:** `{collection_name}`")

        except Exception as e:
                st.error("🚨 **URL Verification Failed**")
                st.markdown(f"""
                **Reason:** The URL could not be reached.

                **What this means:**
                - The domain may not exist
                - The website may be offline
                - The URL could be a phishing attempt
                - The domain may be blocked

                **Details:** `{url_input}`

                **Recommendation:** Do NOT trust this URL. Verify it manually before using.
                """)

# Show existing collections
st.markdown("---")
st.markdown("### 📚 Documents in Database")

try:
    collections = client.list_collections()
    if collections:
        for col in collections:
            st.markdown(f"- 📄 `{col.name}`")
    else:
        st.info("No documents uploaded yet.")
except Exception:
    st.info("No documents uploaded yet.")