import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tracker import log_document_upload
from theme import apply_theme

st.set_page_config(page_title="Ingestion", page_icon="📄", layout="wide")
apply_theme()

st.title("📄 Document Ingestion Pipeline")
st.markdown("### Upload documents or paste a URL")

client = chromadb.PersistentClient(path="./chroma_db")


def safe_collection_name(name):
    """Consistent, safe collection name from any source string."""
    base = name.split("/")[0].split("?")[0].split(".")[0]
    base = base.lower().replace(" ", "_").replace("-", "_").replace(":", "_")
    base = "".join(c for c in base if c.isalnum() or c == "_")
    return base[:60] or "unnamed"


# ============================================
# FILE UPLOAD
# ============================================
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    file_path = os.path.join(".", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document"):
        with st.spinner("Reading, chunking, and embedding..."):
            documents = []

            try:
                if uploaded_file.name.endswith(".pdf"):
                    from pypdf import PdfReader
                    try:
                        reader = PdfReader(file_path)
                        _ = reader.pages[0]
                    except Exception:
                        st.error("🔒 This PDF is encrypted or unreadable. Please upload an unlocked PDF.")
                        st.stop()

                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    for doc in documents:
                        page_num = doc.metadata.get("page", 0)
                        doc.metadata["location"] = f"Page {page_num + 1}"

                elif uploaded_file.name.endswith(".txt"):
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()
                    for i, doc in enumerate(documents):
                        doc.metadata["location"] = f"Section {i + 1}"

                elif uploaded_file.name.endswith(".docx"):
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                    for i, doc in enumerate(documents):
                        doc.metadata["location"] = f"Section {i + 1}"

                elif uploaded_file.name.endswith(".csv"):
                    import csv
                    from langchain_core.documents import Document as LCDoc
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for i, row in enumerate(reader):
                            row_text = " | ".join([f"{k}: {v.strip()}" for k, v in row.items() if k and v])
                            if len(row_text.strip()) < 10:
                                continue
                            documents.append(LCDoc(
                                page_content=row_text,
                                metadata={"location": f"Row {i + 2}"}
                            ))

                else:
                    st.error("Unsupported file type.")
                    st.stop()

            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")
                st.stop()

            if not documents:
                st.error("❌ No content could be extracted from this file.")
                st.stop()

            # Chunking (small for CSV, larger for others)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            if uploaded_file.name.endswith(".csv"):
                chunks = documents  # one row = one chunk
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                chunks = splitter.split_documents(documents)

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            collection_name = safe_collection_name(uploaded_file.name)

            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.get_or_create_collection(name=collection_name)

            texts = [c.page_content for c in chunks]
            metadatas = [
                {"source": uploaded_file.name, "location": c.metadata.get("location", "Section")}
                for c in chunks
            ]
            ids = [f"chunk_{i}" for i in range(len(texts))]
            embeddings = embedder.encode(texts).tolist()

            collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            log_document_upload(uploaded_file.name, len(chunks), "file")

            st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
            st.markdown(f"**Collection name:** `{collection_name}`")


# ============================================
# URL INGESTION
# ============================================
st.markdown("---")
st.markdown("### 🌐 Or Paste a URL")

url_input = st.text_input("Paste a live URL:", placeholder="https://en.wikipedia.org/wiki/...")

if url_input and st.button("Process URL"):
    with st.spinner("Scraping and processing..."):
        import requests
        from bs4 import BeautifulSoup

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url_input, timeout=20, headers=headers, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            # PERMANENT CHECK: reject empty / JS-only / paywalled pages
            if len(clean_text.strip()) < 800:
                st.error("🚨 **This URL could not be scraped properly.**")
                st.markdown(f"""
**Reason:** The page returned only **{len(clean_text.strip())} characters** of text.

**This usually means:**
- The page needs JavaScript to render (MSN, CNN, React sites)
- The page requires login or a subscription
- The page is behind a paywall or bot protection
- The URL redirects to a loading screen

**Try instead:**
- Wikipedia
- Government sites (.gov, .edu)
- News articles with server-rendered text (BBC, Reuters, AP)
- Any static HTML page
""")
                st.stop()

            # Split into chunks
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            from langchain_core.documents import Document as LCDoc
            documents = [LCDoc(page_content=clean_text, metadata={})]
            chunks = splitter.split_documents(documents)

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            collection_name = "url_" + safe_collection_name(url_input.split("//")[-1].split("/")[0])

            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.get_or_create_collection(name=collection_name)

            texts = [c.page_content for c in chunks]
            metadatas = [
                {"source": url_input, "location": f"Section {i + 1}"}
                for i, _ in enumerate(chunks)
            ]
            ids = [f"url_chunk_{i}" for i in range(len(texts))]
            embeddings = embedder.encode(texts).tolist()

            collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            log_document_upload(url_input, len(chunks), "url")

            st.success(f"✅ Processed {len(chunks)} chunks from URL")
            st.markdown(f"**Collection name:** `{collection_name}`")

        except requests.exceptions.Timeout:
            st.error("🚨 **URL request timed out.** The site took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 **Could not reach this URL.** The domain may not exist or may be blocking scrapers.")
        except requests.exceptions.HTTPError as e:
            st.error(f"🚨 **Server returned an error:** {e}")
        except Exception as e:
            st.error(f"🚨 **Scraping failed:** {e}")


# ============================================
# EXISTING COLLECTIONS
# ============================================
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