import csv
import os

import chromadb
import streamlit as st

from tracker import log_document_upload
from theme import apply_theme, render_navigation, render_page_header, render_top_bar


st.set_page_config(page_title="Ingestion", page_icon="▣", layout="wide")
apply_theme()
render_navigation("pages/2_Ingestion.py")
render_top_bar("Document Ingestion Pipeline")

client = chromadb.PersistentClient(path="./chroma_db")


def safe_collection_name(name):
    base = name.split("/")[0].split("?")[0].split(".")[0]
    base = base.lower().replace(" ", "_").replace("-", "_").replace(":", "_")
    base = "".join(char for char in base if char.isalnum() or char == "_")
    return base[:60] or "unnamed"


def get_queue():
    return st.session_state.setdefault("ingestion_queue", {})


def add_uploaded_files(files):
    queue = get_queue()
    for uploaded_file in files:
        if uploaded_file.name in queue:
            continue
        file_path = os.path.join(".", uploaded_file.name)
        with open(file_path, "wb") as output_file:
            output_file.write(uploaded_file.getbuffer())
        queue[uploaded_file.name] = {
            "name": uploaded_file.name,
            "path": file_path,
            "size": uploaded_file.size,
            "progress": 0,
            "chunks": "---",
            "status": "Queued",
            "selected": True,
            "source_type": "file",
        }


def extract_file_documents(file_path, filename):
    documents = []
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        from pypdf import PdfReader
        from langchain_community.document_loaders import PyPDFLoader

        reader = PdfReader(file_path)
        _ = reader.pages[0]
        documents = PyPDFLoader(file_path).load()
        for document in documents:
            document.metadata["location"] = f"Page {document.metadata.get('page', 0) + 1}"
    elif lower_name.endswith(".txt"):
        from langchain_community.document_loaders import TextLoader

        documents = TextLoader(file_path, encoding="utf-8").load()
        for index, document in enumerate(documents):
            document.metadata["location"] = f"Section {index + 1}"
    elif lower_name.endswith(".docx"):
        from langchain_community.document_loaders import Docx2txtLoader

        documents = Docx2txtLoader(file_path).load()
        for index, document in enumerate(documents):
            document.metadata["location"] = f"Section {index + 1}"
    elif lower_name.endswith(".csv"):
        from langchain_core.documents import Document as LCDoc

        with open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
            for index, row in enumerate(csv.DictReader(input_file)):
                row_text = " | ".join(
                    f"{key}: {value.strip()}" for key, value in row.items() if key and value
                )
                if len(row_text.strip()) >= 10:
                    documents.append(LCDoc(page_content=row_text, metadata={"location": f"Row {index + 2}"}))
    return documents


def process_file(entry):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    documents = extract_file_documents(entry["path"], entry["name"])
    if not documents:
        raise ValueError("No content could be extracted from this file.")
    if entry["name"].lower().endswith(".csv"):
        chunks = documents
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)

    collection_name = safe_collection_name(entry["name"])
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [
        {"source": entry["name"], "location": chunk.metadata.get("location", "Section")}
        for chunk in chunks
    ]
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(texts).tolist()
    collection.add(
        ids=[f"chunk_{index}" for index in range(len(texts))],
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    log_document_upload(entry["name"], len(chunks), "file")
    return collection_name, len(chunks)


def process_url(url):
    import requests
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document as LCDoc
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    clean_text = "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())
    if len(clean_text) < 800:
        raise ValueError(f"The URL returned only {len(clean_text)} characters of readable text.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    ).split_documents([LCDoc(page_content=clean_text, metadata={})])
    collection_name = "url_" + safe_collection_name(url.split("//")[-1].split("/")[0])
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)
    texts = [chunk.page_content for chunk in chunks]
    collection.add(
        ids=[f"url_chunk_{index}" for index in range(len(texts))],
        documents=texts,
        metadatas=[{"source": url, "location": f"Section {index + 1}"} for index in range(len(texts))],
        embeddings=SentenceTransformer("all-MiniLM-L6-v2").encode(texts).tolist(),
    )
    log_document_upload(url, len(chunks), "url")
    return collection_name, len(chunks)


render_page_header(
    "Document ingestion",
    "Workspace / pipeline manager",
    "Prepare a research corpus and make its provenance searchable.",
)

st.markdown("### Upload Documents")
upload_header, ocr_column = st.columns([3, 1])
with upload_header:
    st.caption("Supported formats: PDF, DOCX, TXT, CSV • Up to 200MB per file")
with ocr_column:
    advanced_ocr = st.toggle("Advanced OCR", value=True, help="Reserved for OCR-enabled document extraction.")

uploaded_files = st.file_uploader(
    "Click to browse or drag and drop files here",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
    help="Files are staged in the pipeline queue before processing.",
)
if uploaded_files:
    add_uploaded_files(uploaded_files)
    st.success(f"{len(uploaded_files)} file(s) staged for processing.")

st.caption("Intelligent routing categorizes documents before chunking and embedding.")

queue = get_queue()
st.markdown("### Active Pipeline Queue")
queue_actions_left, queue_actions_right = st.columns([2, 1])
with queue_actions_left:
    if st.button("Pause All", use_container_width=True):
        st.session_state.ingestion_paused = not st.session_state.get("ingestion_paused", False)
with queue_actions_right:
    process_selected = st.button("Process Selected", type="primary", use_container_width=True)

if process_selected and not st.session_state.get("ingestion_paused", False):
    for entry in queue.values():
        if not entry["selected"] or entry["status"] == "Completed":
            continue
        entry["status"] = "Processing"
        entry["progress"] = 45
        try:
            collection_name, chunk_count = process_file(entry)
            entry["progress"] = 100
            entry["chunks"] = chunk_count
            entry["status"] = "Completed"
            entry["collection"] = collection_name
        except Exception as error:
            entry["status"] = "Failed"
            entry["progress"] = 0
            entry["error"] = str(error)
elif process_selected:
    st.warning("Pipeline is paused. Resume it before processing selected documents.")

if queue:
    for name, entry in queue.items():
        row_left, row_name, row_progress, row_status, row_action = st.columns([0.35, 2.2, 1.4, 1.1, 0.7])
        with row_left:
            entry["selected"] = st.checkbox("Select", value=entry["selected"], key=f"select_{name}", label_visibility="collapsed")
        with row_name:
            st.markdown(f"**{name}**")
            st.caption(f"{entry['size'] / 1024 / 1024:.2f} MB")
        with row_progress:
            st.progress(entry["progress"] / 100, text=f"{entry['progress']}%")
        with row_status:
            st.caption(entry["status"])
        with row_action:
            if st.button("×", key=f"remove_{name}", help="Remove from queue"):
                del queue[name]
                st.rerun()
        if entry.get("error"):
            st.error(entry["error"])
    st.caption(f"Showing 1-{len(queue)} of {len(queue)} active processes")
else:
    st.info("No files in the pipeline queue. Add documents above to begin.")

st.markdown("### Add a Live Source")
url_column, url_button_column = st.columns([4, 1])
with url_column:
    url_input = st.text_input("Paste a live URL", placeholder="https://example.org/research", label_visibility="collapsed")
with url_button_column:
    process_url_button = st.button("Process URL", use_container_width=True)
if process_url_button and url_input:
    with st.spinner("Scraping and embedding source..."):
        try:
            collection_name, chunk_count = process_url(url_input)
            st.success(f"Processed {chunk_count} chunks into `{collection_name}`.")
        except Exception as error:
            st.error(f"URL ingestion failed: {error}")

st.markdown("### ChromaDB Store")
try:
    collection_names = [collection.name for collection in client.list_collections()]
except Exception:
    collection_names = []

if collection_names:
    explorer_column, details_column = st.columns([1, 2], gap="large")
    with explorer_column:
        selected_collection = st.selectbox("Document collection", collection_names)
        if st.button("🗑️ Delete this collection", key="delete_collection_btn"):
            try:
                client.delete_collection(name=selected_collection)
                st.success(f"Deleted '{selected_collection}'. Refreshing...")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete: {e}")
        vector_query = st.text_input("Search vectors by metadata", placeholder="Search chunks...")
        collection = client.get_collection(selected_collection)
        vector_data = collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = vector_data.get("embeddings")
        if embeddings is None:
            embeddings = [[] for _ in vector_data.get("documents", [])]
        vector_rows = list(zip(
            vector_data.get("documents", []),
            vector_data.get("metadatas", []),
            embeddings,
        ))
        if vector_query:
            vector_rows = [row for row in vector_rows if vector_query.lower() in row[0].lower()]
        if vector_rows:
            selected_index = st.radio(
                "Chunks",
                list(range(len(vector_rows))),
                format_func=lambda index: f"CHK-{index:04d}  {vector_rows[index][0][:70]}...",
                label_visibility="collapsed",
            )
        else:
            selected_index = None
            st.info("No matching chunks.")

    with details_column:
        if selected_index is not None:
            selected_text, selected_metadata, selected_embedding = vector_rows[selected_index]
            st.markdown(f"### CHK-{selected_index:04d} details")
            detail_column, vector_column = st.columns(2)
            with detail_column:
                st.markdown("#### Text Snippet")
                st.markdown(f'<div class="provenance-panel"><em>{selected_text[:1200]}</em></div>', unsafe_allow_html=True)
                st.markdown("#### Metadata")
                st.json(selected_metadata)
            with vector_column:
                st.markdown("#### Dense Vector Representation")
                st.code(
                    "[" + ", ".join(f"{value:.6f}" for value in selected_embedding[:32])
                    + (", ...]" if len(selected_embedding) > 32 else "]"),
                    language="text",
                )
                st.caption(f"Stored embedding dimension: {len(selected_embedding)}")
else:
    st.info("No ChromaDB collections yet. Process a document to explore its chunks.")
