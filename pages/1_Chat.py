import streamlit as st
import time
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from tracker import log_query
from conversations import (
    load_conversations,
    create_new_chat,
    save_message,
    delete_chat,
    get_all_chats_for_document
)
from theme import apply_theme

apply_theme()


st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

import os
from dotenv import load_dotenv

from evidence import run_evidence_verification
from retrieval import hybrid_retrieve

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get all collections
try:
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
except Exception:
    collection_names = []

# ============================================
# SIDEBAR - Document Selector + Chat History
# ============================================
with st.sidebar:
    st.markdown("## 📚 Documents")

    if collection_names:
        selected_doc = st.selectbox(
            "Choose document:",
            collection_names,
            key="selected_doc"
        )
    else:
        st.warning("No documents found. Upload first.")
        selected_doc = None

    st.markdown("---")

    st.markdown("## 💬 Conversations")

    if selected_doc:
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_id = create_new_chat(selected_doc)
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = []
            st.rerun()

        chat_ids = get_all_chats_for_document(selected_doc)
        if chat_ids:
            st.markdown(f"**{len(chat_ids)} saved conversation(s)**")

            for chat_id in reversed(chat_ids):
                conversations = load_conversations()
                chat_data = conversations.get(selected_doc, {}).get(chat_id, {})
                created = chat_data.get("created_at", "Unknown")
                msg_count = len(chat_data.get("messages", []))

                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"💬 {created} ({msg_count} msgs)", key=f"load_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = chat_data.get("messages", [])
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}", help="Delete this chat"):
                        delete_chat(selected_doc, chat_id)
                        if st.session_state.get("current_chat_id") == chat_id:
                            st.session_state.current_chat_id = None
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.info("No conversations yet. Click 'New Chat' to start.")

    st.markdown("---")

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

# ============================================
# MAIN CHAT AREA
# ============================================
st.title("💬 Research Chat")

if selected_doc:
    st.markdown(f"### Chatting with: `{selected_doc}`")
else:
    st.markdown("### Upload a document to start chatting")

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the selected document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not selected_doc:
        with st.chat_message("assistant"):
            st.error("No document selected. Please upload a document first.")
    else:
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = create_new_chat(selected_doc)

        save_message(selected_doc, st.session_state.current_chat_id, "user", query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = load_embedder()
                collection = client.get_collection(name=selected_doc)

                from settings import load_settings as load_user_settings
                user_settings = load_user_settings()
                top_k_value = user_settings.get("top_k", 5)

                start_retrieval = time.time()
                chunks, metadatas = hybrid_retrieve(
                    query,
                    collection,
                    embedder,
                    top_k=top_k_value,
                    candidate_k=15
                )
                retrieval_time = (time.time() - start_retrieval) * 1000

                context_text = ""
                sources = []

                for i, chunk in enumerate(chunks):
                    source = metadatas[i].get("source", "unknown")
                    page = metadatas[i].get("page", "unknown")
                    context_text += f"\n\n--- CHUNK {i+1} (Source: {source}, Page: {page}) ---\n{chunk}"
                    sources.append(f"Chunk {i+1}: {source}, Page {page}")

                conversation_history = ""
                for msg in st.session_state.messages[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_history += f"\n{role}: {msg['content']}"

                                # Detect intent
                intent = "GENERAL"
                query_lower = query.lower()

                if any(word in query_lower for word in ["real or fake", "is this real", "is this fake", "legit", "trustworthy", "can i trust", "phish", "scam", "verify this url", "safe to visit"]):
                    intent = "VERIFY"
                elif any(word in query_lower for word in ["summar", "what is this about", "overview", "briefly describe", "give me a summary"]):
                    intent = "SUMMARIZE"
                elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
                    intent = "COMPARE"
                elif any(word in query_lower for word in ["define", "meaning of", "what does", "explain"]):
                    intent = "EXPLAIN"
                elif any(word in query_lower for word in ["when", "where", "who", "which year", "what year", "how many", "how much"]):
                    intent = "FACTUAL"
                else:
                    intent = "GENERAL"

                # Build prompt based on intent
                if intent == "VERIFY":
                    prompt = f"""You are a URL and document verification specialist. Your ONLY job is to determine if a URL or document is trustworthy.

SOURCE: {selected_doc}

QUESTION: {query}

CONTENT EXCERPTS:
{context_text[:1000]}

VERIFICATION TASK:
1. Check if the source URL is from a known trusted domain.
2. Check if the URL uses HTTPS.
3. Look for suspicious patterns in the URL (misspellings, unusual subdomains, IP addresses).
4. Check if the content matches what would be expected from that source.
5. Give a clear VERDICT: LEGITIMATE or SUSPICIOUS.

Your response should be:
- A clear verdict (LEGITIMATE or SUSPICIOUS)
- 3-5 bullet points explaining WHY
- A final trust recommendation

DO NOT summarize the content. ONLY focus on trustworthiness."""

                elif intent == "SUMMARIZE":
                    prompt = f"""You are Provenance, a helpful AI research assistant.

SOURCE: {selected_doc}

QUESTION: {query}

CONTENT EXCERPTS:
{context_text}

TASK: Provide a concise, well-structured summary of the document/URL based ONLY on the excerpts above.
- Use headings and bullet points where appropriate.
- Highlight key facts, dates, and figures.
- End with the sources you used.

Your summary:"""

                elif intent == "COMPARE":
                    prompt = f"""You are Provenance, a helpful AI research assistant.

SOURCE: {selected_doc}

QUESTION: {query}

CONTENT EXCERPTS:
{context_text}

TASK: Compare the items mentioned in the user's question based ONLY on the excerpts above.
- Clearly explain the similarities and differences.
- Use a table or bullet points if helpful.
- Cite which chunks you used.

Your comparison:"""

                elif intent == "EXPLAIN":
                    prompt = f"""You are Provenance, a helpful AI research assistant.

SOURCE: {selected_doc}

QUESTION: {query}

CONTENT EXCERPTS:
{context_text}

TASK: Explain the concept/term the user is asking about based ONLY on the excerpts above.
- Give a clear definition first.
- Then provide details and examples from the document.
- Cite your sources.

Your explanation:"""

                elif intent == "FACTUAL":
                    prompt = f"""You are Provenance, a helpful AI research assistant.

SOURCE: {selected_doc}

QUESTION: {query}

CONTENT EXCERPTS:
{context_text}

TASK: Answer the user's factual question based ONLY on the excerpts above.
- Be direct and precise.
- Include specific dates, numbers, and facts where available.
- Cite your sources.

Your answer:"""

                else:
                    prompt = f"""You are Provenance, a helpful AI research assistant. You have access to a document called "{selected_doc}".

Here is the recent conversation history:
{conversation_history}

Here are relevant excerpts from the document:
{context_text}

The user's current question is: {query}

Instructions:
1. Answer the user's question based on the document excerpts above.
2. If the exact answer is not in the excerpts, use whatever related information IS there to give a helpful response.
3. If the excerpts are completely unrelated, say "I don't have enough information in this document to answer that question."
4. Be conversational and natural, like ChatGPT.
5. If the user is asking a follow-up question, use the conversation history to understand what they mean.
6. Mention which document chunks you used in your answer.

Your answer:"""

                start_generation = time.time()
                groq_client = Groq(api_key=GROQ_API_KEY)
                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                )
                generation_time = (time.time() - start_generation) * 1000

                answer = response.choices[0].message.content

                # === EVIDENCE VERIFICATION ENGINE ===
                evidence_result = run_evidence_verification(answer, context_text, GROQ_API_KEY)

                trust_score = evidence_result["trust_score"]
                summary = evidence_result["summary"]
                claims = evidence_result["claims"]

                if trust_score >= 80:
                    verification_result = "verified"
                    verification_label = f"✅ VERIFIED (Trust Score: {trust_score}/100)"
                elif trust_score >= 50:
                    verification_result = "partial"
                    verification_label = f"⚠️ PARTIALLY VERIFIED (Trust Score: {trust_score}/100)"
                else:
                    verification_result = "flagged"
                    verification_label = f"❌ LOW CONFIDENCE (Trust Score: {trust_score}/100)"

                log_query(query, answer, sources, retrieval_time, generation_time, verification_result)

                st.markdown(answer)

                # Only show verification for VERIFY intent
                if intent == "VERIFY":
                    if verification_label:
                        st.markdown(f"**Verification:** {verification_label}")

                    if trusted_info:
                        with st.expander("🔍 Verification Details"):
                            st.markdown(trusted_info[:800])

                # Always show sources for all intents
                with st.expander("📚 Sources Used"):
                    for s in sources:
                        st.info(s)

                # Export Answer Button
                export_text = f"QUESTION: {query}\n\nANSWER:\n{answer}\n\nSOURCES:\n"
                for s in sources:
                    export_text += f"- {s}\n"

                st.download_button(
                    label="📥 Download This Answer",
                    data=export_text.encode("utf-8"),
                    file_name="provenance_answer.txt",
                    mime="text/plain"
                )

                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_message(selected_doc, st.session_state.current_chat_id, "assistant", answer)