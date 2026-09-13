import streamlit as st
import time
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from tracker import log_query
from conversations import (
    load_conversations, create_new_chat, save_message,
    delete_chat, get_all_chats_for_document
)
from retrieval import hybrid_retrieve
from verifier import verify_answer
from theme import apply_theme

apply_theme()
st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path="./chroma_db")

try:
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
except Exception:
    collection_names = []

with st.sidebar:
    st.markdown("## 📚 Documents")
    if collection_names:
        selected_doc = st.selectbox("Choose document:", collection_names, key="selected_doc")

        # Reset chat when the document changes
        if "last_selected_doc" not in st.session_state:
            st.session_state.last_selected_doc = selected_doc
        elif st.session_state.last_selected_doc != selected_doc:
            st.session_state.last_selected_doc = selected_doc
            st.session_state.messages = []
            st.session_state.current_chat_id = None

    st.markdown("---")
    st.markdown("## 💬 Conversations")

    if selected_doc:
        if st.button("➕ New Chat", use_container_width=True):
            new_id = create_new_chat(selected_doc)
            st.session_state.current_chat_id = new_id
            st.session_state.messages = []
            st.rerun()

        chat_ids = get_all_chats_for_document(selected_doc)
        if chat_ids:
            st.markdown(f"**{len(chat_ids)} saved conversation(s)**")
            for chat_id in reversed(chat_ids):
                convos = load_conversations()
                chat_data = convos.get(selected_doc, {}).get(chat_id, {})
                created = chat_data.get("created_at", "Unknown")
                count = len(chat_data.get("messages", []))
                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button(f"💬 {created} ({count})", key=f"load_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = chat_data.get("messages", [])
                        st.rerun()
                with c2:
                    if st.button("🗑️", key=f"del_{chat_id}"):
                        delete_chat(selected_doc, chat_id)
                        if st.session_state.get("current_chat_id") == chat_id:
                            st.session_state.current_chat_id = None
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.info("No conversations yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

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

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Re-render verification and citations if this is an assistant message with data
        if m["role"] == "assistant" and m.get("verification"):
            st.markdown(f"**Verification:** {m['verification']}")

            with st.expander("🔍 Evidence & Citations", expanded=False):
                summary = m.get("summary", {})
                claims = m.get("claims", [])
                if summary.get("note"):
                    st.info(f"ℹ️ {summary['note']}")
                line = (f"**Coverage:** {summary.get('supported', 0)} supported, "
                        f"{summary.get('partial', 0)} partial, "
                        f"{summary.get('unsupported', 0)} unsupported "
                        f"(out of {summary.get('total', 0)} verifiable claims)")
                if summary.get("interpretations", 0) > 0:
                    line += f" — {summary['interpretations']} interpretation(s) excluded"
                st.markdown(line)
                st.markdown("---")
                for i, c in enumerate(claims):
                    icon = "✅" if c["status"] == "SUPPORTED" else ("⚠️" if c["status"] == "PARTIAL" else "❌")
                    badge = " 🧮" if c.get("type") == "CALCULATION" else (" 💭" if c.get("type") == "INTERPRETATION" else "")
                    st.markdown(f"{icon} **Claim {i+1}**{badge}: {c['claim']}")
                    st.markdown(f"   *{c.get('type', 'FACTUAL')} — {c['status']}*")
                    if c.get("similarity"):
                        st.markdown(f"   Similarity: {c['similarity']}")
                    if c.get("reason"):
                        st.markdown(f"   {c['reason']}")
                    if c.get("citation"):
                        cit = c["citation"]
                        st.markdown(f"   📎 `{cit['source']}`, {cit['location']}, Chunk {cit['chunk']} ({cit['confidence']})")
                    st.markdown("")

            if m.get("sources"):
                with st.expander("📚 Sources Used"):
                    for s in m["sources"]:
                        st.info(s)

query = st.chat_input("Ask a question about the selected document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not selected_doc:
        with st.chat_message("assistant"):
            st.error("No document selected.")
    else:
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = create_new_chat(selected_doc)
        save_message(selected_doc, st.session_state.current_chat_id, "user", query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = load_embedder()
                collection = client.get_collection(name=selected_doc)

                from settings import load_settings
                us = load_settings()
                top_k = us.get("top_k", 5)

                history = ""
                for m in st.session_state.messages[-6:]:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history += f"\n{role}: {m['content']}"

                last_answer = ""
                for m in reversed(st.session_state.messages):
                    if m["role"] == "assistant" and len(m["content"]) > 50:
                        last_answer = m["content"][:800]
                        break

                # Rewrite ONLY for follow-ups
                retrieval_query = query
                q_lower = query.lower()
                follow_markers = ["that", "it", "this", "the same", "in binary", "in french",
                                  "as a table", "tell me more", "what about", "and what",
                                  "explain that", "translate", "convert"]
                looks_followup = (len(st.session_state.messages) > 1
                                  and any(mk in q_lower for mk in follow_markers)
                                  and len(query.split()) < 10)

                if looks_followup:
                    rewrite_prompt = f"""Rewrite the user's latest question into a self-contained question using the previous answer.

Previous answer:
{last_answer}

Latest question: {query}

Rules:
- Include the specific topic from the previous answer.
- If already standalone, return unchanged.
- Return ONLY the rewritten question.

Rewritten:"""
                    try:
                        gclient_rw = Groq(api_key=GROQ_API_KEY)
                        rw = gclient_rw.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[{"role": "user", "content": rewrite_prompt}],
                            temperature=0.0, max_tokens=1500
                        )
                        raw = rw.choices[0].message.content.strip()
                        raw = raw.strip('"').strip("'").strip("*").strip()
                        if len(raw) >= 5 and "**" not in raw:
                            retrieval_query = raw
                    except Exception:
                        pass

                # Transformation requests skip retrieval
                transform_kw = ["in binary", "in french", "in spanish", "in json",
                                "as a table", "as json", "as csv", "in hex",
                                "in morse", "summarize that", "one sentence"]
                is_transform = any(kw in q_lower for kw in transform_kw)

                if is_transform and last_answer:
                    prompt = f"""The user wants a transformation of your previous answer.

Previous answer:
{last_answer}

Request: {query}

Apply the transformation. Output ONLY the transformed content."""
                    chunks, metadatas = [], []
                else:
                    start = time.time()
                    chunks, metadatas = hybrid_retrieve(retrieval_query, collection, embedder,
                                                         top_k=top_k, candidate_k=15)
                    retrieval_time = (time.time() - start) * 1000

                    context_text = ""
                    sources = []
                    for i, chunk in enumerate(chunks):
                        src = metadatas[i].get("source", "unknown")
                        loc = metadatas[i].get("location", "Section")
                        context_text += f"\n\n--- CHUNK {i+1} ({src}, {loc}) ---\n{chunk}"
                        sources.append(f"Chunk {i+1}: {src}, {loc}")

                    if looks_followup:
                        prompt = f"""You are Provenance, a precise AI research assistant.

Recent conversation:
{history}

Document excerpts from "{selected_doc}":
{context_text}

User question: {retrieval_query}

Instructions:
1. The user is asking a follow-up. Use the recent conversation to understand what "that" or "it" refers to.
2. Answer using ONLY the document excerpts above.
3. If the excerpts are unrelated, say exactly: "The document does not contain information about this."
4. Be direct. Use bullet points or tables when helpful.
5. If the excerpts do not contain the information, say: "The document does not contain information about this."
6. Write factual answers as complete sentences. Do NOT output bare lists like "2018, 2019" — write "The data covers the years 2018 and 2019." 
7. Do NOT use LaTeX math notation. Write formulas in plain text like "SMAPE = (1/n) * sum(abs(actual - predicted) / ((abs(actual) + abs(predicted)) / 2)) * 100".

Your answer:"""
                    else:
                        prompt = f"""You are Provenance, a precise AI research assistant.

Document excerpts from "{selected_doc}":
{context_text}

User question: {retrieval_query}

Instructions:
1. This is a new question. Do NOT refer to or continue any previous answer.
2. Answer using ONLY the document excerpts above.
3. If the excerpts are unrelated, say exactly: "The document does not contain information about this."
4. Be direct. Use bullet points or tables when helpful.
5. Preserve exact numbers, dates, and names.
6. Do not invent facts.
7. If the excerpts do not contain the information, say: "The document does not contain information about this."
8. Write factual answers as complete sentences. Do NOT output bare lists like "2018, 2019" — write "The data covers the years 2018 and 2019." 
9. Do NOT use LaTeX math notation. Write formulas in plain text like "SMAPE = (1/n) * sum(abs(actual - predicted) / ((abs(actual) + abs(predicted)) / 2)) * 100".


Your answer:"""

                start = time.time()
                gclient = Groq(api_key=GROQ_API_KEY)
                resp = gclient.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=1500
                )
                generation_time = (time.time() - start) * 1000
                answer = resp.choices[0].message.content

                answer = re.sub(r'\[svg\]\([^\)]+\)', '', answer)
                answer = re.sub(r'\[.*?\]\(http://localhost:8501[^\)]*\)', '', answer)
                answer = re.sub(r'http://localhost:8501[^\s]*', '', answer)
                answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer).strip()

                # Verify only if we retrieved chunks
                if chunks:
                    result = verify_answer(answer, chunks, metadatas, GROQ_API_KEY, retrieval_query)
                    trust = result["trust_score"]
                    summary = result["summary"]
                    claims = result["claims"]
                else:
                    trust = 100
                    summary = {"supported": 0, "partial": 0, "unsupported": 0,
                               "total": 0, "interpretations": 0,
                               "note": "Transformation request — no verification needed"}
                    claims = []

                if trust >= 85:
                    v_result, v_label = "verified", f"✅ VERIFIED ({trust}/100)"
                elif trust >= 60:
                    v_result, v_label = "partial", f"⚠️ MOSTLY VERIFIED ({trust}/100)"
                elif trust >= 30:
                    v_result, v_label = "partial", f"🟡 PARTIALLY VERIFIED ({trust}/100)"
                else:
                    v_result, v_label = "flagged", f"❌ LOW CONFIDENCE ({trust}/100)"

                log_query(query, answer, sources if chunks else [],
                          retrieval_time if chunks else 0,
                          generation_time, v_result, trust, chunks if chunks else [])

                st.markdown(answer)
                st.markdown(f"**Verification:** {v_label}")

                with st.expander("🔍 Evidence & Citations", expanded=False):
                    if summary.get("note"):
                        st.info(f"ℹ️ {summary['note']}")
                    line = (f"**Coverage:** {summary['supported']} supported, "
                            f"{summary['partial']} partial, {summary['unsupported']} unsupported "
                            f"(out of {summary['total']} verifiable claims)")
                    if summary.get("interpretations", 0) > 0:
                        line += f" — {summary['interpretations']} interpretation(s) excluded"
                    st.markdown(line)
                    st.markdown("---")
                    for i, c in enumerate(claims):
                        icon = "✅" if c["status"] == "SUPPORTED" else ("⚠️" if c["status"] == "PARTIAL" else "❌")
                        badge = " 🧮" if c["type"] == "CALCULATION" else (" 💭" if c["type"] == "INTERPRETATION" else "")
                        st.markdown(f"{icon} **Claim {i+1}**{badge}: {c['claim']}")
                        st.markdown(f"   *{c['type']} — {c['status']}*")
                        st.markdown(f"   {c['reason']}")
                        if c.get("citation"):
                            cit = c["citation"]
                            st.markdown(f"   📎 `{cit['source']}`, {cit['location']}, Chunk {cit['chunk']} ({cit['confidence']})")
                        st.markdown("")

                if chunks:
                    with st.expander("📚 Sources Used"):
                        for s in sources:
                            st.info(s)

                export = f"Q: {query}\n\nA:\n{answer}\n\nSOURCES:\n"
                for s in sources if chunks else []:
                    export += f"- {s}\n"
                st.download_button("📥 Download Answer", export.encode("utf-8"),
                                   file_name="provenance_answer.txt", mime="text/plain")

                                # Save the full message with verification metadata
                message_data = {
                    "role": "assistant",
                    "content": answer,
                    "verification": v_label,
                    "trust_score": trust,
                    "summary": summary,
                    "claims": claims,
                    "sources": sources if chunks else []
                }
                st.session_state.messages.append(message_data)
                save_message(selected_doc, st.session_state.current_chat_id, "assistant", answer)