import streamlit as st
import time
import os
import re
import chromadb
from groq import Groq
from dotenv import load_dotenv

from tracker import log_query
from conversations import (
    load_conversations, create_new_chat, save_message,
    delete_chat, get_all_chats_for_document
)
from retrieval import hybrid_retrieve
from verifier import get_embedder, verify_answer
from theme import apply_theme, render_navigation, render_top_bar

apply_theme()
st.set_page_config(page_title="Chat", page_icon="◌", layout="wide")
render_navigation("pages/1_Chat.py")
render_top_bar("Interactive Research Chat")

st.markdown(
    """
    <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 0 !important;
            padding-left: 0.2rem !important;
            padding-right: 0.2rem !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
        }
        [data-testid="stAppViewContainer"] > .main > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
            margin-top: 0 !important;
            gap: 0.5rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(.chat-shell) > [data-testid="stColumn"]:first-child {
            border-right: 1px solid #c3c8bd;
            padding-right: 1rem !important;
            box-sizing: border-box;
        }
        [data-testid="stHorizontalBlock"]:has(.chat-shell) > [data-testid="stColumn"]:last-child {
            padding-left: 1rem !important;
            box-sizing: border-box;
        }
        [data-testid="stColumn"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        [data-testid="stColumn"] > div {
            margin-top: 0 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: 0 !important;
        }
        
        .chat-column-wrap,
        .inspector-column-wrap {
            height: calc(100vh - 185px);
            min-height: 560px;
            overflow-y: auto;
            padding: 0 0.15rem;
            margin-top: 0 !important;
            box-sizing: border-box;
        }
        .chat-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 0.2rem 0.3rem 0.5rem;
            min-height: 0;
        }
        .chat-input-row {
            margin-top: auto;
            padding-top: 0.45rem;
            border-top: 1px solid #dfe3d9;
            flex-shrink: 0;
        }
        .chat-input-row [data-testid="stChatInput"] {
            border-radius: 999px !important;
            border: 1px solid #c3c8bd !important;
            background: #ffffff;
            box-shadow: inset 0 1px 2px rgba(26, 28, 25, 0.03);
        }
        .chat-input-row [data-testid="stChatInput"] textarea,
        .chat-input-row [data-testid="stChatInput"] input {
            background: transparent !important;
            border: none !important;
            font-size: 0.96rem !important;
            padding-left: 1rem !important;
        }
        .chat-input-row button {
            background: #1a3717 !important;
            border-radius: 999px !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 8px 18px rgba(26, 55, 23, 0.18);
        }
       
        .inspector-shell .stMarkdownContainer p,
        .inspector-shell .stMarkdownContainer li,
        .inspector-shell .stMarkdownContainer span,
        .inspector-shell .stMarkdownContainer strong,
        .inspector-shell .stCaptionContainer {
            color: #1a1c19 !important;
        }
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            border-radius: 1rem 1rem 0.5rem 1rem !important;
            padding: 0.9rem 1rem !important;
            box-shadow: 0 4px 12px rgba(26, 28, 25, 0.04);
        }
        [data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
            border-radius: 1rem 0.5rem 1rem 1rem !important;
            background: #1a3717 !important;
            border-color: #1a3717 !important;
            color: #ffffff !important;
        }
        [data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_GROQ_MODEL = "openai/gpt-oss-120b"


def generate_completion(gclient, prompt, max_tokens):
    """Generate text and recover from an unavailable configured model."""
    models = [GROQ_MODEL]
    if GROQ_MODEL != FALLBACK_GROQ_MODEL:
        models.append(FALLBACK_GROQ_MODEL)

    last_error = None
    for model in models:
        try:
            return gclient.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            is_model_error = "model" in error_text or "404" in error_text
            if not is_model_error:
                raise

    raise last_error


client = chromadb.PersistentClient(path="./chroma_db")

try:
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
except Exception:
    collection_names = []

selected_doc = None
conversations = load_conversations()

with st.sidebar:
    st.markdown("## Documents")
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
    st.markdown("## Conversations")

    if selected_doc:
        if st.button("＋ New Chat", use_container_width=True):
            new_id = create_new_chat(selected_doc)
            st.session_state.current_chat_id = new_id
            st.session_state.messages = []
            st.rerun()

        chat_ids = get_all_chats_for_document(selected_doc)
        if chat_ids:
            st.markdown(f"**{len(chat_ids)} saved conversation(s)**")
            for chat_id in reversed(chat_ids):
                chat_data = conversations.get(selected_doc, {}).get(chat_id, {})
                created = chat_data.get("created_at", "Unknown")
                count = len(chat_data.get("messages", []))
                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button(f"◌ {created} ({count})", key=f"load_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = chat_data.get("messages", [])
                        st.rerun()
                with c2:
                    if st.button("×", key=f"del_{chat_id}"):
                        delete_chat(selected_doc, chat_id)
                        if st.session_state.get("current_chat_id") == chat_id:
                            st.session_state.current_chat_id = None
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.info("No conversations yet.")

    st.markdown("---")
    if st.button("× Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

chat_column, inspector_column = st.columns([2.3, 1.05], gap="small", vertical_alignment="top")

with chat_column:

    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    if selected_doc:
        st.markdown(f"**ACTIVE DOCUMENT**  `{selected_doc}`")
    else:
        st.info("Upload a document to activate the research workspace.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

            if m["role"] == "assistant" and m.get("verification"):
                st.markdown(f"**Verification:** {m['verification']}")

                with st.expander("⌕ Evidence & Citations", expanded=False):
                    summary = m.get("summary", {})
                    claims = m.get("claims", [])
                    if summary.get("note"):
                        st.info(f"NOTE: {summary['note']}")
                    line = (f"**Coverage:** {summary.get('supported', 0)} supported, "
                            f"{summary.get('partial', 0)} partial, "
                            f"{summary.get('unsupported', 0)} unsupported "
                            f"(out of {summary.get('total', 0)} verifiable claims)")
                    if summary.get("interpretations", 0) > 0:
                        line += f" — {summary['interpretations']} interpretation(s) excluded"
                    st.markdown(line)
                    st.markdown("---")
                    for i, c in enumerate(claims):
                        icon = "[OK]" if c["status"] == "SUPPORTED" else ("[PARTIAL]" if c["status"] == "PARTIAL" else "[FLAG]")
                        badge = " [CALC]" if c.get("type") == "CALCULATION" else (" [INTERPRETATION]" if c.get("type") == "INTERPRETATION" else "")
                        st.markdown(f"{icon} **Claim {i+1}**{badge}: {c['claim']}")
                        st.markdown(f"   *{c.get('type', 'FACTUAL')} — {c['status']}*")
                        if c.get("similarity"):
                            st.markdown(f"   Similarity: {c['similarity']}")
                        if c.get("reason"):
                            st.markdown(f"   {c['reason']}")
                        if c.get("citation"):
                            cit = c["citation"]
                            st.markdown(f"   REF `{cit['source']}`, {cit['location']}, Chunk {cit['chunk']} ({cit['confidence']})")
                        st.markdown("")

                if m.get("sources"):
                    with st.expander("▣ Sources Used"):
                        for s in m["sources"]:
                            st.info(s)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)
    st.markdown('<div class="stitch-chip-row"><span class="stitch-chip-label">Recommended</span></div>', unsafe_allow_html=True)
    chip_columns = st.columns(3)
    for chip_column, chip_label in zip(chip_columns, ["Compare sources", "Check citations", "Summarize findings"]):
        with chip_column:
            if st.button(chip_label, key=f"chip_{chip_label}", use_container_width=True):
                st.session_state.pending_query = chip_label
                st.rerun()
    query = st.chat_input("Ask a follow-up question about the selected document...")
    query = query or st.session_state.pop("pending_query", None)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
                embedder = get_embedder()
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
                        rw = generate_completion(gclient_rw, rewrite_prompt, 300)
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
                try:
                    resp = generate_completion(gclient, prompt, 900)
                except Exception as error:
                    st.error(
                        "Groq could not generate an answer. Check that your API key "
                        "has access to the configured Groq model."
                    )
                    st.stop()
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
                    v_result, v_label = "verified", f"[VERIFIED] ({trust}/100)"
                elif trust >= 60:
                    v_result, v_label = "partial", f"[MOSTLY VERIFIED] ({trust}/100)"
                elif trust >= 30:
                    v_result, v_label = "partial", f"[PARTIALLY VERIFIED] ({trust}/100)"
                else:
                    v_result, v_label = "flagged", f"[LOW CONFIDENCE] ({trust}/100)"

                log_query(query, answer, sources if chunks else [],
                          retrieval_time if chunks else 0,
                          generation_time, v_result, trust, chunks if chunks else [])

                st.markdown(answer)
                st.markdown(f"**Verification:** {v_label}")

                with st.expander("⌕ Evidence & Citations", expanded=False):
                    if summary.get("note"):
                        st.info(f"NOTE: {summary['note']}")
                    line = (f"**Coverage:** {summary['supported']} supported, "
                            f"{summary['partial']} partial, {summary['unsupported']} unsupported "
                            f"(out of {summary['total']} verifiable claims)")
                    if summary.get("interpretations", 0) > 0:
                        line += f" — {summary['interpretations']} interpretation(s) excluded"
                    st.markdown(line)
                    st.markdown("---")
                    for i, c in enumerate(claims):
                        icon = "[OK]" if c["status"] == "SUPPORTED" else ("[PARTIAL]" if c["status"] == "PARTIAL" else "[FLAG]")
                        badge = " [CALC]" if c["type"] == "CALCULATION" else (" [INTERPRETATION]" if c["type"] == "INTERPRETATION" else "")
                        st.markdown(f"{icon} **Claim {i+1}**{badge}: {c['claim']}")
                        st.markdown(f"   *{c['type']} — {c['status']}*")
                        st.markdown(f"   {c['reason']}")
                        if c.get("citation"):
                            cit = c["citation"]
                            st.markdown(f"   REF `{cit['source']}`, {cit['location']}, Chunk {cit['chunk']} ({cit['confidence']})")
                        st.markdown("")

                if chunks:
                    with st.expander("▣ Sources Used"):
                        for s in sources:
                            st.info(s)

                                # ---- Enhanced export with full evidence ----
                export_lines = [
                    "=" * 70,
                    "PROVENANCE RAG — ANSWER EXPORT",
                    "=" * 70,
                    f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Document: {selected_doc}",
                    "",
                    "-" * 70,
                    "QUESTION",
                    "-" * 70,
                    query,
                    "",
                    "-" * 70,
                    "ANSWER",
                    "-" * 70,
                    answer,
                    "",
                    "-" * 70,
                    "VERIFICATION",
                    "-" * 70,
                    f"Status: {v_label}",
                    f"Trust score: {trust}/100",
                    f"Coverage: {summary.get('supported', 0)} supported, {summary.get('partial', 0)} partial, {summary.get('unsupported', 0)} unsupported (out of {summary.get('total', 0)})",
                    "",
                ]

                if claims:
                    export_lines.append("-" * 70)
                    export_lines.append("CLAIM-BY-CLAIM VERIFICATION")
                    export_lines.append("-" * 70)
                    for i, c in enumerate(claims):
                        export_lines.append(f"\n[Claim {i+1}] {c.get('claim', '')}")
                        export_lines.append(f"  Type: {c.get('type', 'FACTUAL')}")
                        export_lines.append(f"  Status: {c.get('status', 'UNKNOWN')}")
                        if c.get("reason"):
                            export_lines.append(f"  Reason: {c['reason']}")
                        if c.get("citation"):
                            cit = c["citation"]
                            export_lines.append(f"  Source: {cit.get('source', '')}, {cit.get('location', '')}, Chunk {cit.get('chunk', '')} ({cit.get('confidence', '')})")
                    export_lines.append("")

                if sources:
                    export_lines.append("-" * 70)
                    export_lines.append("SOURCES USED")
                    export_lines.append("-" * 70)
                    for s in sources:
                        export_lines.append(f"  • {s}")
                    export_lines.append("")

                export_lines.append("=" * 70)
                export_lines.append("END OF EXPORT")
                export_lines.append("=" * 70)

                export = "\n".join(export_lines)
                fname = f"provenance_answer_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button("↓ Download Full Answer (with evidence + sources)",
                                   export.encode("utf-8"),
                                   file_name=fname, mime="text/plain")

                # Save the full message with verification metadata
                message_data = {
                    "role": "assistant",
                    "content": answer,
                    "verification": v_label,
                    "trust_score": trust,
                    "summary": summary,
                    "claims": claims,
                    "sources": sources if chunks else [],
                    "raw_chunks": chunks if chunks else [],
                }
                st.session_state.messages.append(message_data)
                save_message(selected_doc, st.session_state.current_chat_id, "assistant", answer)
                st.rerun()

# ============================================================
# SOURCE INSPECTOR (rendered AFTER chat processing)
# ============================================================
with inspector_column:
    st.markdown('<div class="inspector-shell">', unsafe_allow_html=True)
    st.markdown("### Source Inspector")
    st.caption("LIVE EVIDENCE PANEL")

    msgs = st.session_state.get("messages", [])

    assistant_msgs = [
        (i, m) for i, m in enumerate(msgs) if m.get("role") == "assistant"
    ]

    if selected_doc:
        st.markdown(f"**Document analysis**  \n`{selected_doc}`")

        if assistant_msgs:
            options = ["Latest"] + [
                f"Answer {n+1}" for n in range(len(assistant_msgs))
            ]
            choice = st.selectbox(
                "Inspect answer:",
                options,
                index=0,
                key="inspector_answer_choice"
            )

            if choice == "Latest":
                inspected_msg = assistant_msgs[-1][1]
            else:
                n = int(choice.split()[1]) - 1
                inspected_msg = assistant_msgs[n][1]
        else:
            inspected_msg = None

        if inspected_msg and inspected_msg.get("trust_score") is not None:
            trust_score = inspected_msg.get("trust_score", 0)
            confidence = "High confidence" if trust_score >= 85 else "Review evidence"
            st.metric("Trust score", f"{trust_score}/100", confidence)
        else:
            st.info("Ask a question to populate source evidence.")

        st.markdown("#### Extracted fragments")
        fragments = inspected_msg.get("raw_chunks", []) if inspected_msg else []
        if not fragments and inspected_msg:
            fragments = inspected_msg.get("sources", [])

        if fragments:
            for index, fragment in enumerate(fragments[:4], start=1):
                text = fragment if isinstance(fragment, str) else str(fragment)
                st.markdown(
                    f'<div class="provenance-panel"><b>Fragment {index}</b><br>'
                    f'<span style="font-style:italic;">{text[:360]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Retrieved chunks will appear here after your first question.")
    else:
        st.markdown("### Source Inspector")
        st.caption("Select or ingest a document to inspect evidence.")

    st.markdown("#### Provenance timeline")

    has_docs = bool(selected_doc)
    has_query = len(msgs) > 0
    has_answer = len(msgs) > 1 and msgs[-1].get("role") == "assistant"

    st.markdown(
        f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; line-height:2; color:#494551;">
        {'●' if has_docs else '○'} Document indexed<br>
        {'●' if has_query else '○'} Retrieved for this query<br>
        {'●' if has_answer else '○'} Answer verified
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)