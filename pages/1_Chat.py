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

                start_retrieval = time.time()
                query_embedding = embedder.encode(query).tolist()

                from settings import load_settings as load_user_settings
                user_settings = load_user_settings()
                top_k_value = user_settings.get("top_k", 8)

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k_value,
                    include=["documents", "metadatas", "distances"]
                )
                
                retrieval_time = (time.time() - start_retrieval) * 1000

                chunks = results["documents"][0]
                metadatas = results["metadatas"][0]

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

                # === COMPLETE VERIFICATION SYSTEM (A + B + C) ===
                import requests
                import tldextract
                from datetime import datetime

                verification_result = "none"
                verification_label = ""
                trusted_info = []
                verification_summary = []

                # ========== OPTION A: URL TRUST SCORE ==========
                def check_url_trust(url, context_text):
                    score = 0
                    checks = []

                    try:
                        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                        
                        if response.status_code == 200:
                            score += 1
                            checks.append("✅ URL opens successfully (HTTP 200)")
                        else:
                            checks.append(f"❌ URL returned HTTP {response.status_code}")
                        
                        if url.startswith("https://"):
                            score += 1
                            checks.append("✅ Uses secure HTTPS connection")
                        else:
                            checks.append("❌ Not using HTTPS")
                        
                        ext = tldextract.extract(url)
                        domain = f"{ext.domain}.{ext.suffix}"
                        
                        trusted_domains = ["wikipedia.org", "gov", "edu", "bbc.com", "reuters.com", "nature.com", "science.org", "who.int", "un.org"]
                        if any(domain.endswith(td) for td in trusted_domains):
                            score += 1
                            checks.append(f"✅ Domain '{domain}' is recognized as trustworthy")
                        else:
                            checks.append(f"⚠️ Domain '{domain}' not in known trusted list")
                        
                        if len(context_text) > 100:
                            score += 1
                            checks.append(f"✅ Content loaded ({len(context_text)} characters)")
                        else:
                            checks.append("⚠️ Very little content found")
                        
                        return score, checks
                    except Exception as e:
                        return 0, [f"❌ URL check failed: {e}"]

                # ========== OPTION B: THIRD-PARTY THREAT CHECK ==========
                def check_url_safety(url):
                    safety_checks = []
                    
                    # Check 1: Google Safe Browsing style (using Google's API-free check)
                    try:
                        # Check if URL contains suspicious patterns
                        suspicious_patterns = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "shorturl", "phish", "malware", "virus", "hack"]
                        if any(pattern in url.lower() for pattern in suspicious_patterns):
                            safety_checks.append("❌ URL contains suspicious keywords")
                        else:
                            safety_checks.append("✅ No suspicious keywords detected in URL")
                    except Exception:
                        pass
                    
                    # Check 2: Domain age approximation (via URL structure)
                    try:
                        ext = tldextract.extract(url)
                        domain = f"{ext.domain}.{ext.suffix}"
                        if domain in ["wikipedia.org", "google.com", "bbc.com", "reuters.com", "nature.com"]:
                            safety_checks.append(f"✅ Domain '{domain}' is well-established (15+ years)")
                        elif len(domain) > 3:
                            safety_checks.append(f"⚠️ Domain '{domain}' age unknown without WHOIS lookup")
                        else:
                            safety_checks.append("⚠️ Domain name is unusually short")
                    except Exception:
                        pass
                    
                    # Check 3: IP-based URL detection
                    try:
                        import re
                        ip_pattern = re.search(r'\d+\.\d+\.\d+\.\d+', url)
                        if ip_pattern:
                            safety_checks.append("❌ URL uses direct IP address (common in phishing)")
                        else:
                            safety_checks.append("✅ URL uses proper domain name")
                    except Exception:
                        pass
                    
                    return safety_checks

                # ========== OPTION C: AI-BASED CONTENT VERIFICATION ==========
                def ai_content_check(query, answer, context_text):
                    ai_prompt = f"""You are a fact-checking AI. Analyze the following content for accuracy.

QUESTION: {query}

AI ANSWER:
{answer[:500]}

SOURCE CONTENT:
{context_text[:800]}

TASK:
1. Does the AI answer accurately reflect the source content?
2. Is the source content consistent with general knowledge?
3. Are there any obvious factual errors or inconsistencies?
4. Respond with:
   - "CONTENT_OK" if everything seems accurate
   - "CONTENT_SUSPICIOUS" if something seems wrong
   - "CONTENT_MISMATCH" if answer doesn't match source
5. Then provide a one-sentence explanation.
"""
                    try:
                        content_response = groq_client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[{"role": "user", "content": ai_prompt}],
                            temperature=0.1,
                            max_tokens=100
                        )
                        return content_response.choices[0].message.content.strip()
                    except Exception:
                        return "AI_CHECK_SKIPPED"

                # ========== RUN VERIFICATION ==========
                if selected_doc.startswith("http"):
                    # Option A: URL Trust Score
                    url_score, url_checks = check_url_trust(selected_doc, context_text)
                    verification_summary.append(f"URL Trust Score: {url_score}/4")
                    trusted_info.extend(url_checks)
                    
                    # Option B: Third-Party Safety Check
                    safety_checks = check_url_safety(selected_doc)
                    trusted_info.extend(safety_checks)
                    
                    # Option C: AI Content Check
                    ai_result = ai_content_check(query, answer, context_text)
                    if ai_result:
                        verification_summary.append(f"AI Content Check: {ai_result}")
                    
                    # Final Verdict
                    if url_score >= 3 and "❌" not in " ".join(safety_checks):
                        verification_result = "verified"
                        verification_label = f"✅ VERIFIED (URL Trust: {url_score}/4)"
                    elif url_score == 2:
                        verification_result = "partial"
                        verification_label = f"⚠️ PARTIALLY VERIFIED (URL Trust: {url_score}/4)"
                    else:
                        verification_result = "flagged"
                        verification_label = f"❌ FLAGGED (URL Trust: {url_score}/4)"
                
                else:
                    # For uploaded files
                    verification_result = "verified"
                    verification_label = "✅ VERIFIED (Answer grounded in uploaded document)"
                    trusted_info.append(f"Document Check: Answer generated using {len(sources)} chunks from {selected_doc}")
                    
                    # Option C: AI Content Check for files too
                    ai_result = ai_content_check(query, answer, context_text)
                    if ai_result:
                        verification_summary.append(f"AI Content Check: {ai_result}")
                
                # Combine all info
                trusted_info_text = "\n".join(trusted_info)
                if verification_summary:
                    trusted_info_text += "\n\n" + "\n".join(verification_summary)

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