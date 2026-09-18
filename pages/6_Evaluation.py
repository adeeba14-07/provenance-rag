import streamlit as st
from tracker import load_stats, save_stats
from theme import apply_theme, render_navigation, render_page_header, render_top_bar

st.set_page_config(page_title="RAGAS Evaluation", page_icon="⌁", layout="wide")
apply_theme()
render_navigation("pages/6_Evaluation.py")
render_top_bar("Evaluation Dashboard")

render_page_header("Evaluation dashboard", "Scientific quality review", "Measure faithfulness, relevance, precision, and recall across recent answers.")

st.markdown("""
The system scores your pipeline on four industry-standard metrics:
- **Faithfulness**: Are answer claims supported by retrieved chunks?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved chunks relevant?
- **Context Recall**: Is the answer grounded in the retrieved context?
""")

st.info("""
**📌 Note on interpretation:**
Interpretive questions (e.g., *"Should I trust this?"*, *"Is this good?"*) will score **lower** on Faithfulness because their claims are opinions, not facts that can be verified from the source.
Factual questions (e.g., *"What is the value for X?"*) will score **higher** because every claim is directly traceable to the document.
**For best results, evaluate with factual questions.**
""")

st.markdown("---")

stats = load_stats()
queries = stats.get("queries_history", [])

if not queries:
    st.info("No queries yet. Ask questions in the Chat page first.")
else:
    st.markdown(f"### Ready to Evaluate: {len(queries)} queries in history")

    if st.button("▶ Run Evaluation", use_container_width=True):
        with st.spinner("Running evaluation... (30 seconds per query)"):
            import os
            from dotenv import load_dotenv
            from evaluation import run_custom_evaluation
            load_dotenv()
            groq_key = os.getenv("GROQ_API_KEY")

            test_cases = []
            for q in queries[-5:]:
    # Use raw chunks if available, otherwise fall back to sources
                contexts = q.get("raw_chunks", []) or q.get("sources", []) or ["No context available"]
                test_cases.append({
                "question": q["question"],
                "answer": q["answer"],
                "contexts": contexts,
                })

            scores = run_custom_evaluation(test_cases, groq_key)

        if scores:
            st.success(f"[OK] Evaluation complete on 4 test cases")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Faithfulness", f"{scores['faithfulness']:.3f}")
            with col2:
                st.metric("Answer Relevancy", f"{scores['answer_relevancy']:.3f}")
            with col3:
                st.metric("Context Precision", f"{scores['context_precision']:.3f}")
            with col4:
                st.metric("Context Recall", f"{scores['context_recall']:.3f}")

            st.markdown("---")
            st.markdown("### Interpretation")

            if scores["faithfulness"] >= 0.8:
                st.success(f"**Faithfulness: {scores['faithfulness']:.3f}** — Excellent grounding.")
            elif scores["faithfulness"] >= 0.6:
                st.warning(f"**Faithfulness: {scores['faithfulness']:.3f}** — Moderate grounding.")
            else:
                st.error(f"**Faithfulness: {scores['faithfulness']:.3f}** — Low grounding.")

            st.markdown("---")
            with st.expander("▦  Per-Case Breakdown"):
                for i, case in enumerate(scores["per_case"]):
                    st.markdown(f"**Case {i+1}:** {case['question'][:100]}")
                    st.markdown(f"- Faithfulness: {case['faithfulness']:.3f}")
                    st.markdown(f"- Answer Relevancy: {case['answer_relevancy']:.3f}")
                    st.markdown(f"- Context Precision: {case['context_precision']:.3f}")
                    st.markdown(f"- Context Recall: {case['context_recall']:.3f}")
                    st.markdown("")

            stats["ragas_scores"] = {
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_precision": scores["context_precision"],
                "context_recall": scores["context_recall"],
                "total_cases": scores["total_cases"],
            }
            save_stats(stats)

st.markdown("---")

if "ragas_scores" in stats:
    st.markdown("### Previous Evaluation Results")
    prev = stats["ragas_scores"]
    if prev.get("total_cases", 0) > 0:
        st.json(prev)