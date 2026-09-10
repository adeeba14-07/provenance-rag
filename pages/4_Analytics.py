import streamlit as st
import pandas as pd
from tracker import load_stats, get_average_retrieval_time, get_average_generation_time
from theme import apply_theme

apply_theme()

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics Evaluation")
st.markdown("### Real performance metrics from your session")

stats = load_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", stats["documents_uploaded"])

with col2:
    st.metric("Total Chunks", stats["total_chunks"])

with col3:
    st.metric("Total Queries", stats["total_queries"])

with col4:
    st.metric("Avg Retrieval", f"{get_average_retrieval_time()} ms")

st.markdown("---")

if stats["queries_history"]:
    df = pd.DataFrame(stats["queries_history"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    st.markdown("### Queries Over Time")
    df["count"] = 1
    df.set_index("timestamp", inplace=True)
    st.line_chart(df["count"].resample("D").sum().fillna(0))

    st.markdown("### Retrieval vs Generation Time")
    chart_df = pd.DataFrame({
        "Retrieval (ms)": [e["retrieval_ms"] for e in stats["queries_history"]],
        "Generation (ms)": [e["generation_ms"] for e in stats["queries_history"]]
    })
    st.bar_chart(chart_df)

    st.markdown("### Verification Results")
    ver_df = pd.DataFrame({
        "Verified": [stats["verification_results"]["verified"]],
        "Partial": [stats["verification_results"]["partial"]],
        "Flagged": [stats["verification_results"]["flagged"]]
    })
    st.bar_chart(ver_df.T)
else:
    st.info("No queries yet. Analytics will appear after you ask questions.")