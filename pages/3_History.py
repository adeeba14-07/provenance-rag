import streamlit as st
import pandas as pd
from tracker import load_stats
from theme import apply_theme

apply_theme()

st.set_page_config(page_title="History", page_icon="🕐", layout="wide")

st.title("🕐 Audit Trail")
st.markdown("### Your actual query history")

stats = load_stats()

if stats["queries_history"]:
    # Export buttons
    col1, col2 = st.columns(2)

    with col1:
        df = pd.DataFrame(stats["queries_history"])
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="provenance_history.csv",
            mime="text/csv"
        )

    with col2:
        import json
        json_data = json.dumps(stats["queries_history"], indent=4).encode("utf-8")
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="provenance_history.json",
            mime="application/json"
        )

    st.markdown("---")

    for entry in reversed(stats["queries_history"]):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer'][:300]}...")
        st.markdown(f"*{entry['timestamp']}*")
        st.markdown(f"Retrieval: **{entry['retrieval_ms']} ms** | Generation: **{entry['generation_ms']} ms**")
        if entry["sources"]:
            st.markdown("**Sources:**")
            for s in entry["sources"]:
                st.markdown(f"- {s}")
        st.markdown("---")
else:
    st.info("No queries yet. Ask questions in the Chat page to see them here.")