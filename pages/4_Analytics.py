import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import zipfile
from datetime import datetime
from tracker import load_stats, get_average_retrieval_time, get_average_generation_time
from theme import apply_theme, render_navigation, render_page_header, render_top_bar

apply_theme()
render_navigation("pages/4_Analytics.py")
render_top_bar("Analytics Evaluation")

st.set_page_config(page_title="Analytics", page_icon="▥", layout="wide")

render_page_header("Analytical dashboard", "Quality and performance", "A live view of retrieval speed, generation cost, and answer confidence.")

stats = load_stats()


def base_figure(height=330):
    return {
        "height": height,
        "margin": {"l": 42, "r": 20, "t": 12, "b": 36},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#fafaf4",
        "font": {"family": "Manrope, sans-serif", "color": "#494551", "size": 11},
        "hoverlabel": {"font": {"family": "JetBrains Mono, monospace"}},
        "legend": {"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10}},
    }


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

    # ---- Document filter for download ----
    all_docs = sorted(df["sources"].apply(
        lambda s: s[0].split(":")[1].strip().split(",")[0] if s and len(s) > 0 and ":" in s[0] else "unknown"
    ).unique().tolist())

    filter_docs = st.multiselect(
        "Filter analytics by document (leave empty for all):",
        options=all_docs,
        default=[],
        help="Select one or more documents to filter the graphs and downloads."
    )

    if filter_docs:
        df = df[df["sources"].apply(
            lambda s: any(d in str(s) for d in filter_docs)
        )]
        st.caption(f"Filtered to {len(df)} queries from {len(filter_docs)} document(s).")

    # ---- Download all data as CSV ----
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇ Download Analytics Data (CSV)",
        csv_buffer.getvalue().encode("utf-8"),
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("---")

    # ===== CHART 1: Quality Over Time =====
    st.markdown("### RAGAS Quality Over Time")
    st.caption("Trust score and verification coverage across recorded queries")
    quality = go.Figure()
    quality.add_trace(go.Scatter(
        x=df["timestamp"], y=df["trust_score"].fillna(0), mode="lines+markers",
        name="Accuracy", line={"color": "#1a3717", "width": 3},
        marker={"color": "#1a3717", "size": 7},
    ))
    quality.add_trace(go.Scatter(
        x=df["timestamp"], y=(df["verification"].eq("verified").astype(int) * 100),
        mode="lines+markers", name="Context Precision", line={"color": "#2c4f2e", "width": 2, "dash": "dot"},
        marker={"color": "#2c4f2e", "size": 6},
    ))
    quality.update_layout(**base_figure(340),
                          yaxis={"range": [0, 100], "title": "Score", "gridcolor": "#e3e3dd", "zeroline": False},
                          xaxis={"gridcolor": "rgba(0,0,0,0)", "title": ""})
    st.plotly_chart(quality, use_container_width=True, config={"displayModeBar": False})

    # Download individual chart as HTML (self-contained)
    quality_html = quality.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(
        "⬇ Download Quality Chart",
        quality_html.encode("utf-8"),
        file_name="quality_over_time.html",
        mime="text/html",
        key="dl_quality",
    )

    st.markdown("---")

    left_chart, right_chart = st.columns([1.6, 1], gap="large")

    # ===== CHART 2: Retrieval Strategy =====
    with left_chart:
        st.markdown("### Retrieval Strategy")
        st.caption("Latency profile by pipeline stage")
        strategy = go.Figure()
        strategy.add_trace(go.Bar(
            y=["Generation", "Retrieval", "Total query"],
            x=[df["generation_ms"].mean(), df["retrieval_ms"].mean(), (df["generation_ms"] + df["retrieval_ms"]).mean()],
            orientation="h", marker_color=["#c3c8bd", "#1a3717", "#2c4f2e"],
            text=[f"{value:.0f} ms" for value in [df["generation_ms"].mean(), df["retrieval_ms"].mean(),
                                                   (df["generation_ms"] + df["retrieval_ms"]).mean()]],
            textposition="outside", cliponaxis=False, name="Latency",
        ))
        strategy.update_layout(**base_figure(270), showlegend=False,
                               xaxis={"gridcolor": "#e3e3dd", "title": "Milliseconds"},
                               yaxis={"gridcolor": "rgba(0,0,0,0)"})
        st.plotly_chart(strategy, use_container_width=True, config={"displayModeBar": False})
        st.download_button(
            "⬇ Download Retrieval Chart",
            strategy.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
            file_name="retrieval_strategy.html",
            mime="text/html",
            key="dl_strategy",
        )

    # ===== CHART 3: Verification Results =====
    with right_chart:
        st.markdown("### Verification Results")
        st.caption("Answer confidence distribution")
        verification = stats["verification_results"]
        donut = go.Figure(go.Pie(
            labels=["Verified", "Partial", "Flagged"],
            values=[verification["verified"], verification["partial"], verification["flagged"]],
            hole=0.64,
            marker={"colors": ["#1a3717", "#bdefbc", "#66394b"], "line": {"color": "#ffffff", "width": 2}},
            textinfo="label+percent", textfont={"family": "Manrope, sans-serif", "size": 10},
        ))
        donut.update_layout(**{**base_figure(270), "showlegend": False, "margin": {"l": 8, "r": 8, "t": 12, "b": 12}})
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
        st.download_button(
            "⬇ Download Verification Chart",
            donut.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
            file_name="verification_results.html",
            mime="text/html",
            key="dl_verification",
        )

    st.markdown("---")

    # ===== CHART 4: Heatmap =====
    st.markdown("### Corpus Coverage Heatmap")
    st.caption("Query frequency by weekday and hour")
    heat_df = df.copy()
    heat_df["weekday"] = heat_df["timestamp"].dt.day_name().str[:3]
    heat_df["hour"] = heat_df["timestamp"].dt.hour
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))
    counts = heat_df.pivot_table(index="weekday", columns="hour", values="question",
                                 aggfunc="count", fill_value=0).reindex(index=weekdays, columns=hours, fill_value=0)
    heatmap = go.Figure(go.Heatmap(
        z=counts.values, x=[f"{hour:02d}:00" for hour in hours], y=weekdays,
        colorscale=[[0, "#f4f4ee"], [0.25, "#c8ecbe"], [0.6, "#1a3717"], [1, "#2c4f2e"]],
        xgap=2, ygap=2, hovertemplate="%{y} %{x}<br>Queries: %{z}<extra></extra>", showscale=False,
    ))
    heatmap.update_layout(**{**base_figure(260), "margin": {"l": 46, "r": 10, "t": 12, "b": 42},
                              "xaxis": {"side": "bottom", "showgrid": False},
                              "yaxis": {"autorange": "reversed", "showgrid": False}})
    st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})
    st.download_button(
        "⬇ Download Heatmap Chart",
        heatmap.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
        file_name="coverage_heatmap.html",
        mime="text/html",
        key="dl_heatmap",
    )

    st.markdown("---")

    # ===== Download ALL charts in one ZIP =====
    st.markdown("### 📦 Download All Charts + Data")
    st.caption("Get every chart as an interactive HTML file, plus a CSV of the raw data, in a single ZIP archive.")

    if st.button("Build ZIP Package", key="build_zip"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("quality_over_time.html", quality.to_html(include_plotlyjs="cdn", full_html=True))
            zf.writestr("retrieval_strategy.html", strategy.to_html(include_plotlyjs="cdn", full_html=True))
            zf.writestr("verification_results.html", donut.to_html(include_plotlyjs="cdn", full_html=True))
            zf.writestr("coverage_heatmap.html", heatmap.to_html(include_plotlyjs="cdn", full_html=True))
            zf.writestr("analytics_data.csv", df.to_csv(index=False))
            zf.writestr("README.txt",
                        f"Provenance RAG Analytics Export\nGenerated: {datetime.now().isoformat()}\n"
                        f"Queries: {len(df)}\nDocuments: {len(all_docs)}\n"
                        f"Filtered to: {filter_docs if filter_docs else 'All documents'}\n")
        zip_buffer.seek(0)
        st.download_button(
            "⬇ Download Full ZIP Package",
            zip_buffer.getvalue(),
            file_name=f"provenance_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            key="dl_zip",
        )
else:
    st.info("No queries yet. Analytics will appear after you ask questions.")