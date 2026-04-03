import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/debate"

st.set_page_config(
    page_title = "DebateRAG",
    page_icon  = "⚖️",
    layout     = "wide"
)

st.title("⚖️ DebateRAG — Multi-Agent Research Debate")
st.markdown("Enter a research topic and watch AI agents debate it using real academic papers.")

st.divider()


# ── Input ─────────────────────────────────────────────────
topic = st.text_input(
    label       = "Research Topic",
    placeholder = "e.g. RAG is better than fine-tuning for enterprise LLMs"
)

max_papers = st.slider(
    label = "Number of papers to retrieve",
    min_value = 5,
    max_value = 25,
    value     = 15
)

run_button = st.button("Start Debate", type="primary")


# ── Debate trigger ────────────────────────────────────────
if run_button and topic:
    with st.spinner("Fetching papers and running debate..."):
        try:
            response = requests.post(
                API_URL,
                json    = {"topic": topic, "max_papers": max_papers},
                timeout = 120
            )
            data = response.json()

        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    st.success("Debate complete!")
    st.divider()

    # ── FOR argument ──────────────────────────────────────
    with st.expander("🟢 FOR Argument", expanded=True):
        st.markdown(data["for_argument"])

    # ── AGAINST argument ──────────────────────────────────
    with st.expander("🔴 AGAINST Argument", expanded=True):
        st.markdown(data["against_argument"])

    st.divider()

    # ── Rebuttals side by side ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 FOR Rebuttal")
        st.markdown(data["for_rebuttal"])

    with col2:
        st.subheader("🔴 AGAINST Rebuttal")
        st.markdown(data["against_rebuttal"])

    st.divider()

    # ── Verdict ───────────────────────────────────────────
    st.subheader("⚖️ Judge Verdict")
    st.info(data["verdict"])

    # ── Download report ───────────────────────────────────
    st.divider()
    report = json.dumps(data, indent=2)
    st.download_button(
        label    = "Download Full Debate Report",
        data     = report,
        file_name= "debate_report.json",
        mime     = "application/json"
    )

elif run_button and not topic:
    st.warning("Please enter a topic first.")
