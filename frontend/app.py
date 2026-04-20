# frontend/app.py

import streamlit as st
import requests
import json


STREAM_URL   = "http://localhost:8000/debate/stream"
STANDARD_URL = "http://localhost:8000/debate"
import time

def _type_text(text: str, delay: float = 0.01):
    """Simulates typing by revealing words progressively."""
    words    = text.split()
    holder   = st.empty()
    revealed = ""
    for word in words:
        revealed += word + " "
        holder.markdown(revealed + "▌")  # cursor effect
        time.sleep(delay)
    holder.markdown(revealed)            # remove cursor at end
    
st.set_page_config(
    page_title = "DebateRAG",
    page_icon  = "⚖️",
    layout     = "wide"
)

st.title("⚖️ DebateRAG — Multi-Agent Research Debate")
st.markdown("Enter a topic and watch AI agents debate it live using real research papers.")
st.divider()

# ── Input ─────────────────────────────────────────────────
topic = st.text_input(
    label       = "Research Topic",
    placeholder = "e.g. RAG is better than fine-tuning for enterprise LLMs"
)

col_a, col_b = st.columns([2, 1])
with col_a:
    max_papers = st.slider("Papers to retrieve", 5, 25, 15)
with col_b:
    use_stream = st.toggle("Live streaming mode", value=True)

run_button = st.button("Start Debate", type="primary")

# ── Debate trigger ─────────────────────────────────────────
if run_button and topic:

    if use_stream:
        # ── STREAMING MODE ────────────────────────────────
        status_box  = st.empty()
        sections    = {}

        placeholders = {
            "for_argument"     : None,
            "against_argument" : None,
            "for_rebuttal"     : None,
            "against_rebuttal" : None,
            "verdict"          : None,
        }

        labels = {
            "for_argument"     : "🟢 FOR Argument",
            "against_argument" : "🔴 AGAINST Argument",
            "for_rebuttal"     : "🟢 FOR Rebuttal",
            "against_rebuttal" : "🔴 AGAINST Rebuttal",
            "verdict"          : "⚖️ Judge Verdict",
        }

        try:
            with requests.post(
                STREAM_URL,
                json    = {"topic": topic, "max_papers": max_papers},
                stream  = True,
                timeout = 300
            ) as response:

                for line in response.iter_lines():
                    if not line:
                        continue

                    event = json.loads(line.decode("utf-8"))
                    stage   = event.get("stage")
                    content = event.get("content")

                    if stage == "status":
                        status_box.info(f"⏳ {content}")
                    elif stage in placeholders:
                        status_box.empty()
                        label = labels[stage]

                        if stage == "verdict":
                            st.divider()
                            st.subheader(label)
                            st.info(content)

                        elif stage in ("for_rebuttal", "against_rebuttal"):
                            if not st.session_state.get("rebuttal_cols_created"):
                                st.divider()
                                st.subheader("Rebuttals")
                                st.session_state["rebuttal_cols_created"] = True
                                st.session_state["r_col1"], st.session_state["r_col2"] = st.columns(2)

                            if stage == "for_rebuttal":
                                with st.session_state["r_col1"]:
                                    st.markdown(f"**{label}**")
                                    _type_text(content)
                            else:
                                with st.session_state["r_col2"]:
                                    st.markdown(f"**{label}**")
                                    _type_text(content)

                        else:
                            with st.expander(label, expanded=True):
                                _type_text(content)

                        placeholders[stage] = True

                    elif stage == "done":
                        status_box.success("✅ Debate complete!")

                    elif stage == "error":
                        status_box.error(f"Error: {content}")

        except Exception as e:
            st.error(f"Streaming error: {e}")

    else:
        # ── STANDARD MODE (cached) ────────────────────────
        with st.spinner("Running debate..."):
            try:
                response = requests.post(
                    STANDARD_URL,
                    json    = {"topic": topic, "max_papers": max_papers},
                    timeout = 180
                )
                data = response.json()

            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        if data.get("cached"):
            st.success("⚡ Loaded from cache!")
        else:
            st.success("Debate complete!")

        st.divider()

        with st.expander("🟢 FOR Argument", expanded=True):
            st.markdown(data["for_argument"])

        with st.expander("🔴 AGAINST Argument", expanded=True):
            st.markdown(data["against_argument"])

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 FOR Rebuttal")
            st.markdown(data["for_rebuttal"])
        with col2:
            st.subheader("🔴 AGAINST Rebuttal")
            st.markdown(data["against_rebuttal"])

        st.divider()
        st.subheader("⚖️ Judge Verdict")
        st.info(data["verdict"])

        st.divider()
        st.download_button(
            label     = "Download Full Debate Report",
            data      = json.dumps(data, indent=2),
            file_name = "debate_report.json",
            mime      = "application/json"
        )

elif run_button and not topic:
    st.warning("Please enter a topic first.")