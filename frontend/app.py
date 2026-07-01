# frontend/app.py (IMPROVED)

import streamlit as st
import requests
import json
import os
from urllib.parse import urljoin
import time

# ── Configuration ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
STREAM_URL = f"{API_BASE_URL}/debate/stream"
STANDARD_URL = f"{API_BASE_URL}/debate"
REQUEST_TIMEOUT = 300  # 5 minutes

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── UI Utilities ───────────────────────────────────────
def _type_text(text: str, delay: float = 0.01):
    """
    Display text with typing animation for short content.
    Skips animation for long text to avoid excessive delays.
    """
    if len(text) > 500:
        # Long text: display immediately
        st.markdown(text)
    else:
        # Short text: animate word-by-word
        words = text.split()
        if not words:
            return
        
        holder = st.empty()
        revealed = ""
        for word in words:
            revealed += word + " "
            holder.markdown(revealed + "▌")
            time.sleep(delay)
        holder.markdown(revealed)


def _display_verdict(verdict_data):
    """Display structured judge verdict."""
    if isinstance(verdict_data, dict):
        if "error" in verdict_data:
            st.error(f"Verdict error: {verdict_data.get('error')}")
            return
        
        # Display structured verdict
        winner = verdict_data.get("winner", "UNKNOWN")
        confidence = verdict_data.get("confidence", 0.0)
        summary = verdict_data.get("summary", "No summary available")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if winner in ["FOR", "AGAINST"]:
                color = "🟢" if winner == "FOR" else "🔴"
                st.markdown(f"### {color} Winner: **{winner}**")
            else:
                st.markdown(f"### ⚪ Winner: **{winner}**")
        
        with col2:
            confidence_pct = int(confidence * 100) if isinstance(confidence, (int, float)) else 0
            st.metric("Confidence", f"{confidence_pct}%")
        
        st.markdown(f"**Summary**: {summary}")
        
        # Display additional verdict details if available
        if "winner_strengths" in verdict_data:
            st.markdown("**Winner's Strengths:**")
            for strength in verdict_data.get("winner_strengths", []):
                st.markdown(f"- {strength}")
        
        if "loser_weaknesses" in verdict_data:
            st.markdown("**Opponent's Weaknesses:**")
            for weakness in verdict_data.get("loser_weaknesses", []):
                st.markdown(f"- {weakness}")
    else:
        # Fallback: display as text
        st.info(str(verdict_data))


def _display_error_message(error_msg: str, title: str = "Error"):
    """Display error with styling."""
    st.error(f"**{title}**: {error_msg}")


# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="DebateRAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚖️ DebateRAG")
st.markdown("### Multi-Agent Research Debate System")
st.markdown("Enter a topic and watch AI agents debate it live using real research papers.")
st.divider()

# ── Sidebar Configuration ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    debug_mode = st.toggle("Debug mode", value=False)
    
    with st.expander("API Settings", expanded=False):
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="Backend API URL"
        )
        if api_url != API_BASE_URL:
            os.environ["API_BASE_URL"] = api_url
    
    st.divider()
    st.markdown("**About**")
    st.markdown("""
    DebateRAG combines:
    - **RAG**: Retrieval-augmented generation with hybrid search
    - **Multi-agent**: Specialized agents for FOR/AGAINST/JUDGE roles
    - **Evidence-based**: All arguments cite real research papers
    """)


# ── Main Input Section ────────────────────────────────
st.header("📝 Debate Setup")

col_topic, col_spacer = st.columns([3, 1])

with col_topic:
    topic = st.text_input(
        label="Research Topic",
        placeholder="e.g., RAG is better than fine-tuning for enterprise LLMs",
        help="Enter a topic for debate (5-200 characters)"
    )

col_a, col_b, col_c = st.columns([2, 1, 1])

with col_a:
    max_papers = st.slider(
        "Papers to retrieve",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="More papers = better evidence but slower"
    )

with col_b:
    use_stream = st.toggle(
        "Live streaming",
        value=True,
        help="Stream results as they're generated"
    )

with col_c:
    use_cache = st.toggle(
        "Use cache",
        value=True,
        help="Return cached result if available"
    )

st.divider()

run_button = st.button(
    "🚀 Start Debate",
    type="primary",
    use_container_width=True,
    disabled=not topic or len(topic) < 5
)

# ── Debate Execution ───────────────────────────────────
if run_button and topic:
    
    if use_stream:
        # ── STREAMING MODE ────────────────────────────
        st.markdown("### 📡 Streaming Debate")
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Stage tracking
        stages_received = {
            "for_argument": False,
            "against_argument": False,
            "for_rebuttal": False,
            "against_rebuttal": False,
            "verdict": False,
        }
        
        stage_containers = {}
        
        try:
            with requests.post(
                STREAM_URL if "http" in STREAM_URL else f"{API_BASE_URL}/debate/stream",
                json={
                    "topic": topic,
                    "max_papers": max_papers,
                    "use_cache": use_cache
                },
                stream=True,
                timeout=REQUEST_TIMEOUT
            ) as response:
                
                if response.status_code != 200:
                    _display_error_message(
                        f"API returned status {response.status_code}: {response.text}",
                        "API Error"
                    )
                else:
                    stage_progress = {"status": 0, "for_argument": 20, "against_argument": 40, "for_rebuttal": 60, "against_rebuttal": 80, "verdict": 100}
                    current_progress = 0
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        try:
                            event = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            if debug_mode:
                                st.warning(f"Failed to parse: {line[:100]}")
                            continue
                        
                        stage = event.get("stage")
                        content = event.get("content", "")
                        
                        if stage == "status":
                            status_placeholder.info(f"⏳ {content}")
                        
                        elif stage == "heartbeat":
                            # Quietly update progress without clearing previous output
                            pass
                        
                        elif stage in stages_received:
                            status_placeholder.empty()
                            stages_received[stage] = True
                            current_progress = stage_progress.get(stage, current_progress)
                            progress_bar.progress(current_progress)
                            
                            if stage == "verdict":
                                st.divider()
                                st.subheader("⚖️ Judge Verdict")
                                verdict_data = json.loads(content) if isinstance(content, str) else content
                                _display_verdict(verdict_data)
                            
                            elif stage in ("for_rebuttal", "against_rebuttal"):
                                # Rebuttal section
                                if "rebuttal_section" not in stage_containers:
                                    st.divider()
                                    st.subheader("🔄 Rebuttals")
                                    col_r1, col_r2 = st.columns(2)
                                    stage_containers["rebuttal_section"] = (col_r1, col_r2)
                                
                                col1, col2 = stage_containers["rebuttal_section"]
                                
                                if stage == "for_rebuttal":
                                    with col1:
                                        st.markdown("**🟢 FOR Rebuttal**")
                                        _type_text(content)
                                else:
                                    with col2:
                                        st.markdown("**🔴 AGAINST Rebuttal**")
                                        _type_text(content)
                            
                            else:
                                # Opening arguments
                                label = {
                                    "for_argument": "🟢 FOR Argument",
                                    "against_argument": "🔴 AGAINST Argument"
                                }.get(stage, stage)
                                
                                with st.expander(label, expanded=True):
                                    _type_text(content)
                        
                        elif stage == "done":
                            progress_bar.progress(100)
                            status_placeholder.success("✅ Debate complete!")
                            st.balloons()
                        
                        elif stage == "error":
                            progress_bar.empty()
                            _display_error_message(content, "Debate Error")
                            break
                        
                        if debug_mode and stage not in ["status", "heartbeat"]:
                            st.write(f"[DEBUG] {stage}: {content[:100]}")
        
        except requests.exceptions.Timeout:
            _display_error_message("Request timed out. Try fewer papers or check API status.", "Timeout")
        except requests.exceptions.ConnectionError:
            _display_error_message(
                f"Cannot connect to API at {API_BASE_URL}. Is the server running?",
                "Connection Error"
            )
        except Exception as e:
            _display_error_message(str(e), "Unexpected Error")
            if debug_mode:
                st.error(f"Full traceback: {type(e).__name__}: {e}")
    
    else:
        # ── STANDARD MODE (cached) ─────────────────────
        with st.spinner("⏳ Running debate..."):
            try:
                response = requests.post(
                    STANDARD_URL if "http" in STANDARD_URL else f"{API_BASE_URL}/debate",
                    json={
                        "topic": topic,
                        "max_papers": max_papers,
                        "use_cache": use_cache
                    },
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code != 200:
                    _display_error_message(
                        f"API Error {response.status_code}: {response.text}",
                        "API Error"
                    )
                    st.stop()
                
                data = response.json()
            
            except requests.exceptions.Timeout:
                _display_error_message("Request timed out.", "Timeout")
                st.stop()
            except requests.exceptions.ConnectionError:
                _display_error_message(f"Cannot connect to {API_BASE_URL}", "Connection Error")
                st.stop()
            except Exception as e:
                _display_error_message(str(e), "Request Error")
                st.stop()
        
        # Display results
        if data.get("cached"):
            st.success("⚡ Loaded from cache!")
        else:
            st.success("✅ Debate complete!")
        
        gen_time = data.get("generation_time_seconds", 0)
        if gen_time > 0:
            st.caption(f"Generated in {gen_time:.1f} seconds")
        
        st.divider()
        
        # Arguments section
        st.subheader("Opening Arguments")
        col_arg1, col_arg2 = st.columns(2)
        
        with col_arg1:
            with st.expander("🟢 FOR Argument", expanded=True):
                st.markdown(data.get("for_argument", "No argument available"))
        
        with col_arg2:
            with st.expander("🔴 AGAINST Argument", expanded=True):
                st.markdown(data.get("against_argument", "No argument available"))
        
        st.divider()
        
        # Rebuttals section
        st.subheader("Rebuttals")
        col_reb1, col_reb2 = st.columns(2)
        
        with col_reb1:
            st.markdown("**🟢 FOR Rebuttal**")
            st.markdown(data.get("for_rebuttal", "No rebuttal available"))
        
        with col_reb2:
            st.markdown("**🔴 AGAINST Rebuttal**")
            st.markdown(data.get("against_rebuttal", "No rebuttal available"))
        
        st.divider()
        
        # Verdict section
        st.subheader("⚖️ Judge Verdict")
        verdict_data = data.get("verdict")
        _display_verdict(verdict_data)
        
        st.divider()
        
        # Export section
        st.subheader("📥 Export")
        
        col_json, col_md = st.columns(2)
        
        with col_json:
            st.download_button(
                label="📄 Download as JSON",
                data=json.dumps(data, indent=2),
                file_name=f"debate_{topic[:30].replace(' ', '_')}.json",
                mime="application/json"
            )
        
        with col_md:
            markdown_content = f"""# Debate Report: {topic}

## FOR Argument
{data.get('for_argument', 'N/A')}

## AGAINST Argument
{data.get('against_argument', 'N/A')}

## FOR Rebuttal
{data.get('for_rebuttal', 'N/A')}

## AGAINST Rebuttal
{data.get('against_rebuttal', 'N/A')}

## Judge Verdict
{json.dumps(data.get('verdict'), indent=2)}
"""
            st.download_button(
                label="📝 Download as Markdown",
                data=markdown_content,
                file_name=f"debate_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown"
            )

elif run_button and not topic:
    st.warning("❌ Please enter a topic (at least 5 characters)")