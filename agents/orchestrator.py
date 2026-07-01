# agents/langgraph_orchestrator.py

import time
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from RAG.cache import (
    cache_debate,
    is_redis_available
)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class DebateState(TypedDict):
    topic: str
    chunks: List[Dict]

    for_argument: Optional[str]
    against_argument: Optional[str]

    for_rebuttal: Optional[str]
    against_rebuttal: Optional[str]

    verdict: Optional[str]

    for_chunks: List[Dict]
    against_chunks: List[Dict]

    timings: Dict[str, float]


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def for_argument_node(state: DebateState):

    from agents.for_agent import run_for_agent

    print("\n[LangGraph] FOR agent running...")

    start = time.time()

    try:
        result = run_for_agent(
            topic=state["topic"],
            all_chunks=state["chunks"]
        )

        state["for_argument"] = result["argument"]
        state["for_chunks"] = result.get("chunks", [])

    except Exception as e:
        state["for_argument"] = f"Error: {str(e)}"

    state["timings"]["for_argument"] = time.time() - start

    return state


def against_argument_node(state: DebateState):

    from agents.against_agent import run_against_agent

    print("[LangGraph] AGAINST agent running...")

    start = time.time()

    try:
        result = run_against_agent(
            topic=state["topic"],
            all_chunks=state["chunks"]
        )

        state["against_argument"] = result["argument"]
        state["against_chunks"] = result.get("chunks", [])

    except Exception as e:
        state["against_argument"] = f"Error: {str(e)}"

    state["timings"]["against_argument"] = time.time() - start

    return state


def for_rebuttal_node(state: DebateState):

    from agents.for_rebuttal import run_for_rebuttal

    print("[LangGraph] FOR rebuttal running...")

    start = time.time()

    try:
        result = run_for_rebuttal(
            topic=state["topic"],
            own_argument=state["for_argument"],
            opposing_argument=state["against_argument"],
            all_chunks=state["chunks"]
        )

        state["for_rebuttal"] = result["rebuttal"]

    except Exception as e:
        state["for_rebuttal"] = f"Error: {str(e)}"

    state["timings"]["for_rebuttal"] = time.time() - start

    return state


def against_rebuttal_node(state: DebateState):

    from agents.against_rebuttal import run_against_rebuttal

    print("[LangGraph] AGAINST rebuttal running...")

    start = time.time()

    try:
        result = run_against_rebuttal(
            topic=state["topic"],
            own_argument=state["against_argument"],
            opposing_argument=state["for_argument"],
            all_chunks=state["chunks"]
        )

        state["against_rebuttal"] = result["rebuttal"]

    except Exception as e:
        state["against_rebuttal"] = f"Error: {str(e)}"

    state["timings"]["against_rebuttal"] = time.time() - start

    return state


def judge_node(state: DebateState):

    from agents.judge_agent import run_judge_agent

    print("[LangGraph] Judge running...")

    start = time.time()

    try:
        result = run_judge_agent(
            topic=state["topic"],
            for_argument=state["for_argument"],
            against_argument=state["against_argument"],
            for_chunks=state["for_chunks"],
            against_chunks=state["against_chunks"],
            for_rebuttal=state["for_rebuttal"],
            against_rebuttal=state["against_rebuttal"]
        )

        state["verdict"] = result["verdict"]

    except Exception as e:
        state["verdict"] = f"Error: {str(e)}"

    state["timings"]["judge_verdict"] = time.time() - start

    return state


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

def build_graph():

    graph = StateGraph(DebateState)

    graph.add_node("for_argument", for_argument_node)
    graph.add_node("against_argument", against_argument_node)

    graph.add_node("for_rebuttal", for_rebuttal_node)
    graph.add_node("against_rebuttal", against_rebuttal_node)

    graph.add_node("judge", judge_node)

    # FLOW

    graph.set_entry_point("for_argument")

    graph.add_edge("for_argument", "against_argument")

    graph.add_edge("against_argument", "for_rebuttal")

    graph.add_edge("for_rebuttal", "against_rebuttal")

    graph.add_edge("against_rebuttal", "judge")

    graph.add_edge("judge", END)

    return graph.compile()


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def run_debate(topic: str, all_chunks: List[Dict]):

    app = build_graph()

    initial_state = {
        "topic": topic,
        "chunks": all_chunks,

        "for_argument": None,
        "against_argument": None,

        "for_rebuttal": None,
        "against_rebuttal": None,

        "verdict": None,

        "for_chunks": [],
        "against_chunks": [],

        "timings": {}
    }

    final_state = app.invoke(initial_state)

    if is_redis_available():
        try:
            cache_debate(topic, final_state)
            print("[LangGraph] Debate cached")
        except Exception as e:
            print(f"[LangGraph] Cache failed: {e}")

    return final_state