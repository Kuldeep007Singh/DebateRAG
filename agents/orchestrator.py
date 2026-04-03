from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.for_agent      import run_for_agent
from agents.against_agent  import run_against_agent
from agents.judge_agent    import run_judge_agent


# ── State schema ──────────────────────────────────────────
class DebateState(TypedDict):
    topic            : str
    for_argument     : Optional[str]
    against_argument : Optional[str]
    for_rebuttal     : Optional[str]
    against_rebuttal : Optional[str]
    verdict          : Optional[str]


# ── Node functions ────────────────────────────────────────
def for_node(state: DebateState) -> DebateState:
    result = run_for_agent(state["topic"])
    state["for_argument"] = result["argument"]
    return state


def against_node(state: DebateState) -> DebateState:
    result = run_against_agent(state["topic"])
    state["against_argument"] = result["argument"]
    return state


def for_rebuttal_node(state: DebateState) -> DebateState:
    topic   = state["topic"]
    counter = state["against_argument"]

    result  = run_for_agent(
        topic = f"""
        Topic: {topic}
        You already argued FOR this topic.
        Now rebut this AGAINST argument:
        {counter}
        """
    )
    state["for_rebuttal"] = result["argument"]
    return state


def against_rebuttal_node(state: DebateState) -> DebateState:
    topic   = state["topic"]
    counter = state["for_argument"]

    result  = run_against_agent(
        topic = f"""
        Topic: {topic}
        You already argued AGAINST this topic.
        Now rebut this FOR argument:
        {counter}
        """
    )
    state["against_rebuttal"] = result["argument"]
    return state


def judge_node(state: DebateState) -> DebateState:
    result = run_judge_agent(
        topic            = state["topic"],
        for_argument     = state["for_argument"],
        against_argument = state["against_argument"],
        for_rebuttal     = state["for_rebuttal"],
        against_rebuttal = state["against_rebuttal"]
    )
    state["verdict"] = result["verdict"]
    return state


# ── Build the graph ───────────────────────────────────────
def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("for_agent",            for_node)
    graph.add_node("against_agent",        against_node)
    graph.add_node("for_rebuttal",         for_rebuttal_node)
    graph.add_node("against_rebuttal",     against_rebuttal_node)
    graph.add_node("judge",                judge_node)

    graph.set_entry_point("for_agent")
    graph.add_edge("for_agent",        "against_agent")
    graph.add_edge("against_agent",    "for_rebuttal")
    graph.add_edge("for_rebuttal",     "against_rebuttal")
    graph.add_edge("against_rebuttal", "judge")
    graph.add_edge("judge",            END)

    return graph.compile()


# ── Run ───────────────────────────────────────────────────
def run_debate(topic: str) -> DebateState:
    graph        = build_debate_graph()
    initial_state = DebateState(
        topic            = topic,
        for_argument     = None,
        against_argument = None,
        for_rebuttal     = None,
        against_rebuttal = None,
        verdict          = None
    )
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run_debate("RAG is better than fine-tuning for enterprise LLMs")
    print(result["verdict"])

