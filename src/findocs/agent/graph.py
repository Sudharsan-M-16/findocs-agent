"""A bounded retrieve-grade-rewrite-answer graph implemented with LangGraph."""

from typing import TypedDict
from langgraph.graph import END, StateGraph

from findocs.agent.answer import extractive_answer, format_cited_answer
from findocs.types import RetrievedChunk


class GraphState(TypedDict):
    question: str
    active_query: str
    retrieved: list[RetrievedChunk]
    grade: str
    retries: int
    answer: str
    verification: dict
    trace: list[str]


def build_graph(retriever, reranker=None, max_retries: int = 2):
    """Create the state machine; dependencies are injected for easy testing."""

    def retrieve(state: GraphState):
        candidates = retriever.search(state["active_query"], k=20)
        selected = reranker.rerank(state["active_query"], candidates, k=5) if reranker else candidates[:5]
        return {"retrieved": selected, "trace": state["trace"] + [f"retrieve:{state['active_query']}"]}

    def grade(state: GraphState):
        query_words = set(state["active_query"].lower().split())
        evidence_words = set(" ".join(x.chunk.text.lower() for x in state["retrieved"]).split())
        overlap = len(query_words & evidence_words) / max(len(query_words), 1)
        verdict = "good" if overlap >= 0.25 else "bad"
        return {"grade": verdict, "trace": state["trace"] + [f"grade:{verdict}:{overlap:.2f}"]}

    def rewrite(state: GraphState):
        rewritten = f"{state['active_query']} financial filing exact figures discussion"
        return {"active_query": rewritten, "retries": state["retries"] + 1, "trace": state["trace"] + [f"rewrite:{rewritten}"]}

    def answer(state: GraphState):
        text = extractive_answer(state["retrieved"])
        result = format_cited_answer(text, state["retrieved"])
        return {
            "answer": result["answer"],
            "verification": result["verification"],
            "trace": state["trace"] + ["answer:extractive"],
        }

    def route_after_grade(state: GraphState):
        if state["grade"] == "good" or state["retries"] >= max_retries:
            return "answer"
        return "rewrite"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "answer": "answer"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    return graph.compile()
