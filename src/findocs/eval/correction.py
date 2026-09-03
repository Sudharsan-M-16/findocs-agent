"""Measure whether the corrective retrieval loop improves answer quality."""

from findocs.agent.graph import build_graph
from findocs.eval.metrics import answer_accuracy, citation_correctness


def run_agent_once(retriever, question: str, max_retries: int, reranker=None) -> dict:
    """Run the same graph with different retry budgets for fair comparison."""

    graph = build_graph(retriever, reranker=reranker, max_retries=max_retries)
    return graph.invoke({
        "question": question,
        "active_query": question,
        "retrieved": [],
        "grade": "",
        "retries": 0,
        "answer": "",
        "verification": {},
        "trace": [],
    })


def evaluate_self_correction(questions: list[dict], retriever, reranker=None) -> list[dict[str, float | str | int]]:
    """Compare no-retry and retry-enabled answer accuracy on the same questions."""

    rows: list[dict[str, float | str | int]] = []
    for item in questions:
        accepted = item.get("accepted_answer_phrases", [])
        for label, retries in {"without_retry": 0, "with_retry": 2}.items():
            state = run_agent_once(retriever, item["question"], max_retries=retries, reranker=reranker)
            rows.append({
                "question_id": item["id"],
                "mode": label,
                "answer_accuracy": answer_accuracy(state["answer"], accepted) if accepted else 0.0,
                "citation_correctness": citation_correctness(state["verification"]),
                "retries_used": state["retries"],
            })
    return rows
