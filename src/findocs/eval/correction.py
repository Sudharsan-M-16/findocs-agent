"""
correction.py — Self-Correction Loop Evaluation
================================================
PURPOSE: Measure whether the corrective retrieval loop actually improves answer quality.

THE CLAIM ON YOUR RESUME:
"Implemented an agentic self-correction loop that improved answer accuracy by X%."

This module is HOW you generate the X%. It runs the same questions through the
same retriever twice:
  1. WITHOUT retry (max_retries=0): one retrieval attempt, generate answer.
  2. WITH retry (max_retries=2): up to 3 retrieval attempts, grade each one.

If the "with retry" accuracy > "without retry" accuracy, the loop adds value.

EXPECTED RESULT:
- On ambiguous queries ("How did revenue change?"), the rewrite adds specificity
  ("How did revenue change? financial filing exact figures discussion") which
  may pull different, more specific chunks.
- On clear queries, the difference should be small (the first retrieval already
  found relevant chunks → grade="good" → no rewrite).

IMPORTANT: answer_accuracy requires non-empty accepted_answer_phrases in
eval_questions.json. If a question has no phrases, it scores 0.0 for both modes —
not because retry failed, but because we can't measure. Fill the phrases first.
"""

from findocs.agent.graph import build_graph
from findocs.eval.metrics import answer_accuracy, citation_correctness


def run_agent_once(retriever, question: str, max_retries: int, reranker=None) -> dict:
    """
    Run the same graph with different retry budgets for fair comparison.

    WHY BUILD A NEW GRAPH EACH TIME?
    LangGraph graphs are stateless templates — each .invoke() call creates a
    fresh execution context. Building a new graph with a different max_retries
    value and then invoking it gives us the "without retry" vs "with retry"
    comparison using the exact same graph code.

    WHY THE SAME RETRIEVER?
    We use the same DenseRetriever + BM25Retriever instance for both runs.
    This ensures the comparison is purely about the retry logic — not about
    any randomness in the retriever itself.

    INITIAL STATE DICT:
    LangGraph's .invoke() takes the initial state dict. We set sensible defaults:
    - active_query = question (starts with the original query)
    - retrieved, grade, answer, verification = empty (nothing retrieved yet)
    - retries = 0 (no rewrites yet)
    - trace = [] (empty audit log)

    RETURNS: the final GraphState dict after the graph completes.
    """

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


def evaluate_self_correction(
    questions: list[dict],
    retriever,
    reranker=None,
) -> list[dict[str, float | str | int]]:
    """
    Compare no-retry and retry-enabled answer accuracy on the same questions.

    FOR EACH QUESTION:
    - Run with max_retries=0 → "without_retry" row.
    - Run with max_retries=2 → "with_retry" row.
    - Compute answer_accuracy against accepted_answer_phrases (if provided).
    - Compute citation_correctness from the verification dict.
    - Record retries_used to show whether the loop actually activated.

    OUTPUT FORMAT (one row per question per mode):
    {
        "question_id": "q001",
        "mode": "with_retry",
        "answer_accuracy": 1.0,
        "citation_correctness": 0.75,
        "retries_used": 1
    }

    WHY TWO ROWS PER QUESTION?
    The ablation comparison needs matching rows for the same question in both
    modes. Storing them together in one CSV with a "mode" column means you can
    filter by mode and average, giving the comparison table.

    WHY retries_used?
    If retries_used = 0 for a "with_retry" question, the grade was "good" on
    the first attempt — no improvement was possible from retries. A case where
    retries_used = 2 and accuracy improved is the clearest demonstration that
    the loop added value.
    """

    rows: list[dict[str, float | str | int]] = []

    for item in questions:
        # accepted phrases are the ground truth for answer quality scoring
        accepted = item.get("accepted_answer_phrases", [])

        for label, retries in {"without_retry": 0, "with_retry": 2}.items():
            state = run_agent_once(
                retriever,
                item["question"],
                max_retries=retries,
                reranker=reranker,
            )
            rows.append({
                "question_id": item["id"],
                "mode": label,
                # If no accepted phrases, answer_accuracy defaults to 0.0 for both modes
                "answer_accuracy": (
                    answer_accuracy(state["answer"], accepted) if accepted else 0.0
                ),
                # Fraction of answer claims that are citation-supported
                "citation_correctness": citation_correctness(state["verification"]),
                # How many times the graph actually rewrote and re-retrieved
                "retries_used": state["retries"],
            })

    return rows
