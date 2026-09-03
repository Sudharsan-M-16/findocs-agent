"""
graph.py — The Corrective Research State Machine (LangGraph)
=============================================================
WHAT IS LANGGRAPH?
LangGraph is a library for building stateful graphs where nodes are Python
functions and edges define the control flow between them. Think of it as a
state machine with a Python function at each state.

WHY A STATE MACHINE AND NOT JUST AN IF/ELSE SCRIPT?
You could write the same logic as:
    results = retrieve(query)
    if grade(results) == "bad" and retries < 2:
        query = rewrite(query)
        results = retrieve(query)
    answer = generate(results)

But a state machine approach gives you:
1. VISUALISABILITY: LangGraph can render the graph as a diagram (nodes + edges).
2. COMPOSABILITY: You can add new nodes/edges without restructuring if/else logic.
3. TRACEABILITY: Each state transition is logged (see the trace list in GraphState).
4. CONTROLLED LOOPS: The cycle is bounded by a counter in state — no infinite loops.

THE GRAPH STRUCTURE:
    [retrieve] → [grade] → IF grade=="bad" AND retries<max → [rewrite] → [retrieve]
                         → IF grade=="good" OR retries>=max → [answer] → END

In graph terms:
    Nodes: retrieve, grade, rewrite, answer
    Fixed edges: retrieve→grade, rewrite→retrieve, answer→END
    Conditional edge: grade → (rewrite or answer) depending on state

WHAT "STATE" MEANS:
State is a shared dict (TypedDict) that every node reads from and writes to.
Nodes return a PARTIAL update — only the keys they change. LangGraph merges
the partial update into the existing state automatically.

INTERVIEW QUESTION: "What's in the state object and why?"
- question: the original user question (never changes)
- active_query: starts = question, changes after each rewrite
- retrieved: the chunks returned by the last retrieve call
- grade: "good" or "bad" from the last grade call
- retries: how many rewrites have happened (caps the loop)
- answer: the final text (set only by the answer node)
- verification: claim-level citation analysis (set by answer node via verify.py)
- trace: append-only list of step descriptions for debugging

INTERVIEW QUESTION: "What stops the loop from running forever?"
retries >= max_retries in route_after_grade(). Once the budget is exhausted,
the conditional edge routes to "answer" regardless of grade quality. This
means we always produce an answer — possibly a poor one — rather than looping.
"""

from typing import TypedDict
from langgraph.graph import END, StateGraph

from findocs.agent.answer import extractive_answer, format_cited_answer
from findocs.types import RetrievedChunk


class GraphState(TypedDict):
    """
    The dict that flows through every node of the graph.

    LangGraph requires TypedDict (not a dataclass) because it uses the type
    annotations to validate what each node can read/write.

    TypedDict creates a regular dict at runtime — it's just typed at type-check
    time. You index it with state["key"], not state.key.
    """

    question: str       # Original user question, unchanged throughout
    active_query: str   # Current retrieval query (changes on rewrites)
    retrieved: list[RetrievedChunk]  # Last retrieve() output
    grade: str          # Last grade() output: "good" or "bad"
    retries: int        # How many times rewrite→retrieve has happened
    answer: str         # Final answer text (empty until answer node runs)
    verification: dict  # Claim verification from verify.py (empty until answer)
    trace: list[str]    # Append-only audit trail of all node visits


def build_graph(retriever, reranker=None, max_retries: int = 2):
    """
    Create the state machine; dependencies are injected for easy testing.

    WHY INJECT retriever AND reranker?
    If the retriever were hardcoded inside this function, you'd need to rebuild
    the graph every time you switch between dense/BM25/hybrid in tests or the
    ablation harness. Injection means one call to build_graph() works for all
    retrieval modes.

    WHY max_retries=2?
    Three total retrieval attempts (1 original + 2 rewrites). Empirically, if
    the first two attempts fail, a third rewrite rarely fixes the underlying
    problem (the question is too ambiguous or the corpus doesn't have the answer).
    Going higher wastes latency without improving quality.
    """

    # ── Node definitions ────────────────────────────────────────────────────

    def retrieve(state: GraphState):
        """
        Retrieve evidence chunks using the injected retriever.

        1. Call retriever.search() with the ACTIVE query (may be rewritten).
        2. If reranker is provided, rerank the top-20 candidates down to 5.
           Otherwise just take the first 5 from the top-k list.
        3. Return partial state update: new retrieved list and trace entry.

        WHY k=20 FOR RETRIEVAL THEN k=5 AFTER RERANKING?
        Bi-encoder retrieval is cheap — fetching 20 adds minimal latency.
        Cross-encoder reranking is expensive — we want it to see a wide pool
        (20) so it can find the truly best 5. If we only retrieved 5 to start,
        the reranker has nothing to improve over.
        """

        candidates = retriever.search(state["active_query"], k=20)
        selected = (
            reranker.rerank(state["active_query"], candidates, k=5)
            if reranker
            else candidates[:5]
        )
        return {
            "retrieved": selected,
            "trace": state["trace"] + [f"retrieve:{state['active_query']}"],
        }

    def grade(state: GraphState):
        """
        Judge whether retrieved evidence is sufficient to answer the question.

        CURRENT IMPLEMENTATION: word overlap heuristic.
        - Take the set of unique words in the active query.
        - Take the set of unique words in all retrieved chunk texts combined.
        - Overlap ratio = |query_words ∩ evidence_words| / |query_words|.
        - If overlap >= 0.25 (25% of query terms appear in evidence): "good".
        - Otherwise: "bad".

        WHY THIS SIMPLE HEURISTIC?
        The "real" grader is the fine-tuned QLoRA model (see finetune/).
        This heuristic lets the graph run without any LLM or trained model,
        making it testable offline. In production you'd replace this with:
            from findocs.finetune.qlora_train import load_grader_model
            verdict = model.predict(question, chunk_text)

        WHY 0.25 THRESHOLD?
        A query like "What are Apple's main risk factors?" has ~7 non-stopword
        tokens. 25% = ~2 tokens must appear in the evidence. This is lenient
        enough that mostly-relevant evidence passes but totally unrelated
        evidence (zero term overlap) fails.

        KNOWN FAILURE MODE:
        "What was the company's effective tax rate?" queries for "effective tax rate".
        If the relevant chunk says "provision for income taxes" (an accounting
        synonym), the overlap is 0 and the grade is "bad" even though the
        evidence is perfect. This is exactly why QLoRA grader training is valuable:
        a trained model learns these domain synonyms.
        """

        query_words = set(state["active_query"].lower().split())
        evidence_words = set(
            " ".join(x.chunk.text.lower() for x in state["retrieved"]).split()
        )
        overlap = len(query_words & evidence_words) / max(len(query_words), 1)
        verdict = "good" if overlap >= 0.25 else "bad"
        return {
            "grade": verdict,
            "trace": state["trace"] + [f"grade:{verdict}:{overlap:.2f}"],
        }

    def rewrite(state: GraphState):
        """
        Produce a new query when evidence was graded insufficient.

        CURRENT IMPLEMENTATION: simple template expansion.
        Appends "financial filing exact figures discussion" to the query.

        WHY THIS IS A STUB:
        A real rewriter would use an LLM to understand WHY retrieval failed
        and generate a semantically different query:
            "What is Apple's effective tax rate?" →
            "Apple provision for income taxes percentage annual report"

        The template expansion is intentionally conservative: it adds finance
        keywords that boost BM25 (exact terms) without changing the semantic
        direction (density won't shift much). This lets us test the rewrite
        loop without an LLM dependency, but in practice the quality gain is
        limited compared to an LLM-generated rewrite.

        INCREMENT retries:
        This counter is checked in route_after_grade(). If retries >= max_retries
        after this rewrite, the next grade→route will go to "answer" regardless.
        """

        rewritten = f"{state['active_query']} financial filing exact figures discussion"
        return {
            "active_query": rewritten,
            "retries": state["retries"] + 1,
            "trace": state["trace"] + [f"rewrite:{rewritten}"],
        }

    def answer(state: GraphState):
        """
        Generate an answer from the retrieved evidence and verify it.

        CALLS:
        - extractive_answer(): takes the top chunk's text as the answer.
          This is conservative — it doesn't synthesise or hallucinate.
        - format_cited_answer(): passes the answer and evidence to verify.py,
          which checks each claim for evidence support and returns citations.

        WHY EXTRACTIVE?
        An extractive answer directly quotes the filing. It cannot hallucinate
        because it doesn't generate new text — it retrieves text. The trade-off
        is lower readability (raw filing text is dense) but zero fabrication risk.

        A PRODUCTION UPGRADE WOULD:
        Pass the question + retrieved chunks to an LLM (e.g. GPT-4, LLaMA-3)
        as context and ask it to synthesise a readable answer. Then run verify.py
        on the synthetic answer to catch any claims the LLM fabricated.
        """

        text = extractive_answer(state["retrieved"])
        result = format_cited_answer(text, state["retrieved"])
        return {
            "answer": result["answer"],
            "verification": result["verification"],
            "trace": state["trace"] + ["answer:extractive"],
        }

    # ── Conditional routing ─────────────────────────────────────────────────

    def route_after_grade(state: GraphState):
        """
        Decide the next node after a grade result.

        RETURNS: "answer" or "rewrite" (must match edge targets in add_conditional_edges)

        LOGIC:
        - If grade is "good": no rewrite needed → go to answer.
        - If retries >= max_retries: budget exhausted → give up and answer anyway.
        - Otherwise: grade is "bad" and budget remains → rewrite and retry.

        WHY CHECK RETRIES HERE AND NOT IN rewrite()?
        The routing function is called AFTER grade(). At that point, retries
        reflects the count from the PREVIOUS rewrite (or 0 for the first run).
        Checking it here means: "if we've already rewritten max_retries times,
        stop even if the grade is still bad."
        """

        if state["grade"] == "good" or state["retries"] >= max_retries:
            return "answer"
        return "rewrite"

    # ── Graph construction ──────────────────────────────────────────────────

    graph = StateGraph(GraphState)

    # Register each Python function as a graph node
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)

    # The graph always enters at "retrieve"
    graph.set_entry_point("retrieve")

    # Fixed edges: after retrieve always grade; after rewrite always retrieve
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)  # END is a special LangGraph sentinel

    # Conditional edge: after grade, call route_after_grade() to pick next node
    # The dict maps return values to target node names
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"rewrite": "rewrite", "answer": "answer"},
    )

    # compile() validates the graph (no disconnected nodes, no missing edges)
    # and returns an executable Runnable with .invoke() method
    return graph.compile()
