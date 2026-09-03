"""
decompose.py — Multi-Company Query Decomposition
=================================================
PROBLEM: "Compare NVIDIA and Microsoft on R&D spending."
A single-company retriever can't answer this. It needs to:
1. Identify which companies are mentioned.
2. Split into one sub-query per company.
3. Run retrieval separately for each company's corpus.
4. Merge the evidence before generating a final answer.

WHY RULE-BASED AND NOT LLM-BASED?
For an interview project, rule-based decomposition has major advantages:
- FULLY DETERMINISTIC: same input always produces same output. Easy to test.
- NO LATENCY/COST: no LLM call needed.
- TRANSPARENT: you can trace exactly why it split the way it did.
- DEFENSIBLE: you can explain every line of code.

A production system would use an LLM planner:
    plan = llm("Is this a multi-company question? If yes, list the companies.")
    sub_queries = [f"For {company}, {question}" for company in plan.companies]

But the rule-based version is the honest starting point. It's also what gets
you the multi-hop resume bullet — you can say "I implemented query decomposition"
whether it's rule-based or LLM-based.

INTERVIEW QUESTION: "How would you scale this beyond 6 companies?"
Answer: Replace the static COMPANY_ALIASES dict with a named-entity recognition
model (spaCy, GLiNER) that extracts company mentions from any text. Then use the
SEC EDGAR lookup to map them to CIKs dynamically.
"""

import re


# Maps lowercase company names and tickers to canonical symbols.
# The canonical symbol is what corpus.py and cli.py use to load filing chunks.
COMPANY_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOG",
    "alphabet": "GOOG",    # Alphabet is Google's parent company
    "nvidia": "NVDA",
    "tesla": "TSLA",
}


def mentioned_companies(question: str) -> list[str]:
    """
    Find supported company names or tickers mentioned in the question.

    HOW IT WORKS — two passes:
    Pass 1: Check for full company names (case-insensitive word boundary match).
            "apple" in "Compare Apple and..." → "AAPL".
    Pass 2: Check for ticker symbols (case-insensitive word boundary).
            "NVDA" in "Compare NVDA and MSFT..." → "NVDA".

    WHY WORD BOUNDARIES (\b)?
    Without \b, "amazon" would match inside "amazonian" or "amazoning".
    \b matches the zero-width position between a word character and a
    non-word character — ensuring we match whole words only.

    WHY DEDUPLICATE?
    "Apple AAPL" mentions the same company twice. The `if symbol not in symbols`
    check ensures each company appears at most once in the output list.

    RETURNS: list of canonical ticker symbols, e.g. ["NVDA", "MSFT"]
             Empty list if no supported companies found.
             Single-element list if only one company found (not a multi-hop query).
    """

    lowered = question.lower()
    symbols: list[str] = []

    # Pass 1: match full names from COMPANY_ALIASES
    for name, symbol in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered) and symbol not in symbols:
            symbols.append(symbol)

    # Pass 2: match ticker symbols (e.g. "NVDA", "MSFT")
    for symbol in set(COMPANY_ALIASES.values()):
        if re.search(rf"\b{symbol.lower()}\b", lowered) and symbol not in symbols:
            symbols.append(symbol)

    return symbols


def decompose_query(question: str) -> list[str]:
    """
    Split a comparison question into company-specific retrieval questions.

    IF the question mentions < 2 companies: return it unchanged (single-company).
    IF it mentions >= 2: return one query per company, each prefixed with
    "For {SYMBOL}, " so the retriever knows which corpus to search.

    EXAMPLE:
    "Compare NVIDIA and Microsoft on R&D spending." →
    [
        "For NVDA, Compare NVIDIA and Microsoft on R&D spending.",
        "For MSFT, Compare NVIDIA and Microsoft on R&D spending."
    ]

    WHY REPEAT THE FULL ORIGINAL QUESTION?
    The sub-query includes the full question so the retriever gets the semantic
    context of "R&D spending" — not just "NVDA R&D". The "For NVDA" prefix
    helps filter to the right company's filing corpus. In the current project,
    corpus.py loads chunks per-company, so the CLI must call run/retrieval
    separately for each sub-query.

    LIMITATION:
    This decomposition is naive — it doesn't understand WHAT aspect of the
    question applies to each company. A smarter decomposer might split:
    "Compare NVIDIA and Microsoft on R&D" →
    "What is NVIDIA's R&D spending?" and "What is Microsoft's R&D spending?"
    """

    companies = mentioned_companies(question)
    if len(companies) < 2:
        # Single-company or ambiguous question: no decomposition needed
        return [question]
    return [f"For {company}, {question}" for company in companies]


def merge_sub_answers(question: str, answers: list[str]) -> str:
    """
    Combine sub-answers without pretending the model reasoned beyond evidence.

    CURRENT IMPLEMENTATION: simple concatenation with numbering.
    "Evidence 1: {NVDA answer}\n\nEvidence 2: {MSFT answer}"

    WHY NOT SYNTHESISE?
    Without an LLM, generating a true comparison ("NVIDIA spent 4x more than
    Microsoft") requires reasoning we can't do rule-based. The safe approach
    is to concatenate and let the user (or a downstream LLM) compare.

    A PRODUCTION UPGRADE:
        context = merge_sub_answers(question, answers)
        final = call_llm(f"Based on this evidence, compare:\n{context}\n\nQuestion: {question}")

    But for this project we're honest: the system retrieves and cites evidence.
    It doesn't yet synthesise comparisons. That's noted in the README limitations.
    """

    if len(answers) == 1:
        return answers[0]

    joined = "\n\n".join(
        f"Evidence {index}: {answer}" for index, answer in enumerate(answers, start=1)
    )
    return f"Question: {question}\n\n{joined}"
