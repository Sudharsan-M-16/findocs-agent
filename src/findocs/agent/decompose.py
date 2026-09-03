"""Small query decomposition utilities for Day 9 multi-hop questions."""

import re


COMPANY_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOG",
    "alphabet": "GOOG",
    "nvidia": "NVDA",
    "tesla": "TSLA",
}


def mentioned_companies(question: str) -> list[str]:
    """Find supported company names or tickers mentioned in the question."""

    lowered = question.lower()
    symbols: list[str] = []
    for name, symbol in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered) and symbol not in symbols:
            symbols.append(symbol)
    for symbol in set(COMPANY_ALIASES.values()):
        if re.search(rf"\b{symbol.lower()}\b", lowered) and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def decompose_query(question: str) -> list[str]:
    """Split a comparison question into company-specific retrieval questions."""

    companies = mentioned_companies(question)
    if len(companies) < 2:
        return [question]
    return [f"For {company}, {question}" for company in companies]


def merge_sub_answers(question: str, answers: list[str]) -> str:
    """Combine sub-answers without pretending the model reasoned beyond evidence."""

    if len(answers) == 1:
        return answers[0]
    joined = "\n\n".join(f"Evidence {index}: {answer}" for index, answer in enumerate(answers, start=1))
    return f"Question: {question}\n\n{joined}"
