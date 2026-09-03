"""Command-line entry points for the FinDocs build."""

import argparse
import json
from pathlib import Path


def load_questions(path: str) -> list[dict]:
    """Read the hand-built evaluation set from JSON."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_companies(value: str) -> list[str]:
    """Accept AAPL,MSFT style input and normalise whitespace/case."""

    return [item.strip().upper() for item in value.split(",") if item.strip()]


def smoke() -> None:
    """Run offline checks that do not require model downloads or network access."""

    from findocs.retrieval.hybrid import reciprocal_rank_fusion
    from findocs.types import Chunk, RetrievedChunk

    a = Chunk("a", "revenue increased", "TEST", "10-K", "2024", "ITEM 7")
    b = Chunk("b", "risk factors", "TEST", "10-K", "2024", "ITEM 1A")
    result = reciprocal_rank_fusion([RetrievedChunk(a, 1, 1, "dense")], [RetrievedChunk(b, 1, 1, "bm25")])
    assert {x.chunk.chunk_id for x in result} == {"a", "b"}
    print("smoke test passed: RRF and imports work")


def run(company: str, email: str | None) -> None:
    """Build indexes for one company and print inspectable retrieval results."""

    from findocs.corpus import load_company_chunks
    from findocs.retrieval.bm25 import BM25Retriever
    from findocs.retrieval.dense import DenseRetriever
    from findocs.retrieval.hybrid import HybridRetriever

    chunks = load_company_chunks(company, email=email)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(dense, BM25Retriever(chunks))
    questions = [
        "What are the main risk factors?",
        "What was the research and development spending?",
        "How did revenue change?",
    ]
    for question in questions:
        print(f"\nQUESTION: {question}")
        for item in hybrid.search(question, k=5):
            print(item.rank, item.score, item.chunk.section, item.chunk.chunk_id, item.chunk.text[:220])
    print(f"\nChunks: {len(chunks)}")


def ask(company: str, email: str | None, question: str, use_reranker: bool) -> None:
    """Run the agent end to end and print the reasoning trace."""

    from findocs.agent.answer import render_cited_answer
    from findocs.agent.graph import build_graph
    from findocs.corpus import load_company_chunks
    from findocs.retrieval.bm25 import BM25Retriever
    from findocs.retrieval.dense import DenseRetriever
    from findocs.retrieval.hybrid import HybridRetriever

    chunks = load_company_chunks(company, email=email)
    hybrid = HybridRetriever(DenseRetriever(chunks), BM25Retriever(chunks))
    reranker = None
    if use_reranker:
        from findocs.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    graph = build_graph(hybrid, reranker=reranker)
    state = graph.invoke({
        "question": question,
        "active_query": question,
        "retrieved": [],
        "grade": "",
        "retries": 0,
        "answer": "",
        "verification": {},
        "trace": [],
    })
    print("TRACE:")
    print(json.dumps(state["trace"], indent=2))
    print("\nANSWER:")
    print(render_cited_answer(state["answer"], state["retrieved"])[:3500])
    print("\nVERIFICATION:")
    print(json.dumps(state["verification"], indent=2))


def label_candidates(companies: str, email: str | None, questions_path: str, output_path: str) -> None:
    """Create a CSV for human labels used by retrieval eval and QLoRA training."""

    from findocs.corpus import load_corpus
    from findocs.eval.pipeline import build_retrieval_stages
    from findocs.finetune.dataset import candidate_rows, write_label_sheet

    questions = load_questions(questions_path)
    chunks = load_corpus(parse_companies(companies), email=email)
    stages = build_retrieval_stages(chunks, use_reranker=False)
    rows = []
    for question in questions:
        rows.extend(candidate_rows(question, stages["dense_bm25_rrf"](question["question"])[:10]))
    write_label_sheet(rows, output_path)
    print(f"wrote {len(rows)} candidate rows to {output_path}")


def show_candidates(label_sheet: str, question_id: str) -> None:
    """Print candidates for one question so manual labels are easier to assign."""

    import csv

    with Path(label_sheet).open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["question_id"] == question_id]
    if not rows:
        raise ValueError(f"No rows found for {question_id} in {label_sheet}.")
    for row in rows:
        print(f"\n{row['question_id']} rank={row['rank']} chunk={row['chunk_id']} section={row['section']} label={row['label']}")
        print(row["chunk_text"][:1200])


def sync_labels(questions_path: str, label_sheet: str, output_path: str) -> None:
    """Copy relevant chunk IDs from a completed label sheet into eval JSON."""

    from findocs.finetune.dataset import relevant_ids_by_question

    questions = load_questions(questions_path)
    labels = relevant_ids_by_question(label_sheet)
    for question in questions:
        if question["id"] in labels:
            question["relevant_chunk_ids"] = labels[question["id"]]
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)
        handle.write("\n")
    print(f"updated {output_path}")


def eval_retrieval(companies: str, email: str | None, questions_path: str, output_path: str, use_reranker: bool) -> None:
    """Run dense/BM25/RRF/reranker ablations and save resume-ready metrics."""

    from findocs.corpus import load_corpus
    from findocs.eval.ablation import aggregate_by_stage, evaluate_stages, write_csv
    from findocs.eval.pipeline import build_retrieval_stages

    questions = load_questions(questions_path)
    chunks = load_corpus(parse_companies(companies), email=email)
    stages = build_retrieval_stages(chunks, use_reranker=use_reranker)
    rows = evaluate_stages(questions, stages)
    summary = aggregate_by_stage(rows)
    write_csv(rows, output_path)
    write_csv(summary, str(Path(output_path).with_name("retrieval_ablation_summary.csv")))
    print(json.dumps(summary, indent=2))


def eval_correction(companies: str, email: str | None, questions_path: str, output_path: str) -> None:
    """Measure answer accuracy and citation correctness with retry disabled/enabled."""

    from findocs.corpus import load_corpus
    from findocs.eval.ablation import aggregate_by_stage, write_csv
    from findocs.eval.correction import evaluate_self_correction
    from findocs.retrieval.bm25 import BM25Retriever
    from findocs.retrieval.dense import DenseRetriever
    from findocs.retrieval.hybrid import HybridRetriever

    questions = load_questions(questions_path)
    chunks = load_corpus(parse_companies(companies), email=email)
    retriever = HybridRetriever(DenseRetriever(chunks), BM25Retriever(chunks))
    rows = evaluate_self_correction(questions, retriever)
    write_csv(rows, output_path)
    summary_input = [
        {
            "stage": row["mode"],
            "question_id": row["question_id"],
            "answer_accuracy": row["answer_accuracy"],
            "citation_correctness": row["citation_correctness"],
        }
        for row in rows
    ]
    summary = aggregate_by_stage(summary_input)
    write_csv(summary, str(Path(output_path).with_name("self_correction_summary.csv")))
    print(json.dumps(summary, indent=2))


def decompose(question: str) -> None:
    """Print the Day 9 sub-queries for a multi-company question."""

    from findocs.agent.decompose import decompose_query

    print(json.dumps(decompose_query(question), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--company", default="AAPL")
    run_parser.add_argument("--email")

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("--company", default="AAPL")
    ask_parser.add_argument("--email")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--rerank", action="store_true")

    label_parser = sub.add_parser("label-candidates")
    label_parser.add_argument("--companies", default="AAPL")
    label_parser.add_argument("--email")
    label_parser.add_argument("--questions", default="data/eval_questions.json")
    label_parser.add_argument("--output", default="data/grader_label_sheet.csv")

    show_parser = sub.add_parser("show-candidates")
    show_parser.add_argument("--labels", default="data/grader_label_sheet.csv")
    show_parser.add_argument("--question-id", required=True)

    sync_parser = sub.add_parser("sync-labels")
    sync_parser.add_argument("--questions", default="data/eval_questions.json")
    sync_parser.add_argument("--labels", default="data/grader_label_sheet.csv")
    sync_parser.add_argument("--output", default="data/eval_questions.json")

    eval_parser = sub.add_parser("eval-retrieval")
    eval_parser.add_argument("--companies", default="AAPL")
    eval_parser.add_argument("--email")
    eval_parser.add_argument("--questions", default="data/eval_questions.json")
    eval_parser.add_argument("--output", default="results/retrieval_ablation.csv")
    eval_parser.add_argument("--rerank", action="store_true")

    correction_parser = sub.add_parser("eval-correction")
    correction_parser.add_argument("--companies", default="AAPL")
    correction_parser.add_argument("--email")
    correction_parser.add_argument("--questions", default="data/eval_questions.json")
    correction_parser.add_argument("--output", default="results/self_correction.csv")

    decompose_parser = sub.add_parser("decompose")
    decompose_parser.add_argument("--question", required=True)

    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "run":
        run(args.company, args.email)
    elif args.command == "ask":
        ask(args.company, args.email, args.question, args.rerank)
    elif args.command == "label-candidates":
        label_candidates(args.companies, args.email, args.questions, args.output)
    elif args.command == "show-candidates":
        show_candidates(args.labels, args.question_id)
    elif args.command == "sync-labels":
        sync_labels(args.questions, args.labels, args.output)
    elif args.command == "eval-retrieval":
        eval_retrieval(args.companies, args.email, args.questions, args.output, args.rerank)
    elif args.command == "eval-correction":
        eval_correction(args.companies, args.email, args.questions, args.output)
    else:
        decompose(args.question)


if __name__ == "__main__":
    main()
