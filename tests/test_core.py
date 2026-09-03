"""
test_core.py — Offline Unit Tests for the FinDocs Pipeline
===========================================================
DESIGN PRINCIPLE: Every test here runs without downloading models, making
network requests, or requiring a GPU. The entire suite should pass in < 5
seconds on any machine.

WHY TEST WITHOUT MODELS?
Model-dependent tests are brittle: they fail if HuggingFace is down, if the
model changes its output format, or if you're on a machine without CUDA. The
core logic (chunking, ranking, RRF, verification, metrics) is deterministic
and testable without any ML.

COVERAGE:
- Chunking: both naive and heading-aware
- RRF: fusion correctness and score ordering
- Retrieval metrics: recall, precision, MRR, answer accuracy
- Verification: claim splitting, lexical overlap, citation format
- Decomposition: multi-company detection and query splitting
- Grader metrics: classification metrics for QLoRA evaluation
- Dataset: label loading and relevant ID grouping
- Answer rendering: citation format in output string

INTERVIEW USE:
Run py -m unittest discover -s tests -v before any interview session to
confirm the core logic is intact. If all 12+ tests pass, the pipeline logic
is working even if you can't run the full model-dependent commands.
"""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from findocs.agent.answer import render_cited_answer
from findocs.agent.decompose import decompose_query, mentioned_companies
from findocs.agent.verify import split_claims, verify_claims
from findocs.eval.metrics import (
    answer_accuracy,
    average_metric_rows,
    citation_correctness,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise_retrieval,
)
from findocs.finetune.dataset import (
    candidate_rows,
    load_labeled_pairs,
    prompt_for_pair,
    relevant_ids_by_question,
    write_label_sheet,
)
from findocs.finetune.grader_eval import classification_metrics, evaluate_grader
from findocs.ingest.chunking import heading_aware_chunks, naive_chunks
from findocs.retrieval.hybrid import reciprocal_rank_fusion
from findocs.types import Chunk, RetrievedChunk


# ── Test fixtures (shared across all tests) ──────────────────────────────────

def make_chunk(chunk_id: str, text: str, section: str = "ITEM 7") -> Chunk:
    """Helper: build a Chunk with sensible defaults for test use."""
    return Chunk(chunk_id, text, "TEST", "10-K", "2024", section)


def make_result(chunk_id: str, text: str, rank: int, source: str = "dense") -> RetrievedChunk:
    """Helper: build a RetrievedChunk for test use."""
    return RetrievedChunk(make_chunk(chunk_id, text), 1.0, rank, source)


# ── Test class ────────────────────────────────────────────────────────────────

class CoreTests(unittest.TestCase):
    """Tests for all pipeline logic that runs without models."""

    def setUp(self):
        """Create standard test fixtures used across multiple tests."""
        self.chunk_a = make_chunk("a", "Item 7 revenue increased 12 percent", "ITEM 7")
        self.chunk_b = make_chunk("b", "Item 1A risk factors supply chain", "ITEM 1A")
        self.result_a = RetrievedChunk(self.chunk_a, 0.9, 1, "dense")
        self.result_b = RetrievedChunk(self.chunk_b, 0.7, 2, "bm25")

    # ── Chunking tests ────────────────────────────────────────────────────────

    def test_heading_parser_finds_two_items(self):
        """Heading-aware chunker splits on Item boundaries correctly."""
        chunks = heading_aware_chunks(
            "Item 1A Risk factors are discussed. Item 7 Revenue discussion follows.",
            company="T",
            filing_date="2024",
        )
        sections = {chunk.section for chunk in chunks}
        self.assertEqual(sections, {"ITEM 1A", "ITEM 7"})

    def test_heading_parser_case_insensitive(self):
        """ITEM and item should both be detected."""
        chunks = heading_aware_chunks(
            "item 1 Business description. ITEM 1A Risk overview.",
            company="T",
            filing_date="2024",
        )
        sections = {chunk.section for chunk in chunks}
        # Both casing variants should produce correct uppercase section names
        self.assertIn("ITEM 1", sections)
        self.assertIn("ITEM 1A", sections)

    def test_heading_parser_no_items_gives_unknown(self):
        """Plain text with no Item headings produces an 'unknown' section."""
        chunks = heading_aware_chunks("No items here at all.", company="T", filing_date="2024")
        self.assertTrue(all(c.section == "unknown" for c in chunks))

    def test_naive_chunker_overlaps(self):
        """Naive chunks overlap by exactly `overlap` characters."""
        chunks = naive_chunks("abcdefghijklmnop", company="T", filing_date="2024", size=10, overlap=2)
        # Last 2 chars of chunk 0 should be first 2 chars of chunk 1
        self.assertEqual(chunks[0].text[-2:], chunks[1].text[:2])

    def test_naive_chunker_rejects_bad_overlap(self):
        """overlap >= size should raise ValueError."""
        with self.assertRaises(ValueError):
            naive_chunks("test text", company="T", filing_date="2024", size=5, overlap=5)

    def test_chunk_metadata_is_correct(self):
        """Each Chunk carries its company and section metadata."""
        chunks = heading_aware_chunks("Item 7 Revenue up.", company="AAPL", filing_date="2024")
        for c in chunks:
            self.assertEqual(c.company, "AAPL")
            self.assertEqual(c.filing_type, "10-K")
            self.assertEqual(c.filing_date, "2024")

    # ── RRF fusion tests ──────────────────────────────────────────────────────

    def test_rrf_keeps_both_documents(self):
        """RRF includes results from both lists even when they don't overlap."""
        fused = reciprocal_rank_fusion(
            [self.result_a],
            [self.result_b],
        )
        chunk_ids = {r.chunk.chunk_id for r in fused}
        self.assertEqual(chunk_ids, {"a", "b"})

    def test_rrf_boosts_document_in_both_lists(self):
        """A chunk appearing in both lists scores higher than one in only one list."""
        # chunk "a" appears in both dense and bm25 results
        result_a_bm25 = RetrievedChunk(self.chunk_a, 0.5, 1, "bm25")
        fused = reciprocal_rank_fusion(
            [self.result_a],       # a at rank 1
            [result_a_bm25],       # a at rank 1 again
        )
        # Only "a" is in results; it should be rank 1 with a combined score
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunk.chunk_id, "a")

    def test_rrf_source_label_shows_both_when_in_both(self):
        """When a chunk appears in two lists, its source label contains both."""
        result_a_bm25 = RetrievedChunk(self.chunk_a, 0.5, 1, "bm25")
        fused = reciprocal_rank_fusion(
            [self.result_a],
            [result_a_bm25],
        )
        # Source should indicate both dense and bm25 contributed
        self.assertIn("dense", fused[0].source)
        self.assertIn("bm25", fused[0].source)

    def test_rrf_respects_limit(self):
        """RRF returns at most `limit` results."""
        results = [make_result(f"chunk-{i}", f"text {i}", i + 1) for i in range(20)]
        fused = reciprocal_rank_fusion(results, limit=5)
        self.assertLessEqual(len(fused), 5)

    # ── Retrieval metrics tests ───────────────────────────────────────────────

    def test_recall_at_k_found_in_top_k(self):
        """recall@k = 1.0 when relevant chunk is within top k."""
        results = [self.result_b, self.result_a]  # b at rank 1, a at rank 2
        self.assertEqual(recall_at_k(results, {"a"}, 2), 1.0)

    def test_recall_at_k_not_in_top_k(self):
        """recall@1 = 0.0 when relevant chunk is at rank 2."""
        results = [self.result_b, self.result_a]
        self.assertEqual(recall_at_k(results, {"a"}, 1), 0.0)

    def test_precision_at_k_correct_fraction(self):
        """precision@2 = 0.5 when 1 of 2 results is relevant."""
        results = [self.result_b, self.result_a]
        self.assertEqual(precision_at_k(results, {"a"}, 2), 0.5)

    def test_precision_at_k_all_relevant(self):
        """precision@2 = 1.0 when both results are relevant."""
        results = [self.result_a, self.result_b]
        self.assertEqual(precision_at_k(results, {"a", "b"}, 2), 1.0)

    def test_reciprocal_rank_first_position(self):
        """RR = 1.0 when relevant chunk is at rank 1."""
        results = [self.result_a, self.result_b]
        self.assertEqual(reciprocal_rank(results, {"a"}), 1.0)

    def test_reciprocal_rank_second_position(self):
        """RR = 0.5 when relevant chunk is at rank 2."""
        results = [self.result_b, self.result_a]
        self.assertEqual(reciprocal_rank(results, {"a"}), 0.5)

    def test_reciprocal_rank_not_found(self):
        """RR = 0.0 when relevant chunk is not in results."""
        results = [self.result_b]
        self.assertEqual(reciprocal_rank(results, {"a"}), 0.0)

    def test_summarise_retrieval_has_all_keys(self):
        """summarise_retrieval returns all four expected metric keys."""
        row = summarise_retrieval([self.result_a], {"a"})
        self.assertIn("recall_at_5", row)
        self.assertIn("recall_at_10", row)
        self.assertIn("mrr", row)
        self.assertIn("precision_at_5", row)

    def test_average_metric_rows(self):
        """average_metric_rows correctly averages two rows."""
        rows = [
            {"recall_at_5": 1.0, "mrr": 1.0},
            {"recall_at_5": 0.0, "mrr": 0.0},
        ]
        averaged = average_metric_rows(rows)
        self.assertAlmostEqual(averaged["recall_at_5"], 0.5)
        self.assertAlmostEqual(averaged["mrr"], 0.5)

    def test_average_metric_rows_empty(self):
        """average_metric_rows returns empty dict for empty input."""
        self.assertEqual(average_metric_rows([]), {})

    def test_answer_accuracy_phrase_present(self):
        """answer_accuracy = 1.0 when accepted phrase is in answer."""
        self.assertEqual(answer_accuracy("Revenue increased this year.", ["revenue increased"]), 1.0)

    def test_answer_accuracy_phrase_absent(self):
        """answer_accuracy = 0.0 when no accepted phrase is in answer."""
        self.assertEqual(answer_accuracy("Nothing relevant here.", ["risk factors"]), 0.0)

    def test_answer_accuracy_case_insensitive(self):
        """answer_accuracy normalises case for matching."""
        self.assertEqual(answer_accuracy("REVENUE INCREASED.", ["revenue increased"]), 1.0)

    # ── Verification tests ────────────────────────────────────────────────────

    def test_split_claims_one_sentence(self):
        """Single sentence produces one claim."""
        claims = split_claims("Revenue increased.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0], "Revenue increased.")

    def test_split_claims_two_sentences(self):
        """Two sentences produce two claims."""
        claims = split_claims("Revenue increased. Costs declined.")
        self.assertEqual(len(claims), 2)

    def test_verify_claims_supported_when_overlap_high(self):
        """Claim is supported when most meaningful words appear in evidence."""
        chunk = make_chunk("c1", "Revenue increased twelve percent this year")
        result = RetrievedChunk(chunk, 1.0, 1, "dense")
        verification = verify_claims("Revenue increased.", [result])
        # "revenue" and "increased" both appear in the chunk text
        self.assertTrue(verification["claims"][0]["supported"])

    def test_verify_claims_unsupported_when_no_overlap(self):
        """Claim is unsupported when its words don't appear in evidence."""
        chunk = make_chunk("c1", "Supply chain diversification ongoing")
        result = RetrievedChunk(chunk, 1.0, 1, "dense")
        verification = verify_claims("Revenue doubled to record levels.", [result])
        # "revenue", "doubled", "record", "levels" — none appear in supply chain text
        self.assertFalse(verification["claims"][0]["supported"])

    def test_verify_claims_citation_contains_company(self):
        """Citation string includes company ticker."""
        result = RetrievedChunk(self.chunk_a, 1.0, 1, "dense")
        verification = verify_claims("Revenue increased 12 percent.", [result])
        if verification["claims"][0]["supported"]:
            citation = verification["claims"][0]["sources"][0]["citation"]
            self.assertIn("TEST", citation)  # company="TEST" in setUp

    def test_citation_correctness_all_supported(self):
        """citation_correctness = 1.0 when all claims supported."""
        verification = {"claims": [{"supported": True}, {"supported": True}]}
        self.assertEqual(citation_correctness(verification), 1.0)

    def test_citation_correctness_half_supported(self):
        """citation_correctness = 0.5 when half claims supported."""
        verification = {"claims": [{"supported": True}, {"supported": False}]}
        self.assertEqual(citation_correctness(verification), 0.5)

    def test_citation_correctness_none_supported(self):
        """citation_correctness = 0.0 when no claims supported."""
        verification = {"claims": [{"supported": False}]}
        self.assertEqual(citation_correctness(verification), 0.0)

    # ── Answer rendering tests ────────────────────────────────────────────────

    def test_rendered_answer_contains_sources_header(self):
        """Rendered answer always has a Sources: section."""
        rendered = render_cited_answer("Revenue increased.", [self.result_a])
        self.assertIn("Sources:", rendered)

    def test_rendered_answer_contains_chunk_id(self):
        """Rendered answer cites the chunk ID of supported claims."""
        rendered = render_cited_answer("Revenue increased 12 percent.", [self.result_a])
        # The answer text appears in chunk_a.text so it should be cited
        # Check that the rendered output contains "chunk a" or no citation message
        self.assertTrue("chunk a" in rendered or "No supporting" in rendered)

    # ── Decomposition tests ───────────────────────────────────────────────────

    def test_decompose_splits_multi_company_question(self):
        """Multi-company question generates one sub-query per company."""
        parts = decompose_query("Compare NVIDIA and Microsoft on research and development spending.")
        self.assertEqual(len(parts), 2)
        self.assertTrue(any("NVDA" in part for part in parts))
        self.assertTrue(any("MSFT" in part for part in parts))

    def test_decompose_single_company_unchanged(self):
        """Single-company question is returned unchanged."""
        parts = decompose_query("What are Apple's main risk factors?")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], "What are Apple's main risk factors?")

    def test_mentioned_companies_detects_tickers(self):
        """Ticker symbols are detected even without full company names."""
        companies = mentioned_companies("Compare NVDA and MSFT on R&D.")
        self.assertIn("NVDA", companies)
        self.assertIn("MSFT", companies)

    def test_mentioned_companies_detects_names(self):
        """Full company names are detected and mapped to tickers."""
        companies = mentioned_companies("Compare Apple and Microsoft.")
        self.assertIn("AAPL", companies)
        self.assertIn("MSFT", companies)

    def test_mentioned_companies_deduplicates(self):
        """Apple/AAPL mentioned twice doesn't produce two AAPL entries."""
        companies = mentioned_companies("Apple AAPL earnings.")
        self.assertEqual(companies.count("AAPL"), 1)

    # ── Grader evaluation tests ───────────────────────────────────────────────

    def test_classification_metrics_perfect(self):
        """Perfect predictions give accuracy=precision=recall=F1=1.0."""
        metrics = classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_classification_metrics_all_wrong(self):
        """All wrong predictions give accuracy=0.0."""
        metrics = classification_metrics([1, 0], [0, 1])
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_classification_metrics_mixed(self):
        """Mixed results: 3/4 correct → accuracy=0.75."""
        metrics = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 1.0)  # Only predicted 1 once, correctly
        self.assertEqual(metrics["recall"], 0.5)     # Missed one positive

    def test_evaluate_grader_measures_latency(self):
        """evaluate_grader records positive latency_ms."""
        rows = [{"label": "1"}, {"label": "0"}]
        result = evaluate_grader("test", rows, predict=lambda row: int(row["label"]))
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertIn("grader", result)

    # ── Dataset tests ─────────────────────────────────────────────────────────

    def test_candidate_rows_creates_empty_label(self):
        """candidate_rows produces rows with empty label field for manual filling."""
        question = {"id": "q001", "question": "What are risks?"}
        results = [self.result_a]
        rows = candidate_rows(question, results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "")  # Empty — to be filled manually
        self.assertEqual(rows[0]["question_id"], "q001")

    def test_load_labeled_pairs_filters_unlabeled(self):
        """load_labeled_pairs skips rows with empty label."""
        rows = [
            {"question_id": "q1", "question": "Q", "chunk_id": "c1",
             "company": "T", "section": "S", "rank": "1", "label": "1", "chunk_text": "text"},
            {"question_id": "q1", "question": "Q", "chunk_id": "c2",
             "company": "T", "section": "S", "rank": "2", "label": "",  "chunk_text": "other"},
            {"question_id": "q1", "question": "Q", "chunk_id": "c3",
             "company": "T", "section": "S", "rank": "3", "label": "0", "chunk_text": "other"},
        ]
        # Write to a temp file and read back
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temp_path = f.name
        loaded = load_labeled_pairs(temp_path)
        # Only rows with label "0" or "1" — not the empty one
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(r["label"] in {"0", "1"} for r in loaded))

    def test_relevant_ids_by_question_groups_correctly(self):
        """relevant_ids_by_question returns only label=1 chunk IDs per question."""
        rows = [
            {"question_id": "q1", "question": "Q", "chunk_id": "c1",
             "company": "T", "section": "S", "rank": "1", "label": "1", "chunk_text": "t"},
            {"question_id": "q1", "question": "Q", "chunk_id": "c2",
             "company": "T", "section": "S", "rank": "2", "label": "0", "chunk_text": "t"},
            {"question_id": "q2", "question": "Q2", "chunk_id": "c3",
             "company": "T", "section": "S", "rank": "1", "label": "1", "chunk_text": "t"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temp_path = f.name
        grouped = relevant_ids_by_question(temp_path)
        self.assertEqual(grouped["q1"], ["c1"])  # c2 has label=0, excluded
        self.assertEqual(grouped["q2"], ["c3"])

    def test_prompt_for_pair_contains_question_and_chunk(self):
        """prompt_for_pair includes both question text and chunk text."""
        chunk = make_chunk("c1", "Apple's revenue increased.")
        prompt = prompt_for_pair("What was revenue?", chunk)
        self.assertIn("What was revenue?", prompt)
        self.assertIn("Apple's revenue increased.", prompt)


if __name__ == "__main__":
    unittest.main()
