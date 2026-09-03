"""Offline tests for the logic that should remain stable across model changes."""

import unittest

from findocs.eval.metrics import answer_accuracy, citation_correctness, precision_at_k, recall_at_k, reciprocal_rank
from findocs.agent.answer import render_cited_answer
from findocs.agent.decompose import decompose_query
from findocs.agent.verify import verify_claims
from findocs.finetune.grader_eval import classification_metrics
from findocs.ingest.chunking import heading_aware_chunks, naive_chunks
from findocs.retrieval.hybrid import reciprocal_rank_fusion
from findocs.types import Chunk, RetrievedChunk


class CoreTests(unittest.TestCase):
    """Check chunk boundaries, rank metrics, and fusion independently of models."""

    def setUp(self):
        self.a = Chunk("a", "Item 7 revenue increased", "TEST", "10-K", "2024", "ITEM 7")
        self.b = Chunk("b", "Item 1A risks", "TEST", "10-K", "2024", "ITEM 1A")

    def test_heading_parser_preserves_items(self):
        chunks = heading_aware_chunks("Item 1A Risk factors. Item 7 Revenue discussion.", company="T", filing_date="2024")
        self.assertEqual({chunk.section for chunk in chunks}, {"ITEM 1A", "ITEM 7"})

    def test_naive_chunker_overlaps(self):
        chunks = naive_chunks("abcdefghijklmnop", company="T", filing_date="2024", size=10, overlap=2)
        self.assertTrue(chunks[0].text[-2:] == chunks[1].text[:2])

    def test_metrics(self):
        results = [RetrievedChunk(self.b, 2, 1, "x"), RetrievedChunk(self.a, 1, 2, "x")]
        self.assertEqual(recall_at_k(results, {"a"}, 2), 1.0)
        self.assertEqual(precision_at_k(results, {"a"}, 2), 0.5)
        self.assertEqual(reciprocal_rank(results, {"a"}), 0.5)
        self.assertEqual(answer_accuracy("Revenue increased.", ["revenue increased"]), 1.0)

    def test_rrf_keeps_both_documents(self):
        fused = reciprocal_rank_fusion([RetrievedChunk(self.a, 1, 1, "dense")], [RetrievedChunk(self.b, 1, 1, "bm25")])
        self.assertEqual({x.chunk.chunk_id for x in fused}, {"a", "b"})

    def test_verification_emits_citation_label(self):
        result = verify_claims("Revenue increased.", [RetrievedChunk(self.a, 1, 1, "dense")])
        self.assertEqual(citation_correctness(result), 1.0)
        self.assertIn("TEST 2024 10-K", result["claims"][0]["sources"][0]["citation"])

    def test_rendered_answer_contains_source_section(self):
        rendered = render_cited_answer("Revenue increased.", [RetrievedChunk(self.a, 1, 1, "dense")])
        self.assertIn("Sources:", rendered)
        self.assertIn("chunk a", rendered)

    def test_decompose_splits_multi_company_question(self):
        parts = decompose_query("Compare NVIDIA and Microsoft on research and development spending.")
        self.assertEqual(len(parts), 2)
        self.assertTrue(any("NVDA" in part for part in parts))
        self.assertTrue(any("MSFT" in part for part in parts))

    def test_grader_classification_metrics(self):
        metrics = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
