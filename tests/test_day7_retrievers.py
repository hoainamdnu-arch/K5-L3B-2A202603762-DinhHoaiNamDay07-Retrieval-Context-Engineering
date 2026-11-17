"""Offline unit tests for the Day 7 retrieval logic (no model / vector store).

Covers BM25 ranking, RRF fusion, question-driven filter inference and the
lexical rerank. The dense stage is replaced by a deterministic stub so these
run without FastEmbed or Qdrant.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day7.retrievers import (BM25Retriever, HybridRRFRetriever, RerankRetriever,
                             infer_filters, tokenize)

CHUNKS = [
    {"id": "c1", "text": "The White Rabbit took a watch out of its waistcoat-pocket.",
     "source_id": "pg11", "source": "s", "page": 1, "metadata": {"chapter_number": 1,
     "book_title": "Alice's Adventures in Wonderland"}},
    {"id": "c2", "text": "Alice was swimming in the pool of tears she had wept.",
     "source_id": "pg11", "source": "s", "page": 7, "metadata": {"chapter_number": 2,
     "book_title": "Alice's Adventures in Wonderland"}},
    {"id": "c3", "text": "It is a Looking-glass book; hold it up to a glass to read it.",
     "source_id": "pg12", "source": "s", "page": 6, "metadata": {"chapter_number": 1,
     "book_title": "Through the Looking-Glass"}},
]


class StubDense:
    """Returns chunks in a fixed order regardless of query (deterministic)."""
    name = "dense"
    def __init__(self, order):
        self.order = order
        self.by_id = {c["id"]: c for c in CHUNKS}
    def search(self, query, *, filters=None, top_k=3):
        out = []
        for rank, cid in enumerate(self.order[:top_k], 1):
            c = self.by_id[cid]
            out.append({**c, "score": 1.0 / rank, "rank": rank, "strategy": "dense"})
        return out


class BM25Tests(unittest.TestCase):
    def test_exact_phrase_ranks_right_chunk_first(self):
        bm25 = BM25Retriever(CHUNKS)
        hits = bm25.search("watch waistcoat-pocket", top_k=3)
        self.assertEqual(hits[0]["id"], "c1")
        self.assertTrue(hits[0]["score"] > 0)
        self.assertEqual(hits[0]["rank"], 1)

    def test_filter_restricts_candidates(self):
        bm25 = BM25Retriever(CHUNKS)
        hits = bm25.search("glass", filters={"book_title": "Through the Looking-Glass"}, top_k=3)
        self.assertTrue(all(h["source_id"] == "pg12" for h in hits))


class FilterInferTests(unittest.TestCase):
    def test_infers_book_and_chapter(self):
        f = infer_filters("In chapter 5 of Through the Looking-Glass, what rule for jam?")
        self.assertEqual(f["book_title"], "Through the Looking-Glass")
        self.assertEqual(f["chapter_number"], 5)

    def test_roman_numeral_chapter(self):
        self.assertEqual(infer_filters("Chapter IV events")["chapter_number"], 4)

    def test_no_signal_returns_empty(self):
        self.assertEqual(infer_filters("What did the rabbit carry?"), {})


class HybridAndRerankTests(unittest.TestCase):
    def test_rrf_recovers_lexical_hit_dense_missed(self):
        # Dense ranks c2, c1 first and never returns c3; BM25 will surface c3.
        dense = StubDense(["c2", "c1", "c3"])
        bm25 = BM25Retriever(CHUNKS)
        hybrid = HybridRRFRetriever(dense, bm25, pool=3)
        hits = hybrid.search("Looking-glass book glass", top_k=3)
        ids = [h["id"] for h in hits]
        self.assertIn("c3", ids)
        self.assertEqual(hits[0]["rank"], 1)

    def test_rerank_promotes_lexical_match(self):
        # Dense puts the lexical match (c1) last; BM25 rerank should promote it.
        dense = StubDense(["c2", "c3", "c1"])
        bm25 = BM25Retriever(CHUNKS)
        rerank = RerankRetriever(dense, bm25, pool=3)
        hits = rerank.search("watch waistcoat-pocket", top_k=1)
        self.assertEqual(hits[0]["id"], "c1")


class TokenizeTests(unittest.TestCase):
    def test_casefold_and_split(self):
        self.assertEqual(tokenize("Watch, Waistcoat-Pocket!"), ["watch", "waistcoat", "pocket"])


if __name__ == "__main__":
    unittest.main()
