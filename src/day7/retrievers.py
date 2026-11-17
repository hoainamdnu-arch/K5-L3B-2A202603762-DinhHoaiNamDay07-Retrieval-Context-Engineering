"""Retrieval methods for Day 7.

Three distinct methods, all returning the same record shape
(id, text, source_id, source, page, metadata, score, rank, strategy):

  * DenseRetriever     — bge-small-en-v1.5 vectors in the Day 6 Qdrant store.
  * BM25Retriever      — Okapi BM25 over the same chunks (pure stdlib).
  * HybridRRFRetriever — Reciprocal Rank Fusion of dense + BM25 rankings.

Plus composable improvements used as separate configs:
  * FilterInferRetriever — infers book_title / chapter_number from the question
                           and applies them as a metadata filter (dense).
  * RerankRetriever      — over-fetch N candidates, re-score with a BM25 lexical
                           pass, keep top_k (first-stage dense, second-stage lex).

Filters are only inferred from the user's own question text, never from gold.
"""
from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day7 import config as C
from day7.dense_index import DenseEmbedder, QdrantDense


# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _tag(records: list[dict], strategy: str) -> list[dict]:
    out = []
    for rank, r in enumerate(records, start=1):
        out.append({**r, "rank": rank, "strategy": strategy})
    return out


class Retriever(Protocol):
    name: str
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict]: ...


# ---------------------------------------------------------------------------
class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: DenseEmbedder, store: QdrantDense):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        qv = self.embedder.embed_query(query)
        hits = self.store.search(qv, top_k=top_k, filters=filters)
        return _tag(hits, self.name)


# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over the chunk corpus."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c["text"]) for c in chunks]
        self.freqs = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for f in self.freqs for term in f)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_terms: list[str], idx: int) -> float:
        f = self.freqs[idx]
        dl = self.lengths[idx]
        s = 0.0
        for term in query_terms:
            if term not in f:
                continue
            idf = self.idf.get(term, 0.0)
            tf = f[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return s

    def rank(self, query: str, candidate_idx: list[int] | None = None) -> list[tuple[int, float]]:
        terms = tokenize(query)
        idxs = candidate_idx if candidate_idx is not None else range(len(self.chunks))
        scored = [(i, self.score(terms, i)) for i in idxs]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda t: (-t[1], self.chunks[t[0]]["id"]))
        return scored


def _record(chunk: dict, score: float) -> dict:
    return {
        "id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"],
        "source": chunk["source"], "page": chunk["page"],
        "metadata": chunk.get("metadata", {}), "score": float(score),
    }


def _passes_filter(chunk: dict, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for key, value in filters.items():
        got = chunk["source_id"] if key == "document_id" else meta.get(key)
        if got != value:
            return False
    return True


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.index = BM25Index(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        cand = [i for i, c in enumerate(self.chunks) if _passes_filter(c, filters)]
        ranked = self.index.rank(query, candidate_idx=cand)[:top_k]
        return _tag([_record(self.chunks[i], s) for i, s in ranked], self.name)


# ---------------------------------------------------------------------------
class HybridRRFRetriever:
    """Reciprocal Rank Fusion of the dense and BM25 rankings.

    RRF score = sum over retrievers of 1 / (k + rank). Rank-based fusion needs no
    score normalization across the two incomparable score scales.
    """
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60, pool: int = 10):
        self.dense = dense
        self.bm25 = bm25
        self.k = k
        self.pool = pool

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.pool)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=self.pool)
        scores: dict[str, float] = {}
        record: dict[str, dict] = {}
        for hits in (dense_hits, bm25_hits):
            for rank, h in enumerate(hits, start=1):
                scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (self.k + rank)
                record.setdefault(h["id"], h)
        ordered = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:top_k]
        fused = [{**record[cid], "score": sc} for cid, sc in ordered]
        return _tag(fused, self.name)


# ---------------------------------------------------------------------------
# Improvement config: query-driven metadata filter inference.
# ---------------------------------------------------------------------------
_BOOK_PATTERNS = [
    (re.compile(r"\bwonderland\b|adventures in wonderland", re.I), "Alice's Adventures in Wonderland"),
    (re.compile(r"looking[- ]glass|through the looking", re.I), "Through the Looking-Glass"),
]
_CHAPTER_RE = re.compile(r"\bchapter\s+(\d+|[ivxlc]+)\b", re.I)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
          "ix": 9, "x": 10, "xi": 11, "xii": 12}


def infer_filters(question: str) -> dict[str, Any]:
    """Infer a metadata filter purely from the user's question wording."""
    filters: dict[str, Any] = {}
    for pat, title in _BOOK_PATTERNS:
        if pat.search(question):
            filters["book_title"] = title
            break
    m = _CHAPTER_RE.search(question)
    if m:
        tok = m.group(1).lower()
        num = int(tok) if tok.isdigit() else _ROMAN.get(tok)
        if num:
            filters["chapter_number"] = num
    return filters


class FilterInferRetriever:
    """Dense retrieval + metadata filter inferred from the question. If the
    filtered search returns nothing, fall back to the unfiltered dense search so
    a wrong inference cannot zero out recall."""
    name = "dense_filter"

    def __init__(self, dense: DenseRetriever):
        self.dense = dense

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        inferred = dict(filters or {})
        inferred.update(infer_filters(query))
        if inferred:
            hits = self.dense.search(query, filters=inferred, top_k=top_k)
            if hits:
                return _tag(hits, self.name)
        return _tag(self.dense.search(query, filters=filters, top_k=top_k), self.name)


# ---------------------------------------------------------------------------
# Improvement config: two-stage retrieve-then-rerank.
# ---------------------------------------------------------------------------
class RerankRetriever:
    """First stage: dense over-fetch `pool` candidates. Second stage: re-score
    those candidates with BM25 and keep top_k. Compares ranking before/after."""
    name = "dense_rerank_bm25"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, pool: int = 12):
        self.dense = dense
        self.bm25 = bm25
        self.pool = pool
        self._idx_by_id = {c["id"]: i for i, c in enumerate(bm25.chunks)}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        candidates = self.dense.search(query, filters=filters, top_k=self.pool)
        terms = tokenize(query)
        rescored = []
        for h in candidates:
            idx = self._idx_by_id.get(h["id"])
            lex = self.bm25.index.score(terms, idx) if idx is not None else 0.0
            # Blend: lexical primary, dense score as tie-break signal.
            rescored.append((h, lex, h["score"]))
        rescored.sort(key=lambda t: (-t[1], -t[2], t[0]["id"]))
        top = [{**h, "score": lex if lex > 0 else dense_s} for h, lex, dense_s in rescored[:top_k]]
        return _tag(top, self.name)
