"""Local dense index: BAAI/bge-small-en-v1.5 (FastEmbed ONNX) + Qdrant.

Self-contained reuse of the Day 6 embedding model and chunks. `build` embeds
every chunk from data/day7/chunks.json and writes a persistent Qdrant collection
under .local/qdrant; `DenseEmbedder`/`QdrantDense` are reused by the retrievers.
Errors (missing model, zero vectors) propagate; no fake fallback.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day7 import config as C


def _finite_nonzero(vec: list[float]) -> bool:
    return len(vec) == C.EMBED_DIM and all(math.isfinite(v) for v in vec) and sum(v * v for v in vec) > 0


class DenseEmbedder:
    """Wraps FastEmbed. Documents are embedded as-is; queries get the bge prefix."""

    def __init__(self, cache_dir: Path | None = None, threads: int = 4):
        from fastembed import TextEmbedding
        self.model_name = C.EMBED_MODEL
        self.dimensions = C.EMBED_DIM
        cache = str(cache_dir or C.FASTEMBED_CACHE)
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=cache, threads=threads)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed empty document text")
        vectors = [v.tolist() for v in self.model.embed(texts, batch_size=32)]
        if len(vectors) != len(texts) or not all(_finite_nonzero(v) for v in vectors):
            raise ValueError("Invalid document embeddings produced")
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        vec = next(iter(self.model.embed([C.QUERY_PREFIX + question], batch_size=1))).tolist()
        if not _finite_nonzero(vec):
            raise ValueError("Invalid query embedding")
        return vec


class QdrantDense:
    """Thin wrapper over a persistent local Qdrant collection (cosine, exact)."""

    def __init__(self, path: Path | None = None):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=str(path or C.INDEX_DIR))
        self.collection = C.QDRANT_COLLECTION

    def close(self) -> None:
        self.client.close()

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def build(self, chunks: list[dict], embedder: DenseEmbedder) -> dict:
        from qdrant_client import models
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=C.EMBED_DIM, distance=models.Distance.COSINE),
        )
        vectors = embedder.embed_documents([c["text"] for c in chunks])
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "chunk_id": chunk["id"], "text": chunk["text"],
                "source_id": chunk["source_id"], "source": chunk["source"],
                "page": chunk["page"], "metadata": chunk.get("metadata", {}),
            }
            points.append(models.PointStruct(id=idx, vector=vector, payload=payload))
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start:start + 128], wait=True)
        count = self.count()
        if count != len(chunks):
            raise RuntimeError(f"Index count mismatch: {count} != {len(chunks)}")
        return {"collection": self.collection, "count": count, "dim": C.EMBED_DIM}

    def search(self, query_vector: list[float], top_k: int,
               filters: dict[str, Any] | None = None) -> list[dict]:
        from qdrant_client import models
        conditions = []
        for key, value in (filters or {}).items():
            if key not in C.FILTER_FIELDS:
                raise ValueError(f"Unsupported filter field: {key}")
            # Metadata fields are nested under payload['metadata'] except the
            # promoted document_id; map both forms.
            qkey = "source_id" if key == "document_id" else f"metadata.{key}"
            conditions.append(models.FieldCondition(key=qkey, match=models.MatchValue(value=value)))
        res = self.client.query_points(
            self.collection, query=query_vector, limit=top_k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            search_params=models.SearchParams(exact=True), with_payload=True,
        )
        hits = []
        for p in res.points:
            pay = dict(p.payload)
            hits.append({
                "id": pay["chunk_id"], "text": pay["text"],
                "source_id": pay["source_id"], "source": pay["source"],
                "page": pay["page"], "metadata": pay.get("metadata", {}),
                "score": float(p.score),
            })
        return hits


def main() -> None:
    chunks = json.loads(C.CHUNKS_JSON.read_text(encoding="utf-8"))
    embedder = DenseEmbedder()
    store = QdrantDense()
    try:
        info = store.build(chunks, embedder)
    finally:
        store.close()
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
