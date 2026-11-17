"""Benchmark the retrieval methods/configs on a split and emit comparable metrics.

Metrics (answerable questions only, per BENCHMARK.md):
  * Hit@1             — gold evidence inside the rank-1 chunk.
  * Evidence Hit@3    — gold evidence inside the top-3 chunks.
  * Context Evidence Rate — gold evidence inside the packed context sent to the
                            model (pack_context with the config's budget).
Also reports Recall@k / MRR, median latency and n/N for every config. Results
go to results/retrieval/<split>/ as traces.json + summary.json, plus a combined
comparison.json.

Usage:
  python src/day7/eval_retrieval.py --split dev
  python src/day7/eval_retrieval.py --split test --configs dense bm25 dense_rerank_bm25
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from baseline import hit, pack_context  # starter helpers

from day7 import config as C
from day7.dense_index import DenseEmbedder, QdrantDense
from day7.retrievers import (BM25Retriever, DenseRetriever, FilterInferRetriever,
                             HybridRRFRetriever, RerankRetriever)


def build_registry(chunks: list[dict], embedder: DenseEmbedder, store: QdrantDense) -> dict:
    dense = DenseRetriever(embedder, store)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense, bm25)
    dense_filter = FilterInferRetriever(dense)
    rerank = RerankRetriever(dense, bm25)
    hybrid_wide = HybridRRFRetriever(dense, bm25, pool=10)
    # (name -> (retriever, top_k, budget, note)).
    # The three base methods use the starter default 800-char budget for an
    # apples-to-apples ranking comparison; the improvement configs raise the
    # budget to 3000 so the ~1.4k-char evidence chunk can actually fit.
    return {
        "dense":             (dense,        3,  800,  "Dense vectors (bge-small) on the Day 6 store"),
        "bm25":              (bm25,         3,  800,  "Okapi BM25 lexical baseline"),
        "hybrid_rrf":        (hybrid,       3,  800,  "RRF fusion of dense + BM25"),
        "dense_filter":      (dense_filter, 3, 3000,  "Dense + question-inferred metadata filter, budget 3000"),
        "dense_rerank_bm25": (rerank,       3, 3000,  "Dense over-fetch 12 -> BM25 rerank -> top 3, budget 3000"),
        "hybrid_wide_ctx":   (hybrid_wide,  5, 3000,  "Hybrid RRF, top_k=5, wider 3000-char context"),
    }


def evaluate_config(name, retriever, top_k, budget, note, questions):
    traces, hit1, hit3, ctx_hit, mrr_sum, recalls, latencies = [], 0, 0, 0, 0.0, [], []
    for q in questions:
        flt = q.get("filter")
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=flt, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        marked = [{**r, "hit": hit(r, q["gold"])} for r in results]
        context, ctx_ids = pack_context(marked, budget=budget)
        h1 = bool(marked and marked[0]["hit"])
        h3 = any(m["hit"] for m in marked[:3])
        c_hit = any(m["hit"] and m["id"] in ctx_ids for m in marked)
        first_hit_rank = next((i + 1 for i, m in enumerate(marked) if m["hit"]), None)
        rr = 1.0 / first_hit_rank if first_hit_rank else 0.0
        hit1 += h1; hit3 += h3; ctx_hit += c_hit; mrr_sum += rr
        recalls.append(1.0 if any(m["hit"] for m in marked) else 0.0)
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q.get("type"),
            "question": q["question"], "config": name, "top_k": top_k, "budget_chars": budget,
            "filter": flt, "gold": q["gold"],
            "results": [{"id": m["id"], "page": m["page"], "source_id": m["source_id"],
                         "score": round(m["score"], 5), "rank": m["rank"], "hit": m["hit"],
                         "preview": m["text"][:90].replace("\n", " ")} for m in marked],
            "context_ids": ctx_ids, "context_chars": len(context),
            "hit1": h1, "hit3": h3, "context_hit": c_hit, "reciprocal_rank": rr,
            "elapsed_ms": round(elapsed, 2),
        })
    n = len(questions)
    summary = {
        "config": name, "note": note, "top_k": top_k, "budget_chars": budget, "n": n,
        "hit1": {"n": hit1, "N": n, "rate": round(hit1 / n, 4) if n else None},
        "evidence_hit3": {"n": hit3, "N": n, "rate": round(hit3 / n, 4) if n else None},
        "context_evidence_rate": {"n": ctx_hit, "N": n, "rate": round(ctx_hit / n, 4) if n else None},
        "recall_at_k": round(statistics.mean(recalls), 4) if recalls else None,
        "mrr": round(mrr_sum / n, 4) if n else None,
        "median_ms": round(statistics.median(latencies), 2) if latencies else None,
    }
    return summary, traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="*", default=None, help="subset of config names; default all")
    args = ap.parse_args()

    chunks = json.loads(C.CHUNKS_JSON.read_text(encoding="utf-8"))
    questions = [q for q in json.loads(C.QUESTIONS_JSON.read_text(encoding="utf-8"))
                 if q["split"] == args.split]
    if not questions:
        raise SystemExit(f"No answerable questions in split {args.split}")

    embedder = DenseEmbedder()
    store = QdrantDense()
    out_dir = C.RESULTS_DIR / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        registry = build_registry(chunks, embedder, store)
        names = args.configs or list(registry)
        comparison = []
        for name in names:
            if name not in registry:
                raise SystemExit(f"Unknown config: {name}")
            retriever, top_k, budget, note = registry[name]
            summary, traces = evaluate_config(name, retriever, top_k, budget, note, questions)
            (out_dir / f"{name}.traces.json").write_text(
                json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out_dir / f"{name}.summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            comparison.append(summary)
            print(f"{name:20} hit@1={summary['hit1']['rate']} "
                  f"ev_hit@3={summary['evidence_hit3']['rate']} "
                  f"ctx_rate={summary['context_evidence_rate']['rate']} "
                  f"mrr={summary['mrr']} med_ms={summary['median_ms']}")
        (out_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        store.close()


if __name__ == "__main__":
    main()
