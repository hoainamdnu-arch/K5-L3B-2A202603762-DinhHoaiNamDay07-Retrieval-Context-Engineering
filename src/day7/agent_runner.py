"""Run the agentic RAG loop over the benchmark cases and save traces.

For each case: the Groq controller orchestrates `search`/`answer` over a chosen
retriever (default: Hybrid RRF, top_k=5, budget=3000, the dev-selected config).
Writes results/agent/<split>/<config>/traces.json and cases.json, shaped for
starter/agent_metrics.py. Provider/retriever errors are captured as status
'error', not silently replaced.

Usage:
  python src/day7/agent_runner.py --split dev --config hybrid
  python src/day7/agent_runner.py --split test --config baseline_bm25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from agentic import run_agent

from day7 import config as C
from day7.dense_index import DenseEmbedder, QdrantDense
from day7.groq_controller import GroqController
from day7.retrievers import (BM25Retriever, DenseRetriever, HybridRRFRetriever)


# Frozen agent retrieval configs (chosen on dev before any test run).
AGENT_CONFIGS = {
    # Chosen: Hybrid RRF (selected on dev retrieval metrics), top_k=5. Agent
    # budget raised 3000 -> 4000 to fit ~3 chunks so evidence from later turns is
    # not crowded out by earlier mediocre hits (observed on dev).
    "hybrid":        {"retriever": "hybrid_rrf", "top_k": 5, "budget": 4000, "max_searches": 4},
    "baseline_bm25": {"retriever": "bm25",       "top_k": 3, "budget": 800,  "max_searches": 4},
    "dense":         {"retriever": "dense",      "top_k": 5, "budget": 4000, "max_searches": 4},
}


def make_retriever(kind, chunks, embedder, store):
    dense = DenseRetriever(embedder, store)
    bm25 = BM25Retriever(chunks)
    if kind == "dense":
        return dense
    if kind == "bm25":
        return bm25
    if kind == "hybrid_rrf":
        return HybridRRFRetriever(dense, bm25)
    raise ValueError(f"Unknown retriever {kind}")


def final_context(events: list[dict]) -> dict:
    searches = [e for e in events if e.get("event") == "search"]
    if not searches:
        return {"context": "", "context_ids": []}
    last = searches[-1]
    return {"context": last.get("context", ""), "context_ids": last.get("context_ids", [])}


def run_case(case, retriever, controller, cfg):
    start = time.perf_counter()
    calls0 = controller.calls
    tok0 = (controller.total_prompt_tokens, controller.total_completion_tokens)
    try:
        result = run_agent(case["question"], controller, retriever,
                           filters=case.get("filter"), max_searches=cfg["max_searches"],
                           top_k=cfg["top_k"], budget=cfg["budget"])
        status = result["status"]
        events = result["events"]
        answer = result["answer"]
        citations = result["citations"]
        error = None
    except Exception as exc:  # provider / retriever failure
        status, events, answer, citations = "error", [], "", []
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    elapsed = (time.perf_counter() - start) * 1000
    n_searches = sum(1 for e in events if e.get("event") == "search")
    fc = final_context(events)
    trace = {
        "query_id": case["id"], "split": case["split"], "answerable": case["answerable"],
        "config": cfg["name"], "model": controller.model, "status": status,
        "question": case["question"], "answer": answer, "citations": citations,
        "n_searches": n_searches, "model_calls": controller.calls - calls0,
        "events": events, "final_context": fc["context"], "final_context_ids": fc["context_ids"],
        "elapsed_ms": round(elapsed, 2),
        "prompt_tokens": controller.total_prompt_tokens - tok0[0],
        "completion_tokens": controller.total_completion_tokens - tok0[1],
        "error": error,
    }
    return trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--config", choices=list(AGENT_CONFIGS), default="hybrid")
    ap.add_argument("--limit", type=int, default=None, help="run only first N cases (debug)")
    args = ap.parse_args()

    C.load_dotenv()
    cfg = {"name": args.config, **AGENT_CONFIGS[args.config]}
    chunks = json.loads(C.CHUNKS_JSON.read_text(encoding="utf-8"))
    cases = [c for c in json.loads(C.AGENT_CASES_JSON.read_text(encoding="utf-8"))
             if c["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]

    embedder = DenseEmbedder()
    store = QdrantDense()
    controller = GroqController()
    out_dir = C.RESULTS_DIR / "agent" / args.split / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    try:
        retriever = make_retriever(cfg["retriever"], chunks, embedder, store)
        for case in cases:
            trace = run_case(case, retriever, controller, cfg)
            traces.append(trace)
            print(f"{case['id']:4} [{trace['status']:8}] searches={trace['n_searches']} "
                  f"calls={trace['model_calls']} ans={trace['answer'][:60]!r}")
    finally:
        store.close()
        controller.close()

    # Trace file shaped for agent_metrics.py (needs query_id/status/events/citations).
    metrics_traces = [{"query_id": t["query_id"], "status": t["status"],
                       "events": t["events"], "citations": t["citations"]} for t in traces]
    agent_cases = [{"id": c["id"], "split": c["split"], "answerable": c["answerable"]} for c in cases]
    (out_dir / "traces.json").write_text(json.dumps(metrics_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "cases.json").write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "full_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    totals = {
        "split": args.split, "config": args.config, "model": controller.model,
        "cases": len(cases), "model_calls": controller.calls,
        "prompt_tokens": controller.total_prompt_tokens,
        "completion_tokens": controller.total_completion_tokens,
        "statuses": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
