"""Paths and shared configuration for the Day 7 pipeline.

Everything resolves relative to the repo root so the pipeline runs from a clone
without hard-coded absolute paths. The Day 6 project is located as a sibling
directory (override with the DAY6_ROOT environment variable).
"""
from __future__ import annotations

import os
from pathlib import Path

# src/day7/config.py -> repo root is two parents up from this file's folder.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data" / "day7"
RESULTS_DIR = REPO_ROOT / "results"
INDEX_DIR = REPO_ROOT / ".local" / "qdrant"

CHUNKS_JSON = DATA_DIR / "chunks.json"
QUESTIONS_JSON = DATA_DIR / "questions.json"
AGENT_CASES_JSON = DATA_DIR / "agent_cases.json"

# ---- Day 6 reuse -----------------------------------------------------------
# The Day 6 repo holds the raw corpus, the chosen chunk export (strategy C) and
# the FastEmbed model cache. We reuse its chunks/IDs/sources and embedding model.
_DEFAULT_DAY6 = REPO_ROOT.parent / "K5-L3B-DinhHoaiNam-2A202603762Day06-DataIngestionEngineering"
DAY6_ROOT = Path(os.environ.get("DAY6_ROOT", str(_DEFAULT_DAY6)))
DAY6_CHUNKS = DAY6_ROOT / "artifacts" / "chunks_C.jsonl"      # selected strategy C
DAY6_DEV = DAY6_ROOT / "benchmark" / "dev.jsonl"
DAY6_TEST = DAY6_ROOT / "benchmark" / "test.jsonl"
DAY6_MANIFEST = DAY6_ROOT / "data" / "manifest.json"
# Reuse the already-downloaded ONNX model to avoid a network fetch; falls back
# to a Day 7-local cache dir if the Day 6 cache is absent.
FASTEMBED_CACHE = Path(os.environ.get(
    "FASTEMBED_CACHE",
    str(DAY6_ROOT / ".cache" / "fastembed")))
if not FASTEMBED_CACHE.exists():
    FASTEMBED_CACHE = REPO_ROOT / ".cache" / "fastembed"

# ---- Embedding model (identical to Day 6) ----------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# bge asymmetric prefix: queries are prefixed, documents are not. Matches Day 6.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QDRANT_COLLECTION = "day7_alice_bge_small"

# ---- Metadata filter fields the dense store understands --------------------
FILTER_FIELDS = ("book_title", "chapter_number", "section", "language", "document_id")

# ---- Agent model (Groq, OpenAI-compatible REST API) ------------------------
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY_ENV = "GROQ_API_KEY"


def require_groq_key() -> str:
    key = os.environ.get(GROQ_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{GROQ_API_KEY_ENV} is not set. Put it in the local .env (git-ignored) "
            "or export it before running the agent.")
    return key


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no external dependency). Values already in the real
    environment win, so an exported key is never overwritten by the file."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
