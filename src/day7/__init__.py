"""Day 7 — Agentic RAG over the Day 6 Alice corpus.

Public-domain corpus (Project Gutenberg #11 and #12), local ONNX embeddings
(BAAI/bge-small-en-v1.5 via FastEmbed) and a local Qdrant vector store, reused
from Day 6. The agent controller calls a real tool-calling model (Groq).
"""
