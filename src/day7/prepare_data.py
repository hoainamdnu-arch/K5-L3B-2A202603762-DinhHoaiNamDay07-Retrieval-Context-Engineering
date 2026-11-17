"""Build the Day 7 benchmark from the frozen Day 6 artifacts.

Outputs (all JSON arrays, UTF-8) under data/day7/:
  * chunks.json        — Day 6 strategy-C chunks in the Day 7 chunk schema.
  * questions.json     — answerable questions (dev + test) with `gold` evidence.
  * agent_cases.json   — every case incl. insufficient-evidence ones, with
                         `answerable` and `expected_behavior`.

Nothing here infers gold from answers or inspects held-out evidence to tune a
system; it only reshapes the frozen Day 6 benchmark and adds authored
insufficient-evidence probes whose absence is verified against the raw corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day7 import config as C


# ---------------------------------------------------------------------------
# Chunk conversion: Day 6 export -> Day 7 chunk schema.
# ---------------------------------------------------------------------------
def _readable_id(raw: dict, counter: dict) -> str:
    """Short, stable, human-readable chunk id so a model can cite it verbatim.

    Form: <page_id>_c<n>, e.g. pg11_u001_c1. Falls back to doc/page if no
    page_id. The original Day 6 sha256 id is preserved in metadata.day6_chunk_id.
    """
    meta = raw.get("metadata", {})
    base = meta.get("page_id") or f"{raw['document_id']}_p{raw['page']}"
    base = base.replace(":", "_")
    counter[base] = counter.get(base, 0) + 1
    return f"{base}_c{counter[base]}"


def convert_chunk(raw: dict, counter: dict) -> dict:
    meta = dict(raw.get("metadata", {}))
    meta["day6_chunk_id"] = raw["chunk_id"]
    return {
        "id": _readable_id(raw, counter),
        "text": raw["text"],
        "source_id": raw["document_id"],
        "source": raw["source"],
        "page": raw["page"],
        "metadata": meta,
    }


def load_day6_chunks() -> list[dict]:
    rows = [json.loads(l) for l in C.DAY6_CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    counter: dict = {}
    return [convert_chunk(r, counter) for r in rows]


# ---------------------------------------------------------------------------
# Gold evidence overrides: three Day 6 questions have evidence that straddles a
# strategy-C chunk boundary. We replace the single long quote with verified
# single-chunk substrings (recorded as alternative gold evidences). Each entry
# below is checked at build time to be a substring of some chunk on the right
# source/page, so no gold spans two chunks.
# ---------------------------------------------------------------------------
GOLD_OVERRIDES: dict[str, list[str]] = {
    # Answer has two parts (pebbles -> little cakes; eating one -> shrinking);
    # each part lives in a separate chunk, so both are listed as valid evidence.
    "D04": [
        "the pebbles were all turning into little cakes",
        "she began shrinking directly",
    ],
    # "too much pepper in the soup" and "too much of it in the air" split chunks.
    "D06": [
        "too much pepper in that soup",
        "too much of it in the air",
    ],
    # The pay-off line "Bread-and-butter" sits in the second chunk of the page.
    "T04": [
        "Bread-and-butter, of course",
    ],
}


def convert_question(raw: dict) -> dict:
    qid = raw["id"]
    if qid in GOLD_OVERRIDES:
        gold = [{"source_id": raw["gold_source_id"], "page": raw["gold_page"], "evidence": ev}
                for ev in GOLD_OVERRIDES[qid]]
    else:
        gold = [{"source_id": raw["gold_source_id"], "page": raw["gold_page"],
                 "evidence": raw["evidence_quote"]}]
    out = {
        "id": qid,
        "split": raw["split"],
        "question": raw["question"],
        "expected_answer": raw["expected_answer"],
        "gold": gold,
        "type": raw.get("type"),
    }
    if raw.get("filter"):
        out["filter"] = raw["filter"]
    return out


def load_day6_questions() -> list[dict]:
    out = []
    for path in (C.DAY6_DEV, C.DAY6_TEST):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(convert_question(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Insufficient-evidence probes: near-topic questions whose answers are NOT in
# the indexed text. Absence was verified against data/normalized/pg11+pg12.txt
# (see README / report). These never carry gold; the agent must abstain.
# ---------------------------------------------------------------------------
INSUFFICIENT_EVIDENCE = [
    {
        "id": "U01", "split": "dev",
        "question": "What price is printed on the paper label tucked into the Mad Hatter's hat?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": ("The famous '10/6' price tag appears only in Tenniel's illustration of "
                           "the Hatter, never in the narrative text. Strings '10/6' and 'ten and "
                           "six' do not occur in the normalized corpus."),
    },
    {
        "id": "U02", "split": "dev",
        "question": "What is Alice's surname (family name) in the two Alice books?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": ("Alice is only ever called 'Alice'. The words 'surname' and 'last name' "
                           "do not occur and no family name is given anywhere in the text."),
    },
    {
        "id": "U03", "split": "test",
        "question": "How much money does the White Rabbit's watch cost?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": ("The watch is described as taken from a waistcoat pocket but no monetary "
                           "value or price is ever attached to it in the text."),
    },
    {
        "id": "U04", "split": "test",
        "question": "What is Alice's normal height in inches when she is neither enlarged nor shrunk?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": ("Specific heights are only given for her transformed states (e.g. 'ten "
                           "inches high', 'nine feet high', 'three inches high'); her ordinary "
                           "height is never stated numerically."),
    },
]


def build_agent_cases(answerable_qs: list[dict]) -> list[dict]:
    cases = []
    for q in answerable_qs:
        cases.append({"id": q["id"], "split": q["split"], "question": q["question"],
                      "answerable": True, "expected_behavior": "search_and_answer"})
    for u in INSUFFICIENT_EVIDENCE:
        cases.append({"id": u["id"], "split": u["split"], "question": u["question"],
                      "answerable": False, "expected_behavior": u["expected_behavior"],
                      "absence_reason": u["absence_reason"]})
    return cases


def verify_gold(chunks: list[dict], questions: list[dict]) -> None:
    """Every gold evidence must sit wholly inside one chunk of the right source/page."""
    by_dp: dict[tuple, list[dict]] = {}
    for c in chunks:
        by_dp.setdefault((c["source_id"], c["page"]), []).append(c)
    problems = []
    for q in questions:
        for g in q["gold"]:
            cand = by_dp.get((g["source_id"], g["page"]), [])
            if not any(g["evidence"] in c["text"] for c in cand):
                problems.append((q["id"], g["evidence"][:50]))
    if problems:
        raise SystemExit(f"Gold evidence not contained in a single chunk: {problems}")


def verify_absence(absent: list[dict]) -> None:
    """Sanity-check the insufficient-evidence probes against the raw corpus."""
    texts = []
    for name in ("pg11.txt", "pg12.txt"):
        p = C.DAY6_ROOT / "data" / "normalized" / name
        if p.exists():
            texts.append(p.read_text(encoding="utf-8").lower())
    blob = "\n".join(texts)
    if not blob:
        print("  (normalized corpus not found; skipping absence recheck)")
        return
    forbidden = {"U01": ["10/6", "ten and six"], "U02": ["surname", "last name"]}
    for u in absent:
        for token in forbidden.get(u["id"], []):
            if token in blob:
                raise SystemExit(f"{u['id']}: expected-absent token {token!r} found in corpus")


def main() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_day6_chunks()
    questions = load_day6_questions()
    verify_gold(chunks, questions)
    verify_absence(INSUFFICIENT_EVIDENCE)
    cases = build_agent_cases(questions)

    C.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    C.QUESTIONS_JSON.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    C.AGENT_CASES_JSON.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dev_ans = sum(q["split"] == "dev" for q in questions)
    test_ans = sum(q["split"] == "test" for q in questions)
    dev_un = sum(u["split"] == "dev" for u in INSUFFICIENT_EVIDENCE)
    test_un = sum(u["split"] == "test" for u in INSUFFICIENT_EVIDENCE)
    print(json.dumps({
        "chunks": len(chunks),
        "answerable": {"dev": dev_ans, "test": test_ans},
        "insufficient_evidence": {"dev": dev_un, "test": test_un},
        "cases": len(cases),
        "outputs": [str(C.CHUNKS_JSON), str(C.QUESTIONS_JSON), str(C.AGENT_CASES_JSON)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
