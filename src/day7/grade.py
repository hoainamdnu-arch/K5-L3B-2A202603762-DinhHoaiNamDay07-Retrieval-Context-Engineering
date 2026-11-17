"""Build a human-grading packet and a reviews.json skeleton from agent traces.

The grader (a person, or the author auditing) reads each case's answer, the
citations, the gold evidence and the final context, then fills:
  * answer_grade (0/1/2) for answerable cases,
  * citation_support: one boolean per citation (does that chunk actually support
    the claim?),
  * abstention_correct (bool) for insufficient-evidence cases.
The skeleton pre-lists citations so the booleans line up; grades are left null
until a human sets them. agent_metrics.py refuses null grades, so an ungraded
file cannot be mistaken for a scored one.

Usage:
  python src/day7/grade.py --run results/agent/test/hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day7 import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="results/agent/<split>/<config> dir")
    args = ap.parse_args()

    full = json.loads((args.run / "full_traces.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in json.loads(C.QUESTIONS_JSON.read_text(encoding="utf-8"))}
    cases = {c["id"]: c for c in json.loads(C.AGENT_CASES_JSON.read_text(encoding="utf-8"))}

    skeleton, packet = [], ["# Grading packet", ""]
    for t in full:
        qid = t["query_id"]
        case = cases.get(qid, {})
        answerable = case.get("answerable")
        review = {"query_id": qid,
                  "citation_support": [None] * len(t["citations"])}
        if answerable:
            review["answer_grade"] = None
        else:
            review["abstention_correct"] = None
        skeleton.append(review)

        packet.append(f"## {qid} ({'answerable' if answerable else 'insufficient-evidence'}) "
                      f"— status={t['status']}")
        packet.append(f"- **Q:** {t['question']}")
        if answerable:
            q = questions.get(qid, {})
            packet.append(f"- **Expected:** {q.get('expected_answer')}")
            for g in q.get("gold", []):
                packet.append(f"- **Gold evidence** (p{g['page']}): {g['evidence']}")
        else:
            packet.append(f"- **Absence reason:** {case.get('absence_reason')}")
        packet.append(f"- **Agent answer:** {t['answer']}")
        packet.append(f"- **Citations:** {t['citations']}")
        packet.append(f"- **Searches:** {t['n_searches']} | model_calls={t['model_calls']} | "
                      f"status={t['status']} | error={t['error']}")
        ctx = t["final_context"].replace("\n", " ")
        packet.append(f"- **Final context ids:** {t['final_context_ids']}")
        packet.append(f"- **Final context:** {ctx[:1200]}")
        packet.append("")

    (args.run / "reviews.skeleton.json").write_text(
        json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run / "grading_packet.md").write_text("\n".join(packet), encoding="utf-8")
    print(f"Wrote {args.run / 'reviews.skeleton.json'} and grading_packet.md "
          f"({len(skeleton)} cases). Fill grades -> reviews.json.")


if __name__ == "__main__":
    main()
