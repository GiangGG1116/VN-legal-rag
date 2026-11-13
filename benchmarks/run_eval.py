# -*- coding: utf-8 -*-
"""Benchmark harness — runs the golden set (and optional generated set)
through the full RAG pipeline and scores action accuracy + fact coverage.

Usage:
    python -m benchmarks.run_eval [--set golden] [--json out.json]

Exit code: 0 if all items pass, 1 otherwise (usable as a CI gate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_law.corpus import Corpus
from rag_law.pipeline import RAGPipeline
from rag_law.retrieval import LexicalRetriever
from rag_law.utils import fold

HERE = Path(__file__).resolve().parent

# expected_action (golden) -> acceptable pipeline actions
ACTION_ALIASES = {
    "ANSWER": {"answer", "answer_with_caution"},
    "ANSWER_WITH_CAUTION": {"answer_with_caution"},
    "REFUSE_SCOPE": {"refuse_scope"},
    "REFUSE_NON_LEGAL": {"refuse_non_legal"},
    "REFUSE_NO_DATA": {"refuse_no_data"},
    "REFUSE_INJECTION": {"refuse_injection"},
}


@dataclass
class ItemResult:
    item_id: str
    ok: bool
    expected: str
    actual: str
    failed_facts: list[str]
    answer_preview: str
    reasons: list[str]


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_item(item: dict, result) -> ItemResult:
    expected = item["expected_action"]
    allowed = ACTION_ALIASES.get(expected, {expected.lower()})
    action_ok = result.action in allowed
    facts = [fold(f) for f in item.get("answer_facts", [])]
    folded_answer = fold(result.answer)
    failed = [f for f in facts if f not in folded_answer]
    ok = action_ok and not failed
    reasons = []
    if not action_ok:
        reasons.append(f"action {result.action!r} not in {sorted(allowed)}")
    if failed:
        reasons.append(f"missing facts: {failed}")
    return ItemResult(
        item_id=item["id"],
        ok=ok,
        expected=expected,
        actual=result.action,
        failed_facts=failed,
        answer_preview=result.answer[:200].replace("\n", " | "),
        reasons=reasons,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="golden", choices=["golden", "generated", "all"])
    ap.add_argument("--json", default=None, help="write full report JSON here")
    args = ap.parse_args()

    t0 = time.time()
    corpus = Corpus()
    retriever = LexicalRetriever(corpus)
    (HERE / "runs").mkdir(exist_ok=True)
    pipeline = RAGPipeline(corpus, retriever, audit_path=str(HERE / "runs" / "audit.jsonl"))
    print(f"Corpus: {len(corpus.docs)} docs | article index built in {time.time()-t0:.1f}s")

    paths = []
    if args.set in ("golden", "all"):
        paths.append(HERE / "data" / "golden_seed.jsonl")
    if args.set in ("generated", "all"):
        p = HERE / "data" / "generated_candidates.jsonl"
        if p.exists():
            paths.append(p)
        else:
            print("(no generated_candidates.jsonl — run make_generated.py first)")

    all_items: list[ItemResult] = []
    for p in paths:
        items = load_items(p)
        print(f"\n=== {p.name}: {len(items)} items ===")
        for item in items:
            res = pipeline.handle(item["question"])
            ir = score_item(item, res)
            all_items.append(ir)
            mark = "PASS" if ir.ok else "FAIL"
            print(f"[{mark}] {ir.item_id:<10} expect={ir.expected:<20} got={ir.actual:<20} {(' '.join(ir.reasons)) if ir.reasons else ''}")
            if not ir.ok:
                print(f"        answer: {ir.answer_preview}")

    total = len(all_items)
    passed = sum(1 for r in all_items if r.ok)
    by_action: dict[str, tuple[int, int]] = {}
    for r in all_items:
        tot, ok = by_action.get(r.expected, (0, 0))
        by_action[r.expected] = (tot + 1, ok + (1 if r.ok else 0))

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} passed ({passed*100//max(1,total)}%)")
    for k, (t, ok) in sorted(by_action.items()):
        print(f"  {k:<20} {ok}/{t}")
    if args.json:
        report = {
            "total": total,
            "passed": passed,
            "by_action": {k: {"total": t, "passed": ok} for k, (t, ok) in by_action.items()},
            "items": [r.__dict__ for r in all_items],
        }
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.json}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
