# -*- coding: utf-8 -*-
"""Shared helpers for the 3-tier benchmark.

Tier 1 — Retrieval   : question -> chunk   | every commit  | ~0 cost (no LLM)
Tier 2 — End-to-end  : question -> answer  | before major changes (run_eval.py)
Tier 3 — Regression  : pinned production bugs | every commit | low cost
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_law.classify import QuestionClassifier
from rag_law.corpus import Corpus
from rag_law.pipeline import RAGPipeline
from rag_law.retrieval import LexicalRetriever
from rag_law.utils import fold

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"    # input datasets: golden_seed, regressions, generated candidates
RUNS = HERE / "runs"    # run artifacts: audit JSONL, --json reports (not source)

# expected_action (jsonl) -> acceptable pipeline actions
ACTION_ALIASES = {
    "ANSWER": {"answer", "answer_with_caution"},
    "ANSWER_WITH_CAUTION": {"answer_with_caution"},
    "REFUSE_SCOPE": {"refuse_scope"},
    "REFUSE_NON_LEGAL": {"refuse_non_legal"},
    "REFUSE_NO_DATA": {"refuse_no_data"},
    "REFUSE_INJECTION": {"refuse_injection"},
}

RETRIEVAL_ACTIONS = {"ANSWER", "ANSWER_WITH_CAUTION"}


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------- tier 1
@dataclass
class RetrResult:
    item_id: str
    kind: str              # recall | no_cover | skip
    ok: bool
    rank: int | None = None
    detail: str = ""


def _retrieve_for(corpus: Corpus, classifier: QuestionClassifier,
                  retriever: LexicalRetriever, q: str):
    """Exact production call (see RAGPipeline.handle)."""
    cls = classifier.classify(q)
    return retriever.retrieve(q, cls.in_scope_refs or cls.law_refs or None,
                              cls.law_refs or None)


def run_retrieval_tier(corpus: Corpus, retriever: LexicalRetriever,
                       items: list[dict], top_k: int = 5) -> list[RetrResult]:
    """Tier 1: verify the expected (doc, Điều) is in top-k — or, for
    REFUSE_NO_DATA items, that the asked article has NO covering hit (the
    retrieval-level ground for refusal). No answer generation, no LLM."""
    classifier = QuestionClassifier(corpus)
    results: list[RetrResult] = []
    for item in items:
        doc = item.get("expected_doc")
        Dieu = item.get("expected_dieu")
        action = item["expected_action"]
        if action in RETRIEVAL_ACTIONS and doc and Dieu:
            hits = _retrieve_for(corpus, classifier, retriever, item["question"])[:top_k]
            rank = next((i + 1 for i, h in enumerate(hits)
                         if h.doc_num == doc and h.number == Dieu), None)
            results.append(RetrResult(
                item["id"], "recall", rank is not None, rank,
                f"expect {doc} Dieu {Dieu}"
                + (f" -> rank {rank}" if rank else f" -> NOT in top-{top_k}"),
            ))
        elif action == "REFUSE_NO_DATA" and doc and Dieu:
            hits = _retrieve_for(corpus, classifier, retriever, item["question"])
            covering = [h for h in hits if h.doc_num == doc and h.number == Dieu]
            results.append(RetrResult(
                item["id"], "no_cover", not covering, None,
                f"{doc} Dieu {Dieu} must NOT be retrievable (refusal ground)"
                + (f" -> {len(covering)} covering hit(s)" if covering else " -> clean"),
            ))
        else:
            results.append(RetrResult(
                item["id"], "skip", True, None,
                "decided upstream of retrieval (metadata/amendment/scope/injection/non-legal)",
            ))
    return results


def retrieval_summary(results: list[RetrResult]) -> dict:
    scored = [r for r in results if r.kind == "recall"]
    ranks = [r.rank for r in scored if r.rank]
    return {
        "scored": len(scored),
        "no_cover": sum(1 for r in results if r.kind == "no_cover"),
        "skipped": sum(1 for r in results if r.kind == "skip"),
        "recall_at_k": sum(1 for r in scored if r.ok) / max(1, len(scored)),
        "mrr": (sum(1.0 / r for r in ranks) / len(ranks)) if ranks else 0.0,
        "top1": sum(1 for r in ranks if r == 1) / max(1, len(ranks)),
        "passed": sum(1 for r in results if r.ok),
        "total": len(results),
    }


# ---------------------------------------------------------------- tier 3
@dataclass
class RegResult:
    item_id: str
    bug: str
    ok: bool
    expected: str
    actual: str
    problems: list[str] = field(default_factory=list)


def run_regression_tier(pipeline: RAGPipeline, items: list[dict]) -> list[RegResult]:
    """Tier 3: one item = one pinned production bug. Minimal checks —
    action + must_contain / must_not_contain (matched in folded form)."""
    results: list[RegResult] = []
    for item in items:
        res = pipeline.handle(item["question"])
        allowed = ACTION_ALIASES.get(item["expected_action"],
                                     {item["expected_action"].lower()})
        fa = fold(res.answer)
        problems: list[str] = []
        if res.action not in allowed:
            problems.append(f"action {res.action!r} not in {sorted(allowed)}")
        missing = [f for f in item.get("must_contain", []) if fold(f) not in fa]
        if missing:
            problems.append(f"missing: {missing}")
        extra = [f for f in item.get("must_not_contain", []) if fold(f) in fa]
        if extra:
            problems.append(f"must-not-contain present: {extra}")
        results.append(RegResult(item["id"], item.get("bug", ""), not problems,
                                 item["expected_action"], res.action, problems))
    return results
