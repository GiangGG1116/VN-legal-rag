# -*- coding: utf-8 -*-
"""Citation extraction & validation — anti-hallucination layer.

The answer may only cite:
- Số hiệu (doc numbers) that belong to the allowed set (retrieved docs + the
  explicitly referenced docs from the catalogue), and
- "Điều N" numbers that actually appear in the retrieved context.

Anything else is flagged as a violation (fabricated_ref / unsupported_article)
so the pipeline can refuse to publish the answer instead of shipping a
hallucinated citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag_law.utils import extract_article_refs, extract_doc_refs, fold


@dataclass
class CitationReport:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    cited_docs: list[str] = field(default_factory=list)
    cited_articles: list[int] = field(default_factory=list)

    def add(self, violation: str) -> None:
        self.ok = False
        self.violations.append(violation)


def validate_answer(answer: str, allowed_docs: set[str], context: str) -> CitationReport:
    report = CitationReport()
    cited_docs = extract_doc_refs(answer)
    report.cited_docs = cited_docs
    for d in cited_docs:
        if d not in allowed_docs:
            report.add(f"fabricated_ref:{d}")

    cited_articles = extract_article_refs(answer)
    report.cited_articles = cited_articles
    # an article citation is supported if "Điều N" appears in the retrieved context
    # (the context contains the article headings themselves). Compare in folded
    # form: corpus text may be NFC or NFD-normalized, fold() makes the check
    # independent of that.
    ctx = fold(context)
    for n in cited_articles:
        if (f"dieu {n}." in ctx or f"dieu {n} " in ctx
                or f"dieu {n}," in ctx):
            continue
        report.add(f"unsupported_article:{n}")
    return report


def format_citations(doc_num: str, title: str, article_no: int | None,
                     url: str | None = None) -> str:
    where = f", Điều {article_no}" if article_no else ""
    cite = f"{title} — Số hiệu: {doc_num}{where}"
    if url:
        cite += f" — {url}"
    return cite
