# -*- coding: utf-8 -*-
"""End-to-end RAG pipeline with fail-closed safety gates.

Flow per query:
  InputGuard → QuestionClassifier → Retrieval → decide() → compose answer
  → OutputValidator (citations, disclaimer, caveats) → AuditLog

V1 generation is a deterministic template (verbatim quote of the article +
citation block). Plugging an LLM in later means replacing `_compose_answer`'s
template path with an LLM call that receives the same validated context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rag_law.abstain import Decision, RetrievalHit, decide
from rag_law.classify import Classification, QuestionClassifier
from rag_law.corpus import Corpus
from rag_law.safety import ADVICE_DISCLAIMER, AuditLog, InputGuard, OutputValidator
from rag_law.utils import extract_doc_refs


@dataclass
class PipelineResult:
    query: str
    action: str
    answer: str
    classification: Classification
    decision: Decision
    hits: list[RetrievalHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    valid: bool = True
    latency_ms: float = 0.0


class RAGPipeline:
    def __init__(self, corpus: Corpus, retriever: Any | None = None,
                 min_score: float = 0.12, audit_path: str | None = None):
        self.corpus = corpus
        self.retriever = retriever
        self.min_score = min_score
        self.guard = InputGuard()
        self.validator = OutputValidator()
        self.classifier = QuestionClassifier(corpus)
        self.audit = AuditLog(audit_path)

    # ------------------------------------------------------------------- main
    def handle(self, query: str) -> PipelineResult:
        t0 = time.perf_counter()
        cls = self.classifier.classify(query)
        gate = self.guard.check(query)

        if "injection" in gate.issues or cls.category == "injection":
            decision = Decision("refuse_injection", ["blocked by input guard"],
                                refusal_message="Tôi không thể xử lý yêu cầu này. Hệ thống chỉ trả lời các câu hỏi về Hiến pháp, Bộ luật và Luật do Quốc hội Việt Nam ban hành.")
            answer = decision.refusal_message or ""
            return self._finish(query, cls, decision, answer, [], gate, t0)

        if gate.issues:
            # non-hard issues (too_long / not_vietnamese) are recorded, not fatal
            pass

        hits = self.retriever.retrieve(query, cls.in_scope_refs or cls.law_refs or None,
                                       cls.law_refs or None) if self.retriever else []
        decision = decide(cls, hits, self.corpus, self.min_score)

        if decision.action.startswith("refuse"):
            answer = decision.refusal_message or "Xin lỗi, tôi không thể trả lời câu hỏi này."
        else:
            answer = self._compose_answer(cls, decision, hits)
            if answer is None:
                decision = Decision(
                    "refuse_no_data",
                    ["no usable context after composition"],
                    refusal_message=(
                        "Xin lỗi, tôi không tìm thấy nội dung phù hợp trong bộ dữ liệu pháp luật "
                        f"hiện có (cập nhật đến {self.corpus.snapshot_date}). Vui lòng tra cứu tại vbpl.vn "
                        "hoặc diễn đạt lại câu hỏi (tên luật / số hiệu / số điều)."
                    ),
                )
                answer = decision.refusal_message
        return self._finish(query, cls, decision, answer, hits, gate, t0)

    # ---------------------------------------------------------------- compose
    def _compose_answer(self, cls: Classification, decision: Decision,
                        hits: list[RetrievalHit]) -> str | None:
        corpus = self.corpus
        sub = cls.sub_type

        # --- amendment questions: answer from the amendment graph (metadata) ---
        if sub == "amendment":
            for r in cls.in_scope_refs or cls.law_refs:
                amenders = corpus.amended_by.get(r, [])
                if amenders and ("sua doi" in cls.folded_query or "bo sung" in cls.folded_query):
                    base = corpus.lookup(r)
                    base_txt = f"{base.title} (Số hiệu {r})" if base else r
                    lines = [f"«{base_txt}» được sửa đổi, bổ sung bởi:"]
                    for a in amenders:
                        m = corpus.lookup(a)
                        lines.append(f"- {a}: {m.title if m else a}")
                    return "\n".join(lines) + f"\n\nNguồn: cơ sở dữ liệu pháp luật (catalogue vbpl, {corpus.snapshot_date})."
                if corpus.amends.get(r):
                    m = corpus.lookup(r)
                    lines = [f"{m.title if m else r} (Số hiệu {r}) sửa đổi, bổ sung một số điều của:"]
                    for b in corpus.amends[r]:
                        mb = corpus.lookup(b)
                        lines.append(f"- {b}: {mb.title if mb else b}")
                    return "\n".join(lines) + f"\n\nNguồn: cơ sở dữ liệu pháp luật (catalogue vbpl, {corpus.snapshot_date})."

        # --- metadata questions: status / effective date ---
        if sub in ("status", "effective_date"):
            parts = []
            for r in cls.in_scope_refs or cls.law_refs:
                m = corpus.lookup(r)
                if not m:
                    continue
                if sub == "status":
                    state = m.status_name or ("đang hiệu lực" if m.in_force else "không còn hiệu lực")
                    line = f"{m.title} (Số hiệu {r}): trạng thái «{state}»."
                    rep_docs = corpus.replaced_by(r)
                    if rep_docs:
                        line += " Được thay thế bởi: " + ", ".join(
                            f"{a} ({corpus.title_of(a)})" for a in rep_docs
                        ) + "."
                else:
                    line = f"{m.title} (Số hiệu {r}):"
                    if m.issue_date:
                        line += f" ban hành ngày {m.issue_date}."
                    if m.eff_from:
                        line += f" Có hiệu lực từ {m.eff_from}."
                    if m.eff_to:
                        line += f" Hết hiệu lực từ {m.eff_to}."
                if m.url_vbpl:
                    line += f" Nguồn: {m.url_vbpl}"
                parts.append(line)
            if parts:
                return "\n".join(parts)
            return None

        # --- content questions: verbatim quote of the best article hit ---
        if not hits:
            return None
        top = hits[0]
        meta = corpus.lookup(top.doc_num)
        quote = top.text.strip()
        if len(quote) > 1500:
            quote = quote[:1500].rsplit(" ", 1)[0] + "…"
        lines = [quote, "", f"— {top.title}{' của ' + meta.title if meta else ''} (Số hiệu: {top.doc_num})"]
        if top.chapter:
            lines[1] = f"— {top.title}, {top.chapter}, {meta.title if meta else ''} (Số hiệu: {top.doc_num})"
        if len(hits) > 1:
            others = ", ".join(f"Điều {h.number}" if h.number else "nội dung liên quan" for h in hits[1:4])
            lines.append(f"[Nội dung liên quan khác: {others} của {top.doc_num}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------ audit
    def _finish(self, query: str, cls: Classification, decision: Decision, answer: str,
                hits: list[RetrievalHit], gate: Any, t0: float) -> PipelineResult:
        warnings = list(gate.issues) if gate and gate.issues else []
        if not decision.action.startswith("refuse"):
            # 1) append disclaimer + caveats, 2) validate, 3) fail-closed on violation
            answer = self.validator.finalize(answer, decision)
            context = "\n".join(h.text for h in hits)
            # A verbatim quote may mention other doc numbers that appear in the
            # quoted article body (e.g. the Land Law's final chapter citing the
            # laws it amends). Anti-hallucination holds: only refs that exist in
            # the retrieved context are admissible.
            allowed = set(decision.refs_for_citation) | set(extract_doc_refs(context))
            rep = self.validator.validate(answer, decision, allowed, context)
            if not rep.ok:
                # if the validator finds fabricated refs/articles, do NOT publish.
                decision = Decision(
                    "refuse_no_data",
                    ["output validation failed: " + "; ".join(rep.violations[:3])],
                    refusal_message=(
                        "Xin lỗi, hệ thống phát hiện mâu thuẫn trong trích dẫn của bản trả lời "
                        "(tham chiếu không khớp ngữ cảnh dữ liệu). Để tránh thông tin sai, tôi "
                        "không thể cung cấp câu trả lời này. Vui lòng diễn đạt lại hoặc tra cứu tại vbpl.vn."
                    ),
                )
                answer = decision.refusal_message
                warnings.extend(rep.violations)
        latency = (time.perf_counter() - t0) * 1000
        self.audit.record(
            query=query[:300],
            category=cls.category,
            sub_type=cls.sub_type,
            action=decision.action,
            reasons=decision.reasons,
            caveats=decision.caveats,
            warnings=warnings,
            hits=[{"doc": h.doc_num, "dieu": h.number, "score": h.score} for h in hits[:5]],
            latency_ms=round(latency, 1),
        )
        return PipelineResult(
            query=query,
            action=decision.action,
            answer=answer,
            classification=cls,
            decision=decision,
            hits=hits,
            warnings=warnings,
            valid=True,
            latency_ms=round(latency, 1),
        )
