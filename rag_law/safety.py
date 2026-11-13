# -*- coding: utf-8 -*-
"""Safety mechanisms: input guard, output validator, audit log.

Layering (all fail-closed, all run before anything reaches the user):
  InputGuard    — reject/flag hostile or malformed queries (gate 0)
  OutputValidator — enforce disclaimer, citation validity, refusal markers
  AuditLog      — append-only JSONL trace of every query/decision
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_law.abstain import Decision
from rag_law.classify import INJECTION_RE
from rag_law.citations import validate_answer

MAX_QUERY_LEN = 1500

DISCLAIMER = (
    "Lưu ý: Trả lời dựa trên bộ dữ liệu pháp luật cập nhật đến tháng 06/2026, chỉ bao gồm "
    "Hiến pháp, Bộ luật và Luật do Quốc hội ban hành (chưa có Nghị định, Thông tư). "
    "Đây là thông tin tham khảo, không phải tư vấn pháp lý. Kiểm tra tại vbpl.vn trước khi sử dụng."
)

ADVICE_DISCLAIMER = (
    "Cảnh báo: Đây chỉ là thông tin pháp lý tổng quát, KHÔNG phải tư vấn pháp lý cho trường hợp cá nhân. "
    "Xin liên hệ luật sư có thẩm quyền cho quyết định của bạn."
)

REFUSAL_MARKER = "không thể"


@dataclass
class InputReport:
    ok: bool = True
    issues: list[str] = field(default_factory=list)

    def add(self, issue: str) -> None:
        self.ok = False
        self.issues.append(issue)


class InputGuard:
    """Gate 0 — runs before classification. Hard blocks only; warnings pass through."""

    HARD_BLOCK = {"injection"}

    def check(self, query: str) -> InputReport:
        rep = InputReport()
        q = query or ""
        if not q.strip():
            rep.add("empty_query")
            return rep
        if len(q) > MAX_QUERY_LEN:
            rep.add(f"too_long:{len(q)}")
        if INJECTION_RE.search(q):
            rep.add("injection")
        # language sanity: almost-no-Vietnamese queries are almost certainly out of domain.
        letters = re.findall(r"[a-zA-ZÀ-ỹà-ỹđĐ]+", q)
        total = sum(len(w) for w in letters)
        if total >= 24:
            vi = sum(1 for w in letters for ch in w if ch in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡộùúủũụưừứửữựỳýỷỹỵđĐ")
            if vi / total < 0.03:
                rep.add("not_vietnamese")
        return rep


@dataclass
class OutputReport:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    final_answer: str = ""


class OutputValidator:
    """Gate 3 — the last line of defence before the answer leaves the system."""

    def validate(self, answer: str, decision: Decision, allowed_docs: set[str],
                 context: str) -> OutputReport:
        rep = OutputReport(final_answer=answer)
        action = decision.action

        if action.startswith("refuse"):
            # refusal answers must be the canned, audited message
            if decision.refusal_message and answer.strip() != decision.refusal_message.strip():
                rep.violations.append("refusal_message_modified")
            if REFUSAL_MARKER not in answer and "không tìm thấy" not in answer and "không thuộc" not in answer:
                rep.violations.append("refusal_marker_missing")
            rep.ok = not rep.violations
            return rep

        # ---- answer / answer_with_caution ----
        if DISCLAIMER not in answer:
            rep.violations.append("disclaimer_missing")
        if action == "answer_with_caution" and decision.caveats:
            if not any(c in answer for c in decision.caveats):
                rep.violations.append("caution_missing")

        cite_report = validate_answer(answer, allowed_docs, context)
        rep.violations.extend(cite_report.violations)
        rep.ok = not rep.violations
        return rep

    def finalize(self, answer: str, decision: Decision) -> str:
        """Ensure disclaimer + caveats are present (called before validation)."""
        out = answer.strip()
        for c in decision.caveats:
            if c not in out:
                out += f"\n\n⚠ {c}"
        if decision.action.startswith("answer") and DISCLAIMER not in out:
            out += f"\n\n{DISCLAIMER}"
        return out


class AuditLog:
    """Append-only JSONL audit trail (one record per pipeline run)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def record(self, **kw: Any) -> None:
        if not self.path:
            return
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        rec.update(kw)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
