# -*- coding: utf-8 -*-
"""Shared helpers: diacritics-insensitive matching, doc-number handling."""
from __future__ import annotations

import re
import unicodedata


def fold(s: str) -> str:
    """Lowercase + strip diacritics, normalize whitespace.

    Note: đ/Đ (U+0110/0111) do NOT decompose under NFD, so map them explicitly.
    """
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


# Document number: "31/2024/QH15", "8/2026/QH16", "100/2019/NĐ-CP", ...
# Covers national assembly (QHxx), constitutional assembly (CV), UBTVQH (PL),
# president (CTN), government (NĐ-CP, CP), ministerial (TT, QĐ).
DOC_NUM_RE = re.compile(
    r"\b(\d{1,3})\s*/\s*(\d{4})\s*/\s*(QH\d{1,2}|CV|UBTVQH|PL|CTN|NĐ-?CP|TT|QĐ|CP)\b",
    re.IGNORECASE,
)

# Sub-law types (Level 3-6) — out of scope for V1.
SUB_LAW_TYPES = {"NĐ-CP", "NĐCP", "TT", "QĐ", "CP", "PL", "CTN", "UBTVQH"}


def norm_doc_num(s: str) -> str:
    """Normalize a doc number: strip leading zeros, uppercase the type part."""
    s = s.strip().upper()
    s = re.sub(r"^0+(?=\d)", "", s)
    return s


def extract_doc_refs(text: str) -> list[str]:
    """All doc numbers in text, normalized, deduped, order-preserving."""
    out: list[str] = []
    seen: set[str] = set()
    for m in DOC_NUM_RE.finditer(text):
        num = f"{int(m.group(1))}/{m.group(2)}/{m.group(3).upper()}"
        num = norm_doc_num(num)
        if num not in seen:
            seen.add(num)
            out.append(num)
    return out


def doc_type_of(doc_num: str) -> str:
    m = re.match(r"\d+/\d+/(\w+(?:-\w+)?)$", doc_num)
    return m.group(1) if m else ""


def is_in_scope_ref(doc_num: str) -> bool:
    # Any "QHnn" generation counts as in-scope form (unknown generations like
    # QH99 are unknown docs, handled by the catalogue check — not sub-law).
    t = doc_type_of(doc_num)
    return t.startswith("QH") or t in ("CV", "HP")


ARTICLE_REF_RE = re.compile(r"\bđiều\s+(\d{1,4})\b", re.IGNORECASE)


def extract_article_refs(text: str) -> list[int]:
    """Article numbers mentioned in text, deduped, order-preserving."""
    out: list[int] = []
    seen: set[int] = set()
    for m in ARTICLE_REF_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
