# -*- coding: utf-8 -*-
"""Question classification — gate 1 of the safety pipeline.

Rule-first (deterministic, testable, no LLM required). An LLM-based
classifier can be plugged in later via `llm_classify`; the rule layer is
kept as a hard pre-filter because it must never be bypassed by the model.

Categories
----------
answer_candidate  : in-scope legal question (laws / codes / constitution)
sub_law           : asks about sub-law instruments (NĐ, TT, QĐ, PL, …) — V1 out of scope
legal_advice      : personal-case advice request → answer only with strong caveats
non_legal         : not a legal question at all
injection         : prompt-injection / jailbreak attempt
unknown           : cannot tell → treated as low-confidence, may still be answered
                   but retrieval confidence must be high (fail-closed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_law.corpus import Corpus
from rag_law.utils import DOC_NUM_RE, doc_type_of, fold, is_in_scope_ref

# --- lexical signals ---------------------------------------------------------

def _F(words: list[str]) -> list[str]:
    """Fold a word list — queries are matched in folded form."""
    return [fold(w) for w in words]


SUB_LAW_WORDS = _F([
    "nghị định", "thông tư", "quyết định", "công văn", "chỉ thị",
    "pháp lệnh", "lệnh", "nghị quyết", "hướng dẫn thi hành", "văn bản hướng dẫn",
])
SUB_LAW_NUM_RE = re.compile(r"\b\d{1,3}/\d{4}/(NĐ-?CP|TT-\w+|QĐ-?\w*|CT-?\w*|LCT)\b", re.IGNORECASE)

ADVICE_MARKERS = _F([
    "tôi nên", "tôi phải", "tôi có bị", "tôi có thể", "tôi cần làm gì", "trường hợp của tôi",
    "nhà tôi", "đất của tôi", "xe của tôi", "hợp đồng của tôi", "giúp tôi", "tư vấn giúp",
    "tôi có nên", "liệu tôi",
])
ADVICE_CONTEXT = _F([
    "tranh chấp", "khởi kiện", "ly hôn", "thừa kế", "phạt", "tội", "bị xử",
    "mua", "bán", "thuê", "chuyển nhượng", "bồi thường", "trục xuất", "hộ khẩu",
])

INJECTION_PATTERNS = [
    r"bỏ qua (các )?(hướng dẫn|lệnh|quy tắc|trả lời) (trước|này)",
    r"phớt lờ (các )?(hướng dẫn|lệnh|quy tắc)",
    r"quên (mất )?(hướng dẫn|lệnh|quy tắc) (trước|trên)",
    r"quên đi tất cả",
    r"đóng vai (là|vai)",
    r"ignore (all |any )?(previous|prior|above|earlier)",
    r"disregard (all |any )?(previous|prior)",
    r"you are now",
    r"act as if you (have no|had no)",
    r"reveal (the )?(system|developer) prompt",
    r"system prompt",
    r"new instructions?:",
    r"do whatever i say",
]
INJECTION_RE = re.compile("|".join(f"({p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

DATE_WORDS = _F(["hiệu lực", "ngày ban hành", "ban hành", "có hiệu lực từ", "từ ngày nào", "từ khi nào", "ngày nào có hiệu lực"])
STATUS_WORDS = _F(["còn hiệu lực", "hết hiệu lực", "vẫn còn hiệu lực", "đang hiệu lực", "còn hiệu lực không", "hiệu lực không"])

# The question must actually ASK about the amendment relation
# ("đã được sửa đổi bởi luật nào?", "sửa đổi, bổ sung ... của những luật nào?").
# Mere words like "hiện hành" mean "current wording" → content question, NOT a
# graph question (handled via the consolidation caveat instead).
AMEND_RELATION_RE = re.compile(
    r"duoc (sua doi|bo sung|thay the)"
    r"|bien (luat|bo luat|van ban) (nao|gi)"
    r"|sua doi (,| )?bo sung (,| )?mot so dieu"
    r"|sua doi (,| )?bo sung cua"
    r"|(nao|gi) sua doi"
)

LEGAL_WORDS = _F([
    "luật", "hiến pháp", "bộ luật", "điều khoản", "quy định", "pháp luật",
    "văn bản", "quy phạm", "quốc hội", "nước cộng hòa xã hội chủ nghĩa việt nam",
    "trình tự", "thủ tục", "xử phạt", "tội", "hình phạt", "quyền", "nghĩa vụ",
])


@dataclass
class Classification:
    category: str = "unknown"
    sub_type: str = ""          # article_lookup | effective_date | status | amendment | overview
    folded_query: str = ""
    doc_refs: list[str] = field(default_factory=list)      # explicit Số hiệu in the query
    law_refs: list[str] = field(default_factory=list)      # doc_nums matched via known law names
    asked_articles: list[int] = field(default_factory=list)
    in_scope_refs: list[str] = field(default_factory=list)
    out_of_scope_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reasons: list[str] = field(default_factory=list)


_TITLE_SUFFIX_RE = re.compile(r"\s*so\s+\d{1,3}/\d{4}/[a-z0-9\-]+")
_YEAR_SUFFIX_RE = re.compile(r"\s*nam\s+\d{4}\s*$")


def _base_name(name: str) -> str:
    """Folded name without trailing 'số NNN/NNNN/TYPE' or 'năm YYYY', so
    'Hiến pháp 2013' matches the indexed title 'Hiến pháp năm 2013'."""
    b = _TITLE_SUFFIX_RE.sub("", name).strip()
    return _YEAR_SUFFIX_RE.sub("", b).strip()


class QuestionClassifier:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        # folded law-name -> doc_num (longest names first for matching).
        # Index both the full title and its base form (year/number stripped).
        entries: set[tuple[str, str]] = set()
        for dn, m in corpus.docs.items():
            if not m.title:
                continue
            full = fold(m.title).strip()
            if len(full) >= 8:
                entries.add((full, dn))
            base = _base_name(full)
            if len(base) >= 8:
                entries.add((base, dn))
        self._name_index: list[tuple[str, str]] = sorted(entries, key=lambda x: -len(x[0]))

    # ------------------------------------------------------------------ public
    def classify(self, query: str) -> Classification:
        q = query.strip()
        c = Classification()
        folded = fold(q)
        c.folded_query = folded

        # 0) injection — highest priority, checked on raw text
        if INJECTION_RE.search(q):
            c.category = "injection"
            c.confidence = 0.95
            c.reasons.append("prompt-injection pattern detected")
            return c

        # 1) document references
        c.doc_refs = self._extract_refs(q)
        c.in_scope_refs = [r for r in c.doc_refs if is_in_scope_ref(r)]
        c.out_of_scope_refs = [r for r in c.doc_refs if not is_in_scope_ref(r)]
        c.law_refs = self._match_law_names(folded)
        from rag_law.utils import extract_article_refs
        c.asked_articles = extract_article_refs(q)

        # 2) sub-law (out of scope for V1)
        sub_hit = next((w for w in SUB_LAW_WORDS if w in folded), None)
        if SUB_LAW_NUM_RE.search(q) or (sub_hit and self._has_legal_signal(folded, c)):
            c.category = "sub_law"
            c.sub_type = "sub_law_doc"
            c.confidence = 0.9
            c.reasons.append(f"sub-law instrument mentioned ({sub_hit or 'number pattern'})")
            return c

        # 3) legal advice (personal case)
        advice = any(a in folded for a in ADVICE_MARKERS)
        ctx = any(x in folded for x in ADVICE_CONTEXT)
        legal = self._has_legal_signal(folded, c)
        if advice and (ctx or legal):
            c.category = "legal_advice"
            c.sub_type = "personal_case"
            c.confidence = 0.8
            c.reasons.append("personal legal-advice request")
            return c

        # 4) non-legal
        if not legal and not c.doc_refs and not c.law_refs:
            c.category = "non_legal"
            c.confidence = 0.7
            c.reasons.append("no legal signal")
            return c

        # 5) in-scope legal question → sub-type
        c.category = "answer_candidate"
        if any(w in folded for w in STATUS_WORDS):
            c.sub_type = "status"
        elif any(w in folded for w in DATE_WORDS):
            c.sub_type = "effective_date"
        elif AMEND_RELATION_RE.search(folded) and (c.doc_refs or c.law_refs):
            c.sub_type = "amendment"
        elif c.asked_articles:
            c.sub_type = "article_lookup"
        elif AMEND_RELATION_RE.search(folded):
            c.sub_type = "amendment"
        else:
            c.sub_type = "overview"
        # explicit out-of-scope number co-occurring with an in-scope one: keep going,
        # but flag it so the answer layer can warn.
        if c.out_of_scope_refs and c.in_scope_refs:
            c.reasons.append("query mixes in-scope and sub-law references")
        c.confidence = 0.8 if (c.doc_refs or c.law_refs) else 0.6
        return c

    # ----------------------------------------------------------------- helpers
    def _extract_refs(self, q: str) -> list[str]:
        from rag_law.utils import extract_doc_refs
        return extract_doc_refs(q)

    def _match_law_names(self, folded_q: str) -> list[str]:
        out: list[str] = []
        for name, dn in self._name_index:
            if not name or len(name) < 8:
                continue
            if name in folded_q and dn not in out:
                out.append(dn)
            if len(out) >= 3:
                break
        return out

    def _has_legal_signal(self, folded_q: str, c: Classification) -> bool:
        if c.doc_refs or c.law_refs or c.asked_articles:
            return True
        return any(w in folded_q for w in LEGAL_WORDS)
