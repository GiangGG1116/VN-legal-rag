# -*- coding: utf-8 -*-
"""Abstention engine — the "know when to say I don't know" core.

Combines three independent signals into one fail-closed decision:
1. classification  (scope + intent, from QuestionClassifier)
2. corpus          (is the referenced document actually in the index? data errors?)
3. retrieval       (did we find content that actually answers the question?)

Actions
-------
answer               : proceed to generation with the retrieved context
answer_with_caution  : proceed, but the answer must carry the stated caveat(s)
refuse_scope         : out of the system's scope (sub-law, non-legal)
refuse_no_data       : in-scope question but no reliable data (missing doc,
                       data error, unknown number, low retrieval confidence)
refuse_injection     : hostile query
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag_law.classify import Classification
from rag_law.corpus import Corpus
from rag_law.utils import extract_doc_refs

MIN_SCORE = 0.12          # min normalized lexical score to trust a hit (tuned in eval)
MIN_SCORE_ARTICLE = 0.20  # stricter when a specific article was asked for


@dataclass
class RetrievalHit:
    doc_num: str
    number: int | None      # article number, if chunk is article-level
    text: str
    score: float
    title: str = ""
    chapter: str = ""
    kind: str = "article"   # article | header | appendix | metadata


@dataclass
class Decision:
    action: str
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    refusal_message: str | None = None
    refs_for_citation: list[str] = field(default_factory=list)  # doc_nums the answer may cite

MSG_INJECTION = (
    "Tôi không thể xử lý yêu cầu này. Hệ thống chỉ trả lời các câu hỏi về "
    "Hiến pháp, Bộ luật và Luật do Quốc hội Việt Nam ban hành."
)
MSG_SCOPE = (
    "Câu hỏi này vượt ngoài phạm vi của hệ thống hiện tại. Tôi chỉ tra cứu "
    "Hiến pháp, Bộ luật và Luật (cấp Quốc hội), không bao gồm Nghị định, Thông tư, "
    "Quyết định và các văn bản hướng dẫn. Vui lòng tra cứu trực tiếp tại vbpl.vn."
)
MSG_NON_LEGAL = (
    "Tôi chỉ hỗ trợ trả lời các câu hỏi về pháp luật Việt Nam "
    "(Hiến pháp, Bộ luật, Luật). Câu hỏi của bạn không thuộc lĩnh vực này."
)


def _refuse_no_data(corpus: Corpus, refs: list[str], sub_type: str, why: str) -> Decision:
    if refs:
        r = refs[0]
        title = corpus.title_of(r)
        msg = (
            f"Tôi không thể trả lời câu hỏi này: văn bản «{title}» (Số hiệu {r}) "
            f"không có nội dung đầy đủ trong bộ dữ liệu hiện tại (cập nhật đến "
            f"{corpus.snapshot_date}). {why} "
            f"Vui lòng kiểm tra toàn văn tại vbpl.vn."
        )
    else:
        msg = (
            "Xin lỗi, tôi không tìm thấy nội dung phù hợp trong bộ dữ liệu pháp luật "
            f"hiện có (cập nhật đến {corpus.snapshot_date}). "
            "Câu hỏi có thể liên quan đến văn bản chưa có trong bộ dữ liệu, hoặc tôi "
            "không đủ thông tin để trả lời chính xác. Vui lòng tra cứu tại vbpl.vn "
            "hoặc diễn đạt lại câu hỏi cụ thể hơn (tên luật / số hiệu / số điều)."
        )
    return Decision("refuse_no_data", ["no reliable data: " + why], refusal_message=msg)


def decide(cls: Classification, hits: list[RetrievalHit], corpus: Corpus,
           min_score: float = MIN_SCORE) -> Decision:
    """Fail-closed decision: when in doubt, refuse and say why."""

    # 1) injection
    if cls.category == "injection":
        return Decision("refuse_injection", cls.reasons or ["injection"], refusal_message=MSG_INJECTION)

    # 2) scope refusals
    if cls.category == "non_legal":
        return Decision("refuse_non_legal", cls.reasons, refusal_message=MSG_NON_LEGAL)
    if cls.category == "sub_law":
        return Decision("refuse_scope", cls.reasons, refusal_message=MSG_SCOPE)

    # 3) corpus-level checks on explicitly referenced documents
    refs = cls.in_scope_refs or cls.law_refs
    data_error = None
    missing_doc = None
    for r in refs:
        meta = corpus.lookup(r)
        if meta is None:
            missing_doc = (r, f"Số hiệu {r} không tồn tại trong cơ sở dữ liệu pháp luật.")
        elif meta.data_error and meta.in_corpus:
            data_error = (r, f"toàn văn văn bản {r} trong bộ dữ liệu đang bị lỗi (đã phát hiện ở khâu kiểm tra dữ liệu).")
        elif not meta.in_corpus and r not in {h.doc_num for h in hits}:
            # present in catalogue (status/dates) but no full text
            if cls.sub_type in ("status", "effective_date") and (meta.eff_from or meta.status_name):
                continue  # metadata-only answer is possible
            missing_doc = (r, f"văn bản {r} không có trong bộ dữ liệu hiện hành (đang hiệu lực khác hoặc thiếu toàn văn).")
    if missing_doc:
        return _refuse_no_data(corpus, [missing_doc[0]], cls.sub_type, missing_doc[1])
    if data_error:
        return _refuse_no_data(corpus, [data_error[0]], cls.sub_type, data_error[1])

    # 4) legal advice → allow but force caution
    caveats: list[str] = []
    if cls.category == "legal_advice":
        caveats.append(
            "Đây là thông tin pháp lý tổng quát, KHÔNG phải tư vấn pháp lý cho trường hợp cá nhân. "
            "Người hỏi nên tham vấn luật sư có thẩm quyền."
        )

    # 5) metadata answers do NOT depend on retrieval at all — the catalogue
    # (sqlite) is the authoritative source for status / dates / amendment edges.
    if cls.sub_type in ("status", "effective_date") and refs and all(
        corpus.lookup(r) for r in refs
    ):
        allowed = set(refs)
        for r in refs:
            for rep in corpus.replaced_by(r):
                allowed.add(rep)
                allowed |= set(extract_doc_refs(corpus.title_of(rep)))
        return Decision(
            "answer", cls.reasons + ["metadata-only answer"],
            caveats=caveats, refs_for_citation=sorted(allowed),
        )
    if cls.sub_type == "amendment" and refs and any(
        corpus.amended_by.get(r) or corpus.amends.get(r) for r in refs
    ):
        allowed = set(refs)
        for r in refs:
            neighbors = set(corpus.amended_by.get(r, [])) | set(corpus.amends.get(r, []))
            allowed |= neighbors
            for a in neighbors:
                # amender titles often list the amended laws by number — the
                # graph answer quotes those titles verbatim
                allowed |= set(extract_doc_refs(corpus.title_of(a)))
        return Decision(
            "answer", cls.reasons + ["amendment-graph answer"],
            caveats=caveats, refs_for_citation=sorted(allowed),
        )

    # 6) retrieval signal
    if not hits:
        return _refuse_no_data(corpus, refs, cls.sub_type, "không có kết quả truy vấn phù hợp.")

    top = hits[0]
    threshold = MIN_SCORE_ARTICLE if (cls.sub_type == "article_lookup" and cls.asked_articles) else min_score
    if top.score < threshold:
        return _refuse_no_data(corpus, refs, cls.sub_type,
                               f"độ tin cậy truy vấn thấp (score {top.score:.3f} < {threshold:.2f}).")

    # asked for a specific article? at least one hit must cover it
    if cls.sub_type == "article_lookup" and cls.asked_articles:
        covered = {h.number for h in hits if h.doc_num in (refs or {top.doc_num})}
        if not (set(cls.asked_articles) & covered):
            return _refuse_no_data(
                corpus, refs, cls.sub_type,
                f"không tìm thấy Điều {cls.asked_articles[0]} của văn bản được hỏi trong bộ dữ liệu.",
            )

    # 7) amendment awareness: if the hit's law has been amended and the question
    # targets its CONTENT (an article, or "hiện hành"/current wording), force a
    # consolidation caveat — the corpus is NOT consolidated.
    caveated_refs: set[str] = set()
    if (cls.sub_type in ("article_lookup", "amendment", "overview") and cls.asked_articles) or _asks_current_state(cls):
        amenders = corpus.amended_by.get(top.doc_num, [])
        if amenders:
            names = ", ".join(f"{a} ({corpus.title_of(a)})" for a in amenders[:3])
            caveats.append(
                f"Văn bản {top.doc_num} đã được sửa đổi, bổ sung bởi: {names}. "
                "Nội dung được trích dẫn có thể là bản gốc chưa hợp nhất — cần đối chiếu văn bản sửa đổi."
            )
            # the caveat text names the amenders (and their titles may list
            # further doc numbers) — all admissible citations
            caveated_refs = set(amenders)
            for a in amenders:
                caveated_refs |= set(extract_doc_refs(corpus.title_of(a)))
    # mixed refs warning
    if cls.out_of_scope_refs and cls.in_scope_refs:
        caveats.append(
            "Câu hỏi có tham chiếu cả văn bản cấp dưới luật (ngoài phạm vi hệ thống); "
            "chỉ phần thuộc Luật/Bộ luật/Hiến pháp được trả lời."
        )

    action = "answer_with_caution" if caveats else "answer"
    allowed_refs = sorted(
        {h.doc_num for h in hits} | set(cls.in_scope_refs)
        | set(cls.law_refs) | caveated_refs
    )
    return Decision(action, cls.reasons, caveats=caveats, refs_for_citation=allowed_refs)


def _asks_current_state(cls: Classification) -> bool:
    """Query explicitly asks about the CURRENT wording (consolidation-sensitive)."""
    markers = ["hien hanh", "hien nay", "ban moi nhat", "moi nhat", "dang ap dung"]
    return any(m in cls.folded_query for m in markers)
