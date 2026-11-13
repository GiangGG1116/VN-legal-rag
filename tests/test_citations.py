# -*- coding: utf-8 -*-
from rag_law.citations import format_citations, validate_answer
from rag_law.utils import extract_article_refs, extract_doc_refs, fold, norm_doc_num


def test_norm_doc_num_leading_zeros():
    assert norm_doc_num("05/2026/QH15") == "5/2026/QH15"
    assert norm_doc_num("31/2024/QH15") == "31/2024/QH15"
    assert norm_doc_num("8/2026/QH16") == "8/2026/QH16"


def test_extract_doc_refs():
    refs = extract_doc_refs("Theo Luật 31/2024/QH15 và Nghị định 100/2019/NĐ-CP thì sao?")
    assert "31/2024/QH15" in refs
    assert any(r.startswith("100/2019/NĐ") for r in refs)


def test_extract_doc_refs_dedupe():
    refs = extract_doc_refs("31/2024/QH15 rồi lại 31/2024/QH15")
    assert refs == ["31/2024/QH15"]


def test_extract_article_refs():
    assert extract_article_refs("Điều 12 và điều 45 khoản 2") == [12, 45]


def test_fold():
    assert fold("Đất Đai thuộc Sở Hữu") == "dat dai thuoc so huu"


def test_validate_ok():
    r = validate_answer("Theo Điều 12 Luật Đất đai 31/2024/QH15.", {"31/2024/QH15"}, "Điều 12. Sở hữu đất đai")
    assert r.ok, r.violations


def test_validate_fabricated():
    r = validate_answer("Theo Điều 12 Luật 77/2020/QH14.", {"31/2024/QH15"}, "Điều 12. x")
    assert not r.ok
    assert "fabricated_ref:77/2020/QH14" in r.violations
    # article 12 IS in the context → no unsupported_article
    assert "unsupported_article:12" not in r.violations


def test_format_citations():
    s = format_citations("31/2024/QH15", "Luật Đất đai", 12, "https://vbpl.vn/x")
    assert "31/2024/QH15" in s and "Điều 12" in s and "vbpl.vn" in s
