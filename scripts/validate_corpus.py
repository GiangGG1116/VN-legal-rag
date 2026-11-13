# -*- coding: utf-8 -*-
"""Phase 0 corpus validator — checks body <-> Số hiệu/title for every doc.

Checks per document (2026 split):
  H1  header: starts with '# <title>', '**Số hiệu:**' line contains the row id
      (extracted by regex — tolerates 'số ', 'Luật số ' prefixes)
  B1  body length >= 1000 chars after the '---' separator
      (small amendment laws are legitimately ~1.3k chars)
  B2  at least one '^Điều N' article line, case-insensitive, punctuation
      optional (amendment laws write 'Điều 1' without a dot)
  B3  body's first non-empty line is a national-level marker
      (QUỐC HỘI / CỘNG HÒA / LUẬT / BỘ LUẬT / HIẾN PHÁP / NƯỚC CỘNG HÒA) —
      catches wrong-document bodies like a provincial plan (292/KH-UBND)
      or web-scraping residue ('Document Content'). Starting with 'CHƯƠNG'
      (missing preamble, articles complete) is a WARN, not a FAIL.
  B4  no U+FFFD replacement characters

Usage:
    python -m scripts.validate_corpus                # canonical 2026 parquet
    python -m scripts.validate_corpus --file <p>     # any parquet (e.g. archive)
    --json <path>  report location (default benchmarks/runs/validate_report.json)

Exit 0 only if 0 FAIL. Intended to run on every new snapshot before ingest.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_law.utils import extract_doc_refs, fold, norm_doc_num  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = PROJECT_ROOT / "data" / "uts_vlc" / "data" / "2026-00000-of-00001.parquet"

NATIONAL_PREFIXES = (
    "quoc hoi", "cong hoa", "nuoc cong hoa",
    "luat", "bo luat", "hien phap",
)
WARN_PREFIXES = ("chuong",)   # preamble missing but articles present
ARTICLE_RE = re.compile(r"(?im)^\s*điều\s+\d+")


def _split_header_body(content: str) -> tuple[str, str]:
    m = re.match(r"(\A# .*?---\s*\n)(.*)", content, re.DOTALL)
    if not m:
        return "", content
    return m.group(1), m.group(2)


def _first_body_line(body: str) -> str:
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def validate_doc(doc_num: str, title: str, doc_type: str, content: str) -> tuple[list[str], list[str]]:
    """Return (fails, warns). FAIL = body không tin cậy; WARN = lỗi nhẹ không
    cản RAG (vd thiếu preamble)."""
    fails: list[str] = []
    warns: list[str] = []
    header, body = _split_header_body(content)

    # H1 — header + Số hiệu (extract Số hiệu trong dòng — chịu prefix 'số ', 'Luật số ')
    if not content.lstrip().startswith("#"):
        fails.append("H1: thiếu header '# <title>'")
    elif header:
        m = re.search(r"\*\*Số hiệu:\*\*\s*([^\n]+)", header)
        if doc_type == "constitution":
            if not fold(title).startswith("hien phap"):
                fails.append(f"H1: title Hiến pháp không khớp ({title!r})")
        elif not m:
            fails.append("H1: thiếu dòng '**Số hiệu:**'")
        else:
            refs = extract_doc_refs(m.group(1))
            if norm_doc_num(doc_num) not in [norm_doc_num(r) for r in refs]:
                fails.append(f"H1: Số hiệu header {m.group(1).strip()!r} != id {doc_num!r}")
            if m.group(1).strip() not in ("", norm_doc_num(doc_num)) and " " in m.group(1).strip():
                warns.append(f"H1: dòng Số hiệu không đúng chuẩn '{m.group(1).strip()}' (đủ Số hiệu, thừa prefix)")
    else:
        fails.append("H1: không tách được header '---'")

    # B1 — body length
    if len(body) < 1000:
        fails.append(f"B1: body quá ngắn ({len(body)} ký tự)")

    # B2 — article structure
    n_art = len(ARTICLE_RE.findall(body))
    if n_art == 0:
        fails.append("B2: không có dòng 'Điều N' nào trong body")

    # B3 — national-level first line
    first = fold(_first_body_line(body))
    if first:
        if not any(first.startswith(p) for p in NATIONAL_PREFIXES):
            if any(first.startswith(p) for p in WARN_PREFIXES):
                warns.append("B3: body thiếu preamble (bắt đầu từ 'Chương') — nội dung điều vẫn đầy đủ")
            else:
                fails.append(f"B3: body bắt đầu bằng văn bản không phải cấp Quốc hội: {_first_body_line(body)[:60]!r}")

    # B4 — encoding
    if "\ufffd" in content:
        fails.append(f"B4: {content.count(chr(0xFFFD))} ký tự U+FFFD")

    return fails, warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(CANONICAL))
    ap.add_argument("--json", default=str(PROJECT_ROOT / "benchmarks" / "runs" / "validate_report.json"))
    args = ap.parse_args()

    rows = pq.read_table(args.file).to_pylist()
    report = []
    n_fail = 0
    n_warn = 0
    for d in rows:
        fails, warns = validate_doc(d["id"], d["title"], d["type"], d["content"])
        n_fail += bool(fails)
        n_warn += bool(warns)
        report.append({"id": d["id"], "title": d["title"], "type": d["type"],
                       "chars": len(d["content"]),
                       "status": "FAIL" if fails else ("WARN" if warns else "PASS"),
                       "fails": fails, "warns": warns})

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"file": args.file, "total": len(rows),
                               "failed": n_fail, "warned": n_warn, "docs": report},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"validate {args.file}")
    print(f"docs: {len(rows)} | FAIL: {n_fail} | WARN: {n_warn}")
    for r in report:
        if r["fails"] or r["warns"]:
            tag = "FAIL" if r["fails"] else "WARN"
            print(f"  [{tag}] {r['id']:<18} {r['title'][:50]}")
            for f in r["fails"] + r["warns"]:
                print(f"         - {f}")
    print(f"report -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
