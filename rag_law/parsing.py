# -*- coding: utf-8 -*-
"""Structure-aware parsing of Vietnamese law documents.

Corpus facts (validated on all 306 docs of the 2026 split):
- Headings are plain text lines, not markdown headings: "Chương I", "Phần thứ nhất", "Điều 12. Tiêu đề".
- One document (27/2023/QH15, Luật Nhà ở) prefixes headings with bullet
  markers (" oĐiều 1.", " ·Chương I") — the regexes below tolerate them.
- Amendment laws typically have a single "Điều 1" spanning the whole law.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Article:
    doc_num: str
    number: int
    title: str          # heading after "Điều N."
    text: str           # full article body (up to next article)
    chapter: str = ""   # "Chương I — NHỮNG QUY ĐỊNH CHUNG" (best effort)


# Leading bullet markers tolerated: spaces, "·", "o", "-"
ARTICLE_RE = re.compile(r"(?m)^[\s·o\-]*Điều\s+(\d{1,4})[\.:\s]+(.*)$")
CHAPTER_RE = re.compile(r"(?m)^[\s·o\-]*Chương\s+([IVXLCivxlcd]+\d*|\d+)\s*$")
# Where an article body ends: next "Phụ lục" section or final adoption/signoff.
TAIL_RE = re.compile(r"(?im)^\s*(phụ lục|luật này được quốc hội|bộ luật này được quốc hội|hiến pháp này được quốc hội)")


def parse_articles(doc_num: str, content: str) -> list[Article]:
    articles: list[Article] = []
    matches = list(ARTICLE_RE.finditer(content))
    chapters: list[tuple[int, str]] = []
    for m in CHAPTER_RE.finditer(content):
        # chapter title = next non-empty line (ALL-CAPS), best effort
        tail = content[m.end(): m.end() + 200].strip()
        title = tail.splitlines()[0].strip() if tail else ""
        chapters.append((m.start(), f"Chương {m.group(1)} {title}".strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[m.start():end]
        tm = TAIL_RE.search(body[len("Điều"):])
        if tm:
            body = body[: tm.start()]
        chapter = ""
        for pos, ch in chapters:
            if pos < m.start():
                chapter = ch
            else:
                break
        articles.append(
            Article(
                doc_num=doc_num,
                number=int(m.group(1)),
                title=m.group(2).strip(),
                text=body.strip(),
                chapter=chapter,
            )
        )
    return articles


def parse_header(content: str) -> dict:
    """Extract the markdown header block (# title, Số hiệu, English, Ngày hiệu lực)."""
    out: dict = {}
    tm = re.search(r"(?m)^#\s+(.+)$", content)
    if tm:
        out["title"] = tm.group(1).strip()
    sm = re.search(r"\*\*Số hiệu:\*\*\s*([^\n]+)", content)
    if sm:
        out["so_hieu"] = sm.group(1).strip()
    em = re.search(r"\*\*English:\*\*\s*([^\n]+)", content)
    if em:
        out["english"] = em.group(1).strip()
    hm = re.search(r"\*\*Ngày hiệu lực:\*\*\s*([^\n]+)", content)
    if hm:
        out["eff_date"] = hm.group(1).strip()
    return out
