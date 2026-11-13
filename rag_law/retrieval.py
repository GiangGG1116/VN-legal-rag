# -*- coding: utf-8 -*-
"""Lexical (BM25-like) retriever over article-level index.

This is a deterministic stand-in so the whole pipeline (classify → decide →
retrieve → compose → validate) is testable end-to-end BEFORE the embedding
index exists. Scoring:
  + token overlap with IDF weighting
  + bonus when the asked article number matches
  + bonus when article title tokens appear in the query
Swap-in point for the hybrid (bge-m3 + BM25 + reranker) from RAG_PLAN §3.3.
"""
from __future__ import annotations

import math
import re

from rag_law.abstain import RetrievalHit
from rag_law.corpus import Corpus
from rag_law.parsing import parse_articles
from rag_law.utils import fold

STOPWORDS = {
    "cua", "va", "ve", "la", "theo", "trong", "nhu", "mot", "co", "tai", "voi",
    "nao", "gi", "day", "ay", "bao", "nhieu", "khi", "tu", "den", "duoc",
    "bi", "cho", "neu", "ma", "cung", "hay", "cac", "nhung", "do", "ai",
    "vay", "se", "chua", "da", "toan", "khong",
}

TOKEN_RE = re.compile(r"[a-z0-9à-ỹđ]+")


def tokenize(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(fold(s)) if t not in STOPWORDS and len(t) > 1]


class LexicalRetriever:
    def __init__(self, corpus: Corpus, top_k: int = 6):
        self.corpus = corpus
        self.top_k = top_k
        self._docs: list[dict] = []
        self._df: dict[str, int] = {}
        self._n = 0
        self._build()

    def _build(self) -> None:
        for dn, content in self.corpus.contents.items():
            if self.corpus.docs[dn].data_error:
                continue  # never index a known-bad body (45/2019/QH14)
            law_title = self.corpus.docs[dn].title
            for art in parse_articles(dn, content):
                toks = tokenize(art.title + " " + art.text)
                self._docs.append({
                    "doc_num": dn,
                    "number": art.number,
                    "title": f"Điều {art.number}. {art.title}".strip(),
                    "chapter": art.chapter,
                    "text": art.text,
                    "law_title": law_title,
                    "tokens": toks,
                    "token_set": set(toks),
                })
                for t in set(toks):
                    self._df[t] = self._df.get(t, 0) + 1
        self._n = max(1, len(self._docs))

    def retrieve(self, query: str, doc_refs: list[str] | None = None,
                 law_refs: list[str] | None = None) -> list[RetrievalHit]:
        # doc numbers are handled by the doc filter, not by token scoring
        from rag_law.utils import DOC_NUM_RE
        q_toks = tokenize(DOC_NUM_RE.sub(" ", query))
        if not q_toks:
            return []
        q_set = set(q_toks)
        limit = set(doc_refs or []) | set(law_refs or [])
        scores: list[tuple[float, dict]] = []
        for d in self._docs:
            if limit and d["doc_num"] not in limit:
                continue
            overlap = q_set & d["token_set"]
            if not overlap:
                continue
            idf = {t: math.log((self._n + 1) / (self._df.get(t, 0) + 1)) + 1 for t in overlap}
            base = sum(idf[t] for t in overlap) / (math.log(1 + len(d["tokens"])) + 1)
            score = base
            # asked-article bonus: an explicit "Điều N" in the query is a hard
            # constraint — the user asked for THAT article, so it must dominate
            # over semantically-adjacent articles (e.g. "Điều 101 BLHS" must not
            # be shadowed by Điều 90/91/98 which share the "người dưới 18 tuổi"
            # topic).
            if d["number"] is not None:
                for m in re.finditer(r"điều\s+(\d{1,4})", query, re.IGNORECASE):
                    if int(m.group(1)) == d["number"]:
                        score += 15.0
            # title-token bonus, weighted by rarity and normalized by title
            # length: a rare query term in a short, focused title (e.g. "sở
            # hữu" in "Điều 12. Sở hữu đất đai") beats common terms piling up
            # in a long administrative title.
            title_toks = set(tokenize(d["title"]))
            title_hits = q_set & title_toks
            if title_hits:
                score += 2.0 * sum(idf[t] for t in title_hits) / math.sqrt(len(title_toks))
            scores.append((score, d))
        scores.sort(key=lambda x: -x[0])
        out: list[RetrievalHit] = []
        for score, d in scores[: self.top_k]:
            out.append(RetrievalHit(
                doc_num=d["doc_num"],
                number=d["number"],
                text=d["text"],
                score=round(score, 4),
                title=d["title"],
                chapter=d["chapter"],
                kind="article",
            ))
        return out
