# -*- coding: utf-8 -*-
"""Auto-generate benchmark CANDIDATES from the corpus (scale for retrieval eval).

For a deterministic sample of articles across all docs, emit template
questions. These are machine-generated: they scale to thousands of items for
recall@k measurement, but each answer must still be grounded in the article
text. Items marked "review": true need a human pass before counting as golden.

Usage:
    python -m benchmarks.make_generated [--per-doc 2] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_law.corpus import Corpus
from rag_law.parsing import parse_articles

HERE = Path(__file__).resolve().parent

TEMPLATES = [
    "Điều {n} {law} quy định về điều gì?",
    "Theo Điều {n} của {law}, nội dung chính là gì?",
    "Trích nội dung Điều {n} {law}.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = Corpus()
    out_path = HERE / "data" / "generated_candidates.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for dn, content in corpus.contents.items():
            if corpus.docs[dn].data_error:
                continue
            arts = parse_articles(dn, content)
            arts = [a for a in arts if 200 <= len(a.text) <= 4000]  # skip stubs & monster articles
            if not arts:
                continue
            for a in rng.sample(arts, min(args.per_doc, len(arts))):
                tpl = TEMPLATES[n % len(TEMPLATES)]
                law = corpus.docs[dn].title
                item = {
                    "id": f"gen-{dn.replace('/', '_')}-d{a.number}",
                    "question": tpl.format(n=a.number, law=law),
                    "category": "article_lookup",
                    "expected_action": "ANSWER",
                    "expected_doc": dn,
                    "expected_dieu": a.number,
                    # grounded fact: first 60 chars of the article body (folded at eval time)
                    "answer_facts": [a.text.strip()[:60]],
                    "review": True,
                }
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} candidates -> {out_path}")


if __name__ == "__main__":
    main()
