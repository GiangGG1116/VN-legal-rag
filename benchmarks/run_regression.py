# -*- coding: utf-8 -*-
"""TIER 3 — Regression benchmark: mọi bug production đã gặp, ghim thành item.

Chạy mỗi commit (thấp chi phí). Mỗi item trong regressions.jsonl = 1 bug thật
đã gặp + fix; runner chạy pipeline production đầy đủ nhưng chỉ chấm các điểm
đã ghim: expected_action + must_contain / must_not_contain (dạng folded).

QUY TẮC: mỗi bug phát hiện mới (dev hoặc production) PHẢI được thêm vào
regressions.jsonl trước khi đóng bug — file này chỉ được phép lớn dần.

Exit 0 chỉ khi tất cả item pass (dùng làm CI gate qua pytest:
tests/test_benchmarks.py).

Usage:
    python -m benchmarks.run_regression
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rag_law
sys.path.insert(0, str(Path(__file__).resolve().parent))         # eval_lib

from rag_law.corpus import Corpus
from rag_law.pipeline import RAGPipeline
from rag_law.retrieval import LexicalRetriever

from eval_lib import DATA, RUNS, load_items, run_regression_tier


def main() -> int:
    t0 = time.time()
    corpus = Corpus()
    retriever = LexicalRetriever(corpus)
    RUNS.mkdir(exist_ok=True)
    pipeline = RAGPipeline(corpus, retriever,
                           audit_path=str(RUNS / "audit_regression.jsonl"))
    items = load_items(DATA / "regressions.jsonl")
    results = run_regression_tier(pipeline, items)
    passed = sum(1 for r in results if r.ok)

    print(f"Tier 3 (Regression) — {len(items)} pinned bugs | ready in {time.time()-t0:.1f}s")
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        line = f"[{mark}] {r.item_id:<10} expect={r.expected:<20} got={r.actual}"
        print(line)
        if r.problems:
            print(f"        bug: {r.bug}")
            for p in r.problems:
                print(f"        {p}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{len(results)} pinned bugs stay fixed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
