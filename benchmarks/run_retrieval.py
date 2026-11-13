# -*- coding: utf-8 -*-
"""TIER 1 — Retrieval benchmark: question -> chunk (NO LLM, ~0 cost).

Chạy mỗi commit (fast). Kiểm tra:
  - item ANSWER/CAUTION có expected_doc+expected_dieu: cặp (doc, Điều) phải
    nằm trong top-5 của retriever (đúng đường gọi production)
  - item REFUSE_NO_DATA có expected_doc+expected_dieu: KHÔNG được có hit nào
    phủ đúng điều đó (cơ sở từ chối ở tầng retrieval)
  - item còn lại: quyết định ở tầng trên retrieval (metadata/amendment/scope)
    -> skip

Exit 0 chỉ khi 100% item được chấm pass (dùng làm CI gate qua pytest:
tests/test_benchmarks.py).

Usage:
    python -m benchmarks.run_retrieval
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rag_law
sys.path.insert(0, str(Path(__file__).resolve().parent))         # eval_lib

from rag_law.corpus import Corpus
from rag_law.retrieval import LexicalRetriever

from eval_lib import DATA, load_items, retrieval_summary, run_retrieval_tier


def main() -> int:
    t0 = time.time()
    corpus = Corpus()
    retriever = LexicalRetriever(corpus)
    items = load_items(DATA / "golden_seed.jsonl")
    results = run_retrieval_tier(corpus, retriever, items)
    s = retrieval_summary(results)

    print(f"Tier 1 (Retrieval) — {len(items)} items | index built in {time.time()-t0:.1f}s")
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.item_id:<10} {r.kind:<9} {r.detail}")

    print("\n" + "=" * 60)
    print(f"Recall@5: {s['recall_at_k']*100:.1f}%  ({s['passed']}/{s['total']} items ok)")
    print(f"MRR:      {s['mrr']:.3f}")
    print(f"Top-1:    {s['top1']*100:.1f}%   (scored={s['scored']}, no_cover={s['no_cover']}, skipped={s['skipped']})")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
