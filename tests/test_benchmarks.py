# -*- coding: utf-8 -*-
"""CI gate — chạy Tier 1 (Retrieval) + Tier 3 (Regression) mỗi commit.

Tier 2 (End-to-end) chạy manual trước mỗi thay đổi lớn:
    python -m benchmarks.run_eval
"""
from pathlib import Path

from benchmarks.eval_lib import (
    DATA as BENCH_DATA,
    load_items,
    retrieval_summary,
    run_regression_tier,
    run_retrieval_tier,
)


def test_tier1_retrieval_golden(corpus, retriever):
    """Question -> chunk: top-5 phải chứa đúng (doc, Điều) cho mọi item
    ANSWER/CAUTION; điều của item REFUSE_NO_DATA phải không có hit phủ."""
    items = load_items(BENCH_DATA / "golden_seed.jsonl")
    results = run_retrieval_tier(corpus, retriever, items)
    s = retrieval_summary(results)
    failed = [(r.item_id, r.detail) for r in results if not r.ok]
    assert not failed, f"Tier 1 failures: {failed} | summary={s}"
    assert s["scored"] >= 10, "golden set phải có >= 10 item chấm retrieval"


def test_tier3_regression(pipeline):
    """Mọi bug production đã gặp phải vẫn được giữ fix (fail = regression)."""
    items = load_items(BENCH_DATA / "regressions.jsonl")
    assert items, "regressions.jsonl trống — mọi bug production phải được ghim tại đây"
    results = run_regression_tier(pipeline, items)
    failed = [(r.item_id, r.bug, r.problems) for r in results if not r.ok]
    assert not failed, f"Tier 3 failures (regressions!): {failed}"
