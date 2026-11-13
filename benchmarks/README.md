# Benchmarks — 3 tầng

| Tầng | Nội dung | Chạy khi | Lệnh |
|---|---|---|---|
| **1. Retrieval** | question → chunk (không LLM): top-5 phải chứa đúng `(expected_doc, expected_dieu)`; điều của item `REFUSE_NO_DATA` phải **không** có hit phủ | mỗi commit | `python -m benchmarks.run_retrieval` |
| **2. End-to-end** | question → answer, pipeline production đầy đủ, chấm action + facts | trước mỗi thay đổi lớn | `python -m benchmarks.run_eval` |
| **3. Regression** | mọi bug production đã gặp — mỗi bug = 1 item ghim | mỗi commit | `python -m benchmarks.run_regression` |

Mọi lệnh đều `set PYTHONUTF8=1&& .venv\Scripts\python.exe ...` từ thư mục gốc dự án.
Tất cả runner **exit 0 chỉ khi 100% item pass**.

## Cấu trúc thư mục

```
benchmarks/
├── data/          # DATASET INPUT — chỉ đọc khi eval
│   ├── golden_seed.jsonl            28 items grounded (tầng 1 + 2)
│   ├── regressions.jsonl            bug ghim (tầng 3) — CHỈ ĐƯỢC THÊM, không sửa/xóa
│   └── generated_candidates.jsonl   563 candidates, review: true — vào tầng 2 sau khi review
├── runs/          # ARTIFACT khi chạy (audit JSONL, báo cáo --json) — tự tạo, không commit
├── eval_lib.py    # lib dùng chung: ACTION_ALIASES, run_retrieval_tier (Recall@k/MRR/Top-1),
│                  # run_regression_tier, constants DATA/RUNS
├── run_retrieval.py   # runner tầng 1
├── run_eval.py        # runner tầng 2
├── run_regression.py  # runner tầng 3
└── make_generated.py  # sinh data/generated_candidates.jsonl (deterministic, --seed 7)
```

## CI

`pytest tests/` (chứa `tests/test_benchmarks.py`) tự chạy **tầng 1 + 3 mỗi commit**.
Tầng 2 chạy manual trước thay đổi lớn.

## Quy tắc

- **Mỗi bug production mới** → thêm 1 item vào `data/regressions.jsonl` (id `reg-NNN` tăng dần,
  ghi rõ `bug`) **trước khi** đóng bug.
- Item `review: true` trong `generated_candidates.jsonl` không được đếm vào kết quả
  cho đến khi review thủ công và chuyển qua `golden_seed.jsonl`.
- Audit (`runs/audit*.jsonl`) là append-only — không sửa tay, cần chạy lại thì xóa file rồi chạy.
