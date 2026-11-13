# Kế hoạch xây dựng hệ RAG Pháp luật Việt Nam

**Dự án:** RAG_LAW_Vnew · **Căn cứ dữ liệu:** UTS_VLC (`data/uts_vlc/`) · **Ngày phân tích:** 2026-11-13

---

## 1. Tóm tắt điều hành

Mục tiêu: xây dựng hệ RAG trả lời câu hỏi về **hiến pháp, bộ luật và luật đang hiệu lực** của Việt Nam (cấp 2 hệ thống pháp luật), mỗi câu trả lời **trích dẫn chính xác** Số hiệu – Điều – Khoản và **link nguồn gốc** (vbpl.vn).

Nền tảng dữ liệu đã có sẵn, chất lượng tốt, phù hợp làm RAG ngay:

| Thành phần | Nội dung | Vai trò trong RAG |
|---|---|---|
| `data/2026-00000-of-00001.parquet` | **306 văn bản** (1 Hiến pháp + 6 Bộ luật + 299 Luật), 25,0 triệu ký tự ≈ **7,8M tokens**, Markdown sạch | **Corpus chính** |
| `database/vn_law.sqlite` | 616 luật/bộ luật: `eff_from`, `eff_to`, trạng thái (CHL/HHL1P/HHL/CCHL), `in_force`, agency | Metadata lọc + suy luận thời gian |
| `metadata/2026_sources.json` | 306 × ~8 link (vbpl, công báo, quốc hội, chinhphu, …) = 2.504 link | Trích dẫn nguồn trong câu trả lời |
| `data/2021/2023/2026_01.*.parquet` | Các bản cũ hơn (110/208/318 docs) | **Không index** — chỉ dùng đối chiếu |

**Quyết định chính:**

1. Chỉ index split `2026` (đã khử trùng, đối chiếu vbpl).
2. Chunk theo **cấu trúc pháp lý** (Điều → Khoản), không cắt cố định.
3. Retrieval **hybrid** (dense embedding + BM25 tiếng Việt) + reranker.
4. Xử lý riêng **quan hệ luật sửa đổi → luật gốc** — đặc thù quan trọng nhất của pháp luật VN.
5. **Fail-closed by design:** mọi tầng (input → phân loại → abstention → output) khi nghi ngờ thì TỪ CHỐI chứ không đoán; hệ thống phải "biết trả lời không" khi không có dữ liệu (mục 3.8–3.11, đã triển khai và kiểm chứng).

**Rối lớn nhất trước khi index (mục 2.4) — ĐÃ GIẢI QUYẾT Phase 0 (2026-11-13):** văn bản `45/2019/QH14` (Bộ luật Lao động) trong split 2026 chứa sai nội dung (thực thể là kế hoạch của UBND tỉnh Lào Cai); đã thay bằng BLLĐ 2019 thật từ split 2021 + 4 lỗi dữ liệu khác được validator phát hiện đã sửa. Dataset 2026 hiện sạch (306/306 pass, 0 FAIL).

---

## 2. Kết quả phân tích dữ liệu

### 2.1 Tổng quan 4 splits

| Split | Docs | Ký tự | Định dạng `id` | Chất lượng | Dùng cho RAG? |
|---|---:|---:|---|---|---|
| `2026` | **306** | 25,0M | Số hiệu (`91/2015/QH13`; Hiến pháp = `Hiến pháp 2013`) | Markdown sạch, 306/306 có header "Số hiệu", 0 trùng id, 0 lỗi encoding | ✅ **Corpus chính** |
| `2026_01` | 318 | 24,2M | slug | 35 title trùng, dính issue #1035 (trùng/sai) | ❌ (đã được `2026` thay thế) |
| `2023` | 208 | 17,9M | slug | 16/208 có "Số hiệu", cấu trúc thô | ❌ |
| `2021` | 110 | 10,7M | slug | 1 doc rỗng, 9/110 có "Số hiệu" | ❌ |

Phân bố type của `2026`: **1** hiến pháp, **6** bộ luật, **299** luật. Năm ban hành từ 1999 → 2026.

### 2.2 Cấu trúc văn bản (phân tích trên 306 docs của `2026`)

- **Header chuẩn** (306/306): `# Tiêu đề` + `**English:**` + `**Số hiệu:**` + `**Ngày hiệu lực:**` + `---`. Header ngắn: median 346 ký tự.
- **Cấu trúc nội dung dạng plain text** (không phải Markdown heading): `Chương I …`, `Phần …`, `Điều N. Tiêu đề điều`.
  - **19.602 dòng "Điều"** → đơn vị chunk tự nhiên.
  - 1.770 khối chương (median ~7.000 ký tự, max 216.811).
- **Độ dài một Điều:** min 7, p25 498, **median 843**, p75 1.437, p95 3.113, max 255.005 ký tự.
  - 13% điều > 2.000 ký tự; 523 điều > 4.000 ký tự.
  - Block 255k ký tự là **legit**: luật sửa đổi thường có duy nhất `Điều 1` chứa toàn bộ nội dung sửa đổi (VD `12/2017/QH14`).
- **Phụ lục:** 57 docs có "phụ lục" — danh mục/bảng biểu, thường là thứ người hỏi tìm → phải index riêng.
- **Luật sửa đổi, bổ sung:** 64 docs (title chứa "sửa đổi, bổ sung"); 105 docs tham chiếu "một số điều của Luật …". 5 docs ngắn nhất đều là luật sửa đổi (1.494–3.859 ký tự).
- **Không có** bảng Markdown, không có ảnh, không có ký tự thay thế (U+FFFD) → làm sạch ít.
- **Độ dài doc:** min 1.494, median 56.244, max 661.872 ký tự (Luật Nhà ở 27/2023/QH15).

### 2.3 Metadata thời gian & trạng thái (từ `vn_law.sqlite`)

Bảng `laws` (616 dòng, catalog đầy đủ hơn corpus):

| Trạng thái | `in_force` | Số |
|---|---|---:|
| Còn hiệu lực (CHL) | 1 | 166 |
| Hết hiệu lực một phần (HHL1P) | 1 | 152 |
| Hết hiệu lực toàn bộ (HHL) | 0 | 271 |
| Chưa có hiệu lực (CCHL) | 0 | 24 |
| (trống) | 1 | 3 |

→ **321 văn bản đang hiệu lực** trong catalog. Có sẵn `issue_date`, `eff_from` (612/616), `eff_to` (251) → phục vụ hỏi "ngày nào có hiệu lực", "hiện còn hiệu lực không".

`vlc_2026_reconcile` (318 dòng, đối chiếu split 2026_01 với vbpl): ok_in_force 227, outdated 54, wrong_code 36, no_code 1.

### 2.4 Vấn đề chất lượng dữ liệu ⚠️

| # | Mức độ | Vấn đề | Xử lý |
|---|---|---|---|
| 1 | ~~**CRITICAL**~~ **ĐÃ SỬA (Phase 0, 2026-11-13)** | **`45/2019/QH14` (Bộ luật Lao động) chứa SAI nội dung**: body thực tế là *Kế hoạch 292/KH-UBND ngày 17/9/2019 của UBND tỉnh Lào Cai* (thực hiện Chỉ thị 52-CT/TU) — 128 dòng, 0 "Điều". Trong khi Bộ luật Lao động thật có 220 điều. Bảng reconcile vẫn đánh dấu `ok_in_force`. | **ĐÃ LÀM:** body thay bằng BLLĐ 2019 thật lấy từ split `2021` của UTS_VLC (`bo-luat-lao-dong`, 220 điều, 17 chương, kiểm chứng Số hiệu + chữ ký Chủ tịch Quốc hội); bản gốc HF lưu tại `data/uts_vlc/data/archive/2026-original-HF.parquet`. Validator phát hiện thêm 4 lỗi thật đã sửa: `19/2012/QH13` (case `QhH13`), `07/2022/QH15` (rác HTML/CSS đầu body), `51/2024/QH15` + `99/2025/QH15` (prefix thừa trong dòng Số hiệu) |
| 2 | Cao | **71 luật CHL + 47 HHL1P có trong catalog vbpl nhưng KHÔNG có trong corpus** (bảng `missing_in_vlc`, 118 dòng) — chủ yếu luật mới 2025/2026 (Giám định tư pháp 105/2025, Quy hoạch 112/2025, Bảo hiểm tiền gửi 111/2025, Tín ngưỡng tôn giáo 07/2026, …) | V1: chấp nhận gap — RAG phải nói rõ "không tìm thấy trong bộ dữ liệu" và chỉ link vbpl. V2: backfill từ `th1nhng0` (raw HTML, cần pipeline làm sạch) |
| 3 | Cao | **Luật chưa được "consolidate"**: luật gốc và luật sửa đổi là 2 văn bản riêng → câu hỏi "Điều X hiện quy định thế nào" cần ghép nhiều nguồn | Thiết kế retrieval amending-aware (mục 3.4) |
| 4 | Trung bình | 9 title trùng (Luật Chứng khoán, Luật Đầu tư công, …) — do nhiều phiên bản cùng tên qua các năm | Phân biệt bằng `doc_num`; index luôn theo Số hiệu |
| 5 | Trung bình | **Snapshot điểm thời gian 2026-06** (fetched 2026-05-29): luật ban hành/sửa đổi sau đó chưa có | Pipeline refresh định kỳ (mục 3.7) |
| 6 | Thấp | 1 doc trong `2021` rỗng, các split cũ cấu trúc thô | Không dùng → không ảnh hưởng |

### 2.5 Nguồn trích dẫn (mục tiêu của citation layer)

Mỗi doc có link tới 6 nguồn (4 chính thức + 2 thương mại): vbpl.vn (918 link: toan-van + pdf + lược đồ), congbao.chinhphu.vn (263 + 233 PDF), vanban.chinhphu.vn (302), vietlaw.quochoi.vn (181), thuvienphapluat.vn (303), luatvietnam.vn (303). 306/306 doc khớp 1:1 với corpus.

---

## 3. Thiết kế kiến trúc RAG

```
                         ┌─────────────────────────────────────────────┐
                         │              INDEX PIPELINE (offline)       │
 UTS_VLC 2026 ──────────►│ validate → parse Cấu trúc → chunk → embed  │
 vn_law.sqlite ─────────►│ (metadata: status, eff_from, links)        │
 2026_sources.json ─────►└──────────────┬──────────────────────────────┘
                                        ▼
                    ┌───────────────────┴───────────────────┐
                    │  Qdrant (vectors + payload filters)   │
                    │  BM25/underthesea (sparse, offline)   │
                    └───────────────────┬───────────────────┘
                                        ▼
 Người hỏi ──► Query ──► Hybrid retrieval (RRF) ──► Reranker ──► LLM
                    ▲                                                │
                    │                                          Citation: Số hiệu,
                    │                                          Điều/Khoản, link vbpl,
                    └──────────────────────────────────────────trạng thái hiệu lực
```

### 3.1 Phạm vi

- **V1:** 306 docs của split `2026` (sau khi sửa 45/2019/QH14), chỉ Luật/Bộ luật/Hiến pháp đang hiệu lực.
- **Ngoài phạm vi V1:** Nghị định, Thông tư, Quyết định (cấp 3–6). **V2** mở rộng bằng dataset `th1nhng0/vietnamese-legal-documents` (153K+ docs mọi loại, raw HTML — cần thêm bước HTML→MD + làm sạch).

### 3.2 Chiến lược chunking (structure-aware) — TRỌNG TÂM

| Unit | Cách tạo | Ước lượng | Metadata gắn kèm |
|---|---|---:|---|
| **A. Điều** | Regex `^Điều\s+(\d+)` đến `Điều` kế tiếp (đã kiểm chứng: 19.602 matches) | ~19.600 | `doc_num`, `title`, `so_hieu`, `phần/chương`, `số điều`, `tiêu đề điều`, `status_name`, `eff_from`, `in_force`, `url_vbpl` |
| **B. Sub-chunk khoản** | Chỉ với điều > ~2.500 ký tự: cắt theo khoản (`1.`, `2.`, … / `a)`, `b)`, …), overlap ~20% | +~1.500–2.500 (13% điều dài + các "Điều 1" của luật sửa đổi) | thêm `so_khoan` |
| **C. Non-article** | Header/preamble (căn cứ ban hành), tiêu đề chương (context), **Phụ lục** (cắt theo phụ lục), phần ký ban hành | ~500–800 | `chunk_kind` = `header|chapter|appendix|signoff` |

Tổng: **~21.000–23.000 chunks**, mỗi chunk 256–512 tokens.

**Nguyên tắc:**
- KHÔNG cắt fixed-window (phá vỡ tham chiếu "khoản 1 Điều 12").
- Mỗi chunk chứa sẵn header ngữ cảnh (Số hiệu + tên luật + vị trí) → embedding không cần query expansion cho phần này.
- Text chunk giữ nguyên chính tả/tên riêng — không normalize xuống thường (Số hiệu và tên cơ quan quan trọng với BM25).

### 3.3 Retrieval hybrid

| Tầng | Lựa chọn | Lý do |
|---|---|---|
| Dense embedding | **`BAAI/bge-m3`** (1.024 dim, đa ngôn ngữ, tốt tiếng Việt) | Câu hỏi pháp luật VN diễn đạt tự do; chạy local được (CPU: vài giờ cho 7,8M tokens; có GPU/API thì nhanh hơn) |
| Sparse | **BM25 + phân tách từ `underthesea`** | Số hiệu (`31/2024/QH15`), tên riêng, thuật ngữ pháp lý khớp lexical tốt hơn dense |
| Fusion | **RRF** (k=60) | Kết hợp 2 top-k, không cần hiệu chỉnh điểm |
| Reranker | `BAAI/bge-reranker-v2-m3` (top-30 → top-6) | Nâng precision đầu vào LLM; chạy local |
| Vector DB | **Qdrant** (Docker Desktop đang có sẵn) — payload filter theo `doc_num/type/year/in_force/status` | ~22k chunks là rất nhỏ; Qdrant local đủ, có sẵn filter + hybrid |
| LLM | API (user có Claude subscription) hoặc local qua Ollama cho V1 nội bộ | Sinh câu trả lời có trích dẫn |

**Query understanding (nhẹ, V1):**
- Detect Số hiệu trong query (`\d{1,3}/\d{4}/(QH\d{2}|CV|PL|CTN|CP)`) → hard-filter payload, tăng độ chính xác khi user hỏi đúng luật.
- Lọc mặc định `in_force = 1` (tắt khi user hỏi về văn bản đã hết hiệu lực — cần thông báo rõ).

### 3.4 Xử lý luật sửa đổi (đặc thù pháp luật VN)

1. **Build quan hệ `amends` (lúc index):** từ title và Điều 1 của 64 luật sửa đổi, extract Số hiệu luật gốc (regex `Luật … số (\d+/\d+/QH\d+)`) → lưu `amends: [doc_num]` vào metadata cả hai chiều (luật gốc: `amended_by: [...]`).
   - Ví dụ: `43/2024/QH15` amends `31/2024/QH15`; `12/2017/QH14` amends `100/2015/QH13`.
   - Luật sửa nhiều luật (VD `03/2022/QH15` sửa Luật Đầu tư công + Luật Đầu tư theo PPP) → list nhiều mục tiêu.
2. **Retrieval amending-aware:** khi chunk được retrieve thuộc luật gốc có `amended_by` ≠ ∅, tự động pull kèm chunk liên quan của các luật sửa đổi (match theo số điều được sửa) vào context.
3. **Prompt:** bắt buộc LLM nêu rõ "theo Điều X của Luật …, được sửa đổi bởi Luật số …" và **không phát minh nội dung** — nếu context chỉ có luật gốc mà không có bản sửa đổi phù hợp, phải nói "cần kiểm tra các luật sửa đổi".
4. **V2:** tạo bản *consolidated view* (ghép luật gốc + các sửa đổi theo điều) — việc này cần NLP đối chiếu số điều, làm riêng sau.

### 3.5 Generation & citation

- **Format câu trả lời:** nội dung → khối trích dẫn: `Số hiệu / Điều – Khoản / link vbpl / trạng thái hiệu lực (tại 2026-06)`.
- **System prompt:** dữ liệu là snapshot 2026-06; chỉ Luật/Bộ luật/Hiến pháp; không phải tư vấn pháp lý; nếu ngoài phạm vi (nghị định, thông tư, địa phương) → từ chối + hướng tới vbpl.vn.
- **Anti-hallucination:** chỉ được viện dẫn Số hiệu/Điều xuất hiện trong retrieved context; trả lời "không có trong dữ liệu" khi confidence thấp.

### 3.6 Đánh giá (evaluation)

Đã triển khai phần lõi — xem **mục 3.11** (golden seed 28 items grounded + 563 generated candidates + harness tự chạy + 51 unit/integration tests). Khi LLM vào thay template generation, bổ sung: LLM-as-judge cho faithfulness/relevance, Recall@5/@10 trên `expected_doc`/`expected_dieu` của golden set.

### 3.7 Pipeline cập nhật (refresh)

- **Tần suất:** hàng tháng (hoặc on-demand).
- **Quy trình:** gọi API vbpl (`api/qtdc/public/doc/all` — cùng nguồn đã dùng dựng sqlite) → diff theo `doc_num` với corpus → fetch toàn văn mới/đổi → **validate body khớp Số hiệu/tiêu đề** (bài học từ lỗi 45/2019) → làm sạch HTML→MD → re-chunk + re-embed chỉ docs thay đổi → bump `snapshot_date`.
- Mỗi index giữ `snapshot_date` payload để citation luôn ghi đúng mốc thời gian.

### 3.8 Phân loại câu hỏi — `QuestionClassifier` (gate 1) · **ĐÃ TRIỂN KHAI**

Nguyên tắc: **rule-first, deterministic, không dùng LLM trên đường gate** — LLM classifier có thể cắm vào sau (`llm_classify`) nhưng lớp rule là pre-filter cứng, không model nào được phép bỏ qua. Mọi khớp chuỗi đều làm trên dạng **folded** (lowercase, bỏ dấu, `đ→d` — NFD không phân rã được `đ`).

**Phát hiện tham chiếu trước tiên** (dùng cho cả filter retrieval và kiểm tra corpus):

| Loại | Cách | Ví dụ |
|---|---|---|
| `doc_refs` | regex Số hiệu `\b\d{1,3}/\d{4}/(QH..|CV|QĐ|NĐ-CP|TT|CTN|PL|UBTVQH)\b` | `31/2024/QH15` |
| `in_scope_refs` | type `QH*`/`CV`/`HP` (mọi khóa QH — khóa lạ như QH99 là "văn bản không tồn tại", xử lý ở corpus gate, KHÔNG coi là sub-law) | `999/1999/QH99` → in-scope form |
| `out_of_scope_refs` | type NĐ/TT/QĐ/PL/CTN… | `100/2019/NĐ-CP`, `09/2021/TT-BTNMT` |
| `law_refs` | index tên luật (tên đầy đủ + tên base bỏ hậu tố `số NNN/NNNN/TYPE`, `năm YYYY`) | "Bộ luật Hình sự" → `100/2015/QH13`; "Hiến pháp 2013" → `HIẾN PHÁP 2013` |
| `asked_articles` | regex `điều\s+(\d+)` | "Điều 11" → [11] |

**Thang phân loại (trình tự = ưu tiên, fail-closed):**

| # | Category | Trigger | Sub_type | Hành vi |
|---|---|---|---|---|
| 0 | `injection` | pattern VI+EN ("bỏ qua các hướng dẫn trước…", "ignore all previous…", "system prompt", "you are now"…) | — | `refuse_injection` |
| 1 | `sub_law` | Số hiệu NĐ/TT/QĐ… **hoặc** từ khóa cấp dưới luật ("nghị định", "thông tư", "quyết định"…) kèm tín hiệu pháp lý | `sub_law_doc` | `refuse_scope` |
| 2 | `legal_advice` | marker vụ việc cá nhân ("tôi nên", "trường hợp của tôi", "đất của tôi"…) + ngữ cảnh (tranh chấp, ly hôn, phạt…) | `personal_case` | trả lời **bắt buộc caveat** "KHÔNG phải tư vấn pháp lý" |
| 3 | `non_legal` | không có doc_refs/law_refs/asked_articles và không có từ vựng pháp lý | — | `refuse_non_legal` |
| 4 | `answer_candidate` | còn lại | `status` ("còn hiệu lực không") → `effective_date` ("có hiệu lực từ khi nào") → `amendment` (hỏi **mối quan hệ** sửa đổi: "đã được sửa đổi bởi luật nào?", regex `duoc sua doi / bien luat nao / sua doi bo sung mot so dieu`) → `article_lookup` (có `điều N`) → `overview` | vào abstention engine |

Quy tắc then chốt đã kiểm chứng:
- **"hiện hành" ≠ hỏi quan hệ sửa đổi.** "Điều 3 BLHS *hiện hành* quy định gì" là câu hỏi **nội dung** (article_lookup + caveat hợp nhất), không phải câu hỏi đồ thị — nếu nhầm sang đồ thị sẽ trả lời sai dạng.
- **Mixed refs fail-closed:** query vừa có Số hiệu QH vừa có Thông tư/NĐ → `sub_law` (từ chối ở scope gate) thay vì trả lời nửa vời.

### 3.9 Abstention engine — `decide()` (gate 2) · **ĐÃ TRIỂN KHAI**

Nhập **3 tín hiệu độc lập** thành 1 quyết định fail-closed: **(1) classification** (phạm vi + ý định), **(2) corpus** (văn bản có thật trong index không? có lỗi dữ liệu không?), **(3) retrieval** (có tìm được nội dung thực sự trả lời không?).

**6 hành vi:**

| Action | Ý nghĩa |
|---|---|
| `answer` | trả lời với context đã retrieve |
| `answer_with_caution` | trả lời nhưng **bắt buộc** kèm caveat được nêu |
| `refuse_scope` | ngoài phạm vi hệ thống (NĐ/TT/QĐ) |
| `refuse_non_legal` | không phải câu hỏi pháp luật |
| `refuse_no_data` | đúng phạm vi nhưng **không có dữ liệu tin cậy** (thiếu văn bản, lỗi dữ liệu, số hiệu bịa, score thấp, điều không được phủ) |
| `refuse_injection` | query tấn công |

**Thang gate (theo thứ tự trong `decide()`):**

1. injection → non_legal → sub_law: từ chối bằng **thông báo canned** (cố định, không do LLM sinh).
2. **Corpus gate trên từng Số hiệu được hỏi:**
   - `lookup` không tìm thấy (`999/1999/QH99`) → `refuse_no_data`, message nêu đúng Số hiệu.
   - `data_error` (body sai nội dung — đã từng là 45/2019/QH14, giờ đã sửa) → `refuse_no_data`. Cơ chế `corpus.DATA_ERROR_DOCS` giữ lại cho doc lỗi trong tương lai (hiện rỗng; validator `scripts/validate_corpus.py` là cách phát hiện).
   - Có trong catalogue nhưng **không có toàn văn** (105/2025/QH15…) → từ chối, **trừ** câu hỏi status/date → cho phép **metadata-only answer** (sqlite là nguồn chuẩn cho thời gian/trạng thái vì 298/306 doc thiếu header ngày hiệu lực).
3. **Metadata answers độc lập với retrieval** (deterministic):
   - `status`/`effective_date` → trả lời thẳng từ catalogue: trạng thái, ngày ban hành/hiệu lực, link vbpl; luật hết hiệu lực kèm **"Được thay thế bởi"** (thực thể `replaces`: cùng base title + cùng type + năm sau + đang hiệu lực, VD 45/2013/QH13 → 31/2024/QH15).
   - `amendment` (hỏi quan hệ) → trả lời từ đồ thị `amends`/`amended_by` (build từ Số hiệu trong title của 64 luật sửa đổi, VD 43/2024/QH15 → [31/2024, 27/2023, 29/2023, 32/2024]).
4. **Retrieval gate** (chỉ với câu hỏi nội dung):
   - `top.score < MIN_SCORE (0.12)` — hoặc `0.20` khi hỏi thẳng 1 điều — → `refuse_no_data` (kèm score vào lý do).
   - **Coverage:** hỏi "Điều N" mà không hit nào phủ Điều N → `refuse_no_data` ("không tìm thấy Điều N của văn bản được hỏi").
5. **Consolidation caveat (bắt buộc):** hỏi nội dung điều / "hiện hành" của luật **đã bị sửa** (amended_by ≠ ∅) → `answer_with_caution` + caveat liệt kê luật sửa đổi (VD BLHS → 12/2017/QH14). Corpus KHÔNG consolidate — caveat này là tính mạng của hệ (bài học mục 2.4).

**Retriever stand-in (`LexicalRetriever`):** chạy end-to-end **trước khi có embedding index** (deterministic, offline): IDF-weighted token overlap / độ dài, bonus **+15** cho đúng điều được hỏi (ràng buộc cứng), bonus tiêu đề = Σ IDF theo độ hiếm, chuẩn hóa √độ dài tiêu đề (tránh tiêu đề hành chính dài "ăn gian" điểm), bỏ token Số hiệu khỏi query (đã có doc-filter), lọc theo `in_scope_refs ∪ law_refs`. Điểm thay-swap cho hybrid bge-m3+BM25+reranker ở mục 3.3.

### 3.10 Cơ chế an toàn — layered, fail-closed · **ĐÃ TRIỂN KHAI**

```
Query ──► [0] InputGuard ──► [1] QuestionClassifier ──► Retrieval ──► [2] decide()
                                                                      │
        AuditLog (JSONL append-only, mỗi query 1 dòng) ◄──────────────┤
                                                                      ▼
   [3] OutputValidator  ◄── compose answer (template/LLM) + finalize (caveat + DISCLAIMER)
        │ pass  → xuất bản          │ fail → KHÔNG xuất bản, chuyển refuse_no_data
        ▼
     Người dùng (mọi câu trả lời luôn kèm DISCLAIMER + trích dẫn + snapshot date)
```

| Tầng | Kiểm tra | Vi phạm |
|---|---|---|
| **0 — InputGuard** | rỗng; quá dài (>1500 ký tự); **injection** (hard block, pattern VI+EN); không phải tiếng Việt (tỷ lệ ký tự có dấu < 3% trên ≥24 ký tự — soft warning) | injection → `refuse_injection` canned |
| **1 — Classifier** | như mục 3.8 | từ chối scope/non-legal |
| **2 — decide()** | như mục 3.9 | từ chối no-data + lý do cụ thể |
| **3 — OutputValidator** | (a) **DISCLAIMER bắt buộc** trong mọi câu trả lời (snapshot 06/2026, scope, không phải tư vấn, kiểm tra vbpl.vn); (b) caveat phải có mặt với `answer_with_caution`; (c) **refusal phải đúng 100% thông báo canned** + mang marker từ chối — LLM/sinh nào "biến tấu" lời từ chối đều bị chặn (`refusal_message_modified`, `refusal_marker_missing`); (d) **citation validation** (dưới) | mọi vi phạm → answer **không được xuất bản**, chuyển `refuse_no_data` "mâu thuẫn trong trích dẫn" + ghi warnings |

**Anti-hallucination citation (thực thể `validate_answer`):**
- **Số hiệu:** Số hiệu xuất hiện trong answer phải thuộc allowed set = `refs_for_citation` (docs đã retrieve ∪ refs query ∪ đồ thị sửa đổi/thay thế + Số hiệu trong title của chúng) **∪ Số hiệu xuất hiện trong retrieved context** (trích dẫn verbatim chương cuối của Luật Đất đai có quyền nêu 15/2023, 16/2023, 28/2023 vì trích nguyên văn) → vi phạm: `fabricated_ref:NNN/NNNN/TYPE`.
- **Điều:** "Điều N" trong answer phải có mặt trong context (khớp dạng **folded** — miễn nhiễm NFC/NFD của text corpus) → vi phạm: `unsupported_article:N`. LLM không thể bịa "Điều 999" của luật 260 điều.
- Hệ quả thiết kế: lớp generation (template V1, LLM V1.1) **không bao giờ** tự do — output không qua validator thì không ra khỏi hệ.

**AuditLog:** JSONL append-only, mỗi query 1 dòng: `ts, query(≤300), category, sub_type, action, reasons, caveats, warnings, top-5 hits (doc/dieu/score), latency_ms`. Cho phép kiểm toán "hệ đã từ chối câu nào, vì sao" — yêu cầu bắt buộc với sản phẩm pháp lý.

### 3.11 Benchmark & eval harness — 3 tầng · **ĐÃ TRIỂN KHAI**

**Ba tầng, ba mục đích:**

| Tầng | Nội dung | Chạy khi | Chi phí | Runner / gate |
|---|---|---|---|---|
| **1. Retrieval** | question → chunk: top-5 phải chứa đúng `(expected_doc, expected_dieu)`; điều của item `REFUSE_NO_DATA` phải **KHÔNG** có hit phủ (cơ sở từ chối ở tầng retrieval) | mỗi commit | ~0 (không LLM) | `python -m benchmarks.run_retrieval` — ghim trong pytest |
| **2. End-to-end** | question → answer: pipeline production đầy đủ, chấm action + facts so với golden | trước mỗi thay đổi lớn | trung bình | `python -m benchmarks.run_eval` (exit 0 chỉ khi 100%) |
| **3. Regression** | **mọi bug production đã gặp** — mỗi bug = 1 item ghim (action + `must_contain`/`must_not_contain`, dạng folded) | mỗi commit | thấp | `python -m benchmarks.run_regression` — ghim trong pytest |

**CI:** `pytest` (`tests/test_benchmarks.py`) tự chạy tầng 1 + 3 mỗi commit; tầng 2 chạy manual trước thay đổi lớn.

```
benchmarks/
├── data/                # dataset input — chỉ đọc khi eval
│   ├── golden_seed.jsonl            28 items grounded (tầng 1 + 2)
│   ├── regressions.jsonl            12 bug ghim (tầng 3) — CHỈ ĐƯỢC THÊM
│   └── generated_candidates.jsonl   563 candidates (review: true)
├── runs/                # artifact khi chạy (audit JSONL, báo cáo --json) — tự tạo
├── README.md            # hướng dẫn 3 tầng + quy tắc
├── eval_lib.py          # lib dùng chung (DATA/RUNS, ACTION_ALIASES, tier runners)
├── run_retrieval.py / run_eval.py / run_regression.py   # 3 runner
└── make_generated.py    # sinh generated_candidates.jsonl (seed 7, deterministic)
```

| File | Vai trò |
|---|---|
| `data/golden_seed.jsonl` | **28 items grounded** — nguồn cho tầng 1 + 2; mọi `answer_facts` đã verify với text corpus/sqlite thật (VD "đất đai thuộc sở hữu toàn dân" = Điều 12 Luật 31/2024; "18 năm tù" = Điều 101 BLHS; "2027-01-01" = sqlite 05/2026/QH16) |
| `data/regressions.jsonl` | **12 bug thật đã gặp** ghim thành item (reg-001…reg-012: fold đ/Đ, NFC/NFD citation, law-name index, article bonus, amendment marker, "hiện hành" ≠ quan hệ, mixed refs fail-closed, body lỗi 45/2019, QH lạ, metadata CCHL, replaced-by, validator refusal). **Quy tắc: file chỉ được phép lớn dần — mọi bug mới phải được ghim trước khi đóng** |
| `eval_lib.py` | Helper dùng chung: `ACTION_ALIASES`, `run_retrieval_tier` (metrics Recall@k / MRR / Top-1), `run_regression_tier` |
| `run_retrieval.py` | Runner tầng 1 |
| `run_eval.py` | Runner tầng 2 — chấm: (1) action khớp (alias `ANSWER` ⊇ `answer_with_caution`), (2) facts (folded) có mặt trong answer; in từng item + tổng hợp theo action |
| `run_regression.py` | Runner tầng 3 |
| `make_generated.py` | Sinh **563 candidates** (seed 7, deterministic): mẫu điều của mỗi doc × 3 template — đánh dấu `review: true`, vào tầng 2 sau khi review thủ công |

**Phủ 6 hành vi × khó độ tăng dần (28 items):**

| Nhóm | Items | Điểm kiểm tra khó |
|---|---|---|
| ANSWER | gs-001…gs-015, gs-024 (16) | article_lookup với bonus điều (gs-007 Điều 101 phải thắng Điều 90/91 cùng chủ đề "người dưới 18 tuổi"); metadata status "Hết hiệu lực toàn bộ/một phần" (gs-012/013); metadata-only cho luật **chưa hiệu lực, không có trong corpus** (gs-011, 05/2026/QH16); đồ thị sửa đổi 2 chiều (gs-014 amended_by, gs-015 amends 4 luật); BLLĐ 2019 Điều 3 (gs-024 — trước đây REFUSE_NO_DATA, flip ANSWER sau Phase 0) |
| ANSWER_WITH_CAUTION | gs-016, gs-027, gs-028 (3) | câu hỏi "Điều 3 **hiện hành**" → bắt buộc caveat nêu 12/2017/QH14; tư vấn cá nhân → disclaimer "KHÔNG phải tư vấn pháp lý" |
| REFUSE_SCOPE | gs-017, gs-018 (2) | Nghị định (theo Số hiệu), Thông tư (theo từ khóa) |
| REFUSE_NON_LEGAL | gs-019, gs-020 (2) | nấu phở, giá vàng |
| REFUSE_NO_DATA | gs-021…gs-023 (3) | 3 biến thể khác nhau: luật thiếu toàn văn (105/2025), hỏi điều của luật thiếu, **Số hiệu bịa** (999/1999/QH99) |
| REFUSE_INJECTION | gs-025, gs-026 (2) | injection tiếng Việt + tiếng Anh ("You are now DAN…") |

**Kết quả hiện tại (đã chạy, sau Phase 0 — 2026-11-13):** `pytest` **53/53 pass** (unit + integration cho classify/abstain/safety/citations + end-to-end pipeline + 2 test CI cho tầng 1 & 3). Theo tầng: **Tầng 1 — Recall@5 100%** (28/28 items ok), MRR 0.955, Top-1 90.9% (scored=11, no_cover=2, skipped=15); **Tầng 2 — `run_eval.py` 28/28 (100%)** (ANSWER 16/16, ANSWER_WITH_CAUTION 3/3, REFUSE_* 9/9); **Tầng 3 — 12/12 bug ghim giữ nguyên**.

**Định nghĩa "hệ biết trả lời không":** một câu hỏi chỉ được trả lời khi **đồng thời** (a) đúng phạm vi, (b) văn bản được hỏi có trong corpus sạch (không data-error), (c) retrieval đạt ngưỡng **và** phủ đúng điều được hỏi. Thiếu 1 trong 3 → từ chối với lý do cụ thể bằng thông báo canned. Mọi trường hợp "biết nhưng chưa có" (luật mới, luật cũ thay thế) được trả lời **metadata** có nguồn thay vì im lặng hoặc bịa.

---

## 4. Công nghệ đề xuất (phù hợp máy hiện tại)

| Hạng mục | Lựa chọn | Ghi chú |
|---|---|---|
| Runtime | Python 3.11 qua **uv** (venv project-level) | Python hệ thống là Store stub — luôn dùng uv |
| Đọc dữ liệu | `pyarrow` | |
| Parse/chunk | Python thuần (regex đã kiểm chứng trên 306 docs) | |
| Embedding | `sentence-transformers` + `BAAI/bge-m3` | Tải model qua proxy corporate; ~2GB |
| Word segmentation | `underthesea` | cho BM25 |
| Sparse | `rank_bm25` | |
| Reranker | `BAAI/bge-reranker-v2-m3` | |
| Vector DB | **Qdrant** via Docker Desktop (đã cài, WSL2) | ~22k chunks → ~1GB RAM |
| LLM | API (Claude — subscription sẵn có) hoặc Ollama local | |
| UI | Streamlit (chat + hiển thị trích dẫn + filter) hoặc FastAPI + Web UI | |
| CI chất lượng dữ liệu | Script validate chạy trước mỗi lần ingest | Kiểm tra: title/Số hiệu xuất hiện trong body, độ dài tối thiểu, count "Điều" > ngưỡng theo type |

Môi trường đã xác nhận: uv 0.13.6 + CPython 3.11.18, Docker Desktop 4.97 (WSL2), không có internet trực tiếp — mọi download (PyPI/model) đi qua proxy `http://10.26.4.55:8080` (`UV_HTTP_PROXY`), và cần `PYTHONUTF8=1` khi chạy script xử lý tiếng Việt.

---

## 5. Lộ trình thực hiện

| Phase | Nội dung | Deliverable | Ước lượng |
|---|---|---|---|
| **0. Validate data** ✅ **XONG (2026-11-13)** | (a) Thay body `45/2019/QH14` bằng BLLĐ 2019 thật (lấy từ split 2021 UTS_VLC — không cần mạng); (b) viết validator body↔Số hiệu `scripts/validate_corpus.py` chạy toàn bộ 306 docs; (c) thống kê gap so `missing_in_vlc` | ✅ Dataset `2026` sạch (306/306 pass, 0 FAIL, 1 WARN) + báo cáo `benchmarks/runs/validate_report.json`. **Validator chạy trên mọi snapshot mới trước ingest** | 0,5–1 ngày |
| **1. Index pipeline** | Parser Điều/Khoản/Phụ lục → chunk + metadata (gộp sqlite + sources) → embed bge-m3 → nạp Qdrant + BM25 | ~22k chunks truy vấn được; script ingest tái chạy được | 2–3 ngày |
| **2. Retrieval + amending** | Hybrid RRF + reranker; bảng quan hệ `amends/amended_by`; filter Số hiệu/in-force | Retrieval trả về đúng điều + luật sửa đổi liên quan | 2–3 ngày |
| **3. Generation + eval** | Prompt + citation format; golden set 50–100 QA; đo recall/citation/faithfulness | Báo cáo eval + fix | 2–3 ngày |
| **4. UI + refresh** | Streamlit chat với trích dẫn & filter; pipeline refresh vbpl (thủ công chạy, chưa tự động hóa) | Hệ demo chạy được end-to-end | 2 ngày |

Tổng: **~1,5–2 tuần** (làm part-time).

**Trạng thái triển khai (cập nhật 2026-11-13):** **Phase 0 XONG** — body `45/2019/QH14` đã thay bằng BLLĐ 2019 thật (220 điều, nguồn split 2021) + 4 lỗi dữ liệu khác được validator phát hiện đã sửa + `scripts/validate_corpus.py` chạy 306/306 pass. Các thành phần **an toàn / phân loại / abstention / benchmark** (mục 3.8–3.11) code đầy đủ và kiểm chứng (53/53 tests + 28/28 golden e2e 100% + retrieval tier Recall@5 100% + 12/12 regression) — pipeline end-to-end chạy được trên retriever lexical trước khi có embedding index. Còn lại: **Phase 1–2** (embed bge-m3 + BM25 + Qdrant, thay `LexicalRetriever`), sau đó cắm LLM vào `_compose_answer` (giữ nguyên mọi gate + validator).

**Definition of Done (V1):** hỏi 10 câu mẫu về điều khoản cụ thể, ngày hiệu lực và luật đã sửa đổi → ≥8 câu trả lời đúng nội dung, citation đúng Số hiệu/Điều, link vbpl mở được, và hệ từ chối đúng với câu hỏi về nghị định/thông tư. *(4/4 nhóm từ chối đã đạt 100% ngay trên golden set hiện tại.)*

---

## 6. Rủi ro & giới hạn

| Rủi ro | Tác động | Mitigation |
|---|---|---|
| Snapshot 2026-06: luật mới sau đó thiếu | Trả lời thiếu/với trạng thái cũ | Ghi `snapshot_date` trong mọi citation; refresh hàng tháng; sqlite catalog đã có 24 luật CCHL (hiệu lực 2026-07+) để cảnh báo |
| Chưa consolidate luật sửa đổi | Câu hỏi "điều hiện hành" sai nếu bỏ sót bản sửa | Amending-aware retrieval + prompt yêu cầu nêu luật sửa đổi; V2 làm consolidated view |
| 71+47 luật hiệu lực chưa có trong corpus (missing_in_vlc) | Không trả lời được cho các luật mới 2025/2026 | Flag "không có trong bộ dữ liệu" + link vbpl; V2 backfill từ th1nhng0 |
| Lỗi body sai như 45/2019/QH14 có thể còn ở doc khác | Trả lời sai nghiêm trọng | **ĐÃ GIẢM (Phase 0):** validator body↔Số hiệu chạy 306/306 docs — chỉ còn 1 WARN (61/2024 thiếu preamble, điều vẫn đủ). Validator bắt buộc trên mọi snapshot mới trước ingest |
| Phạm vi chỉ cấp 2 (không có nghị định/thông tư) | Người hỏi về văn bản hướng dẫn sẽ thất vọng | Thông báo rõ scope + hướng dẫn tới vbpl.vn |
| Không phải tư vấn pháp lý | Rủi ro pháp lý cho user | Disclaimer cố định trong mọi câu trả lời |

---

## Phụ lục A — Số liệu khai phá (nguyên liệu)

- 2026 split: 306 docs, 25.006.877 ký tự, ~5,49M từ, ~7,8M tokens; type: 1 HP / 6 BL / 299 Lu.
- 6 Bộ luật: Dân sự 91/2015 (689 điều), Hàng hải 95/2015 (341), Hình sự 100/2015 (425), TT dân sự 92/2015 (519), TT hình sự 101/2015 (498), Lao động 45/2019 (220 điều, 17 chương — đã sửa Phase 0: body thật từ split 2021).
- Hiến pháp 2013: 120 điều, 11 chương, 61.108 ký tự.
- 5 docs dài nhất: Nhà ở 27/2023 (661.872), Hình sự 100/2015 (563.173), TT hình sự 101/2015 (526.610), Đất đai 31/2024 (525.808), TT dân sự 92/2015 (499.799).
- sqlite: `laws` 616 rows (1997–2026), `law_links` 616, `missing_in_vlc` 118 (71 CHL + 47 HHL1P), `vlc_2026_reconcile` 318, `legal_sources` 6, `meta` 5 (fetched 2026-05-29).
- Sources JSON: list 306 items, 2.504 link; domain: vbpl 918, thuvienphapluat 303, luatvietnam 303, vanban.chinhphu 302, congbaocdn 263, datafiles.chinhphu 233, vietlaw.quochoi 181.

## Phụ lục B — Câu hỏi mở cần quyết định trước Phase 1

1. **LLM cho V1:** dùng API Claude (subscription sẵn có) hay local Ollama? (Ảnh hưởng: tốc độ, chi phí, offline.)
2. **UI:** Streamlit nhanh hay FastAPI + front-end riêng?
3. **Backfill 71 luật thiếu (missing_in_vlc):** làm ngay trong V1 (cần pipeline HTML→MD từ th1nhng0) hay để V2?
4. **Ngân sách embedding:** chạy bge-m3 local CPU (vài giờ) hay dùng embedding API để tiết kiệm thời gian?
