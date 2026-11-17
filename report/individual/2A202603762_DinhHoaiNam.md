# Báo cáo Day 7 — Agentic RAG

**Sinh viên:** Đinh Hoài Nam · **MSSV:** 2A202603762 · **Lớp:** K5-L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Kho tri thức tái sử dụng từ
  **Day 6**: hai tiểu thuyết Alice của Lewis Carroll — *Alice's Adventures in
  Wonderland* (Project Gutenberg #11) và *Through the Looking-Glass* (#12), đều
  **public domain**. Văn bản chuẩn hoá: pg11 143.411 ký tự / 55 đơn vị, pg12
  159.892 ký tự / 64 đơn vị (tổng 117 đơn vị nội dung "trang"). Chunk dùng
  **strategy C** đã chốt ở Day 6 (paragraph, tối đa 1400 ký tự) → **273 chunk**.
  Mỗi chunk giữ `id` dễ đọc (`pg11_u001_c1`), `source_id`, `source` (URL
  Gutenberg), `page` (số đơn vị), `metadata` (book_title, chapter_number,
  section, page_id, offset, sha256 nguồn, và `day6_chunk_id` gốc).
- **Vector store, embedding model, phiên bản và số chiều:** Embedding
  **`BAAI/bge-small-en-v1.5`** chạy cục bộ qua **FastEmbed (ONNX, CPU)**,
  **384 chiều**, khoảng cách **cosine** — đúng model Day 6. Query dùng tiền tố
  bất đối xứng của bge (`"Represent this sentence for searching relevant
  passages: "`), document không tiền tố. Vector store **Qdrant cục bộ**
  (`.local/qdrant`, collection `day7_alice_bge_small`, 273 điểm, exact search).
- **Agent model, cách chạy và phân chia dev/test:** Controller agent dùng
  **model thật có tool-calling** — **Groq `openai/gpt-oss-120b`** (API
  OpenAI-compatible, gọi qua `httpx`, `temperature=0`, free tier). Benchmark tái dùng bộ câu hỏi đã đóng
  băng của Day 6 (Day 6 chưa dùng test để chỉnh — `test_used_for_tuning:false`)
  và bổ sung câu thiếu bằng chứng:
  - **dev:** 15 câu có đáp án (D01–D15) + 2 câu thiếu bằng chứng (U01, U02) = 17.
  - **test:** 5 câu có đáp án (T01–T05) + 2 câu thiếu bằng chứng (U03, U04) = 7.
  - Tổng 24 câu (đạt tối thiểu 20+4). Cấu hình được **chốt trên dev trước** khi
    chạy test (xem tiêu chí bên dưới).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** Câu có đáp án mang
  `gold = [{source_id, page, evidence}]`; "hit" = evidence nằm **trọn trong một
  chunk** đúng nguồn/trang (hàm `hit()` của starter). Ba câu Day 6 có evidence
  vắt qua ranh giới chunk strategy C (D04, D06, T04) được thay bằng **đoạn con
  xác thực nằm trong một chunk**, ghi rõ trong `prepare_data.GOLD_OVERRIDES`
  (D04/D06 có đáp án hai phần nên liệt kê cả hai đoạn làm evidence thay thế).
  Câu thiếu bằng chứng (U01–U04) gần chủ đề nhưng **đã kiểm tra là vắng mặt**
  trong văn bản gốc (ví dụ "10/6" và "ten and six" đếm = 0; "surname"/"last
  name" = 0); agent phải **từ chối**. Câu thiếu bằng chứng **không** nằm trong
  mẫu số của retrieval metrics.

> Dữ liệu nằm ở `data/day7/{chunks,questions,agent_cases}.json`; không đưa `gold`
> vào agent hay retrieval tool.

## Retrieval và thử nghiệm

**Ba phương pháp khác nhau** (gồm dense trên vector store Day 6):
1. **Dense** — query embedding bge-small, tìm vector gần nhất (Qdrant, cosine).
2. **BM25** — Okapi BM25 (k1=1.5, b=0.75) thuần stdlib trên cùng chunk.
3. **Hybrid RRF** — hợp nhất xếp hạng dense + BM25 bằng Reciprocal Rank Fusion
   (`score = Σ 1/(60+rank)`), không cần chuẩn hoá điểm giữa hai thang.

**Hai (ba) cấu hình nhằm cải thiện** baseline: suy luận metadata filter từ câu
hỏi (`dense_filter`), retrieve-then-rerank lexical (`dense_rerank_bm25`), và nâng
context budget 800 → 3000 ký tự (áp dụng cho các cấu hình cải thiện; `hybrid_wide_ctx`
còn tăng top-k 3 → 5).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median) |
|---|---|---|---|---|---|---|---|
| Baseline (BM25) | BM25 lexical, k=3, budget 800 | Baseline lexical | dev | 11/15 (0.73) | **15/15 (1.00)** | 1/15 (0.07) | 3.6 ms |
| Dense | Dense bge-small, k=3, budget 800 | Vector store Day 6 | dev | **12/15 (0.80)** | 13/15 (0.87) | 2/15 (0.13) | 55 ms |
| Hybrid RRF | Dense+BM25 RRF, k=3, budget 800 | Hợp nhất 2 tín hiệu | dev | 11/15 (0.73) | **15/15 (1.00)** | 1/15 (0.07) | 77 ms |
| Cải thiện 1 — dense_filter | Dense + filter suy luận, k=3, **budget 3000** | Filter chương/sách + budget lớn giúp evidence vào context | dev | 12/15 (0.80) | 13/15 (0.87) | 12/15 (0.80) | 52 ms |
| Cải thiện 2 — dense_rerank_bm25 | Dense over-fetch 12 → rerank BM25 → top 3, **budget 3000** | Lexical rerank sửa nhầm đoạn | dev | 11/15 (0.73) | 14/15 (0.93) | 14/15 (0.93) | 50 ms |
| **Đã chọn — hybrid_wide_ctx** | **Hybrid RRF, k=5, budget 3000** | Hybrid + context rộng | dev | 11/15 (0.73) | **15/15 (1.00)** | **15/15 (1.00)** | 56 ms |
| Đã chọn — hybrid_wide_ctx | Hybrid RRF, k=5, budget 3000 | (độc lập) | **test** | 4/5 (0.80) | 4/5 (0.80) | 4/5 (0.80) | 64 ms |
| Baseline (BM25) | BM25 lexical, k=3, budget 800 | so sánh test | **test** | 4/5 (0.80) | 4/5 (0.80) | 0/5 (0.00) | 4.2 ms |
| Dense | Dense, k=3, budget 800 | so sánh test | **test** | 1/5 (0.20) | 5/5 (1.00) | 0/5 (0.00) | 56 ms |

Chi phí dịch vụ của retrieval = **0 USD** (embedding + vector store cục bộ).
`n/N` và median latency đo trên cùng corpus/index/câu hỏi; mẫu số chỉ gồm câu có
đáp án. Context Evidence Rate đo trên context đóng gói bởi `pack_context` ở budget
tương ứng.

**Tiêu chí chọn (công bố trước khi chạy test):** ưu tiên (1) Evidence Hit@3 →
(2) Context Evidence Rate → (3) MRR → (4) độ trễ/chi phí. `hybrid_wide_ctx` đứng
đầu mọi tiêu chí evidence trên dev (Hit@3 15/15, Context Rate 15/15, MRR 0.867)
nên được chốt; sau đó mới chạy test.

**Hai lỗi dev — nguyên nhân, cách thử, lý do chọn:**

- **D09** ("How does Alice work out how to read the reversed words in the
  Looking-glass book?", evidence ở pg12 trang 6). **Dense trượt top-3** (lấy
  trang 28/3/37, điểm ~0.68): cụm "Looking-glass book" xuất hiện lặp theo chủ đề
  khắp cuốn #12 nên vector query bị **trôi ngữ nghĩa**. **BM25 bắt đúng** (điểm
  12.4 nhờ khớp cụm từ hiếm), và **Hybrid khôi phục** đưa trang 6 vào top-3. →
  Bài học: QA văn học trích nguyên văn cần tín hiệu lexical; dense đơn lẻ không đủ.
- **D13** ("In chapter 5 of Through the Looking-Glass, what is the White Queen's
  rule for which days jam can be had?", evidence pg12 trang 25). Dense lấy đúng
  **trang** (26/25/27) nhưng **sai đoạn-chunk** (chunk trang 25 được lấy không
  chứa câu evidence) → hit=False. BM25/Hybrid chọn đúng chunk. Filter suy luận
  (chapter 5 + book) cũng là hướng sửa. → Bài học: với chunk nhỏ, dense đúng
  trang nhưng lệch đoạn; lexical/filter phân giải đoạn chính xác.

**Phát hiện cấu hình quan trọng:** ở budget mặc định 800, chunk ~1.4k ký tự
**không vừa** context nên Context Evidence Rate rất thấp (dense 2/15, BM25 1/15)
dù Hit@3 cao — nghĩa là evidence tìm được nhưng **không đến được model**. Nâng
budget 800 → 3000 kéo Context Rate dense_filter 0.13 → 0.80, hybrid 0.07 → 1.00.

**Đánh đổi:** BM25 rẻ nhất (~4 ms) và Hit@3 bằng hybrid trên dev, nhưng thiếu khả
năng khớp ngữ nghĩa cho câu diễn giải; dense có Hit@1 tốt nhất nhưng Hit@3 kém và
trôi trên từ vựng lặp; hybrid cộng ~20 ms so với dense nhưng ổn định nhất. Trên
**test** (tập nhỏ, phương sai cao) dense lại đạt Hit@3 5/5 còn hybrid 4/5 (trượt
**T04** — evidence ở đoạn-chunk thứ hai của trang 54, xếp hạng 5): minh hoạ rằng
cấu hình chọn trên dev **không đảm bảo tối ưu trên test**, và được báo cáo trung
thực theo yêu cầu.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** Agent
  dùng khung `starter/agentic.run_agent` với controller Groq. Model có 2 tool:
  `search(query)` (một lượt retrieval = một event `search`) và
  `provide_answer(answer, citations)`. Điều kiện dừng: model gọi `provide_answer`,
  hoặc chạm `max_searches=4`, hoặc citation không hợp lệ/không có context →
  **abstain**. Context chọn theo hybrid top-5, `pack_context` **budget 4000**
  (nâng từ 3000 để chứa evidence tích luỹ qua nhiều lượt), giữ đoạn từ các lượt
  trước, dedup theo ID. **Lượt search đầu dùng nguyên văn câu hỏi** (trên dev,
  query do model tự viết lại retrieve kém hơn — D09, D15). Citation ảo (ID không
  có trong context) bị loại trước khi trả lời; lỗi provider/retriever **được ghi
  `status:"error"`**, không thay bằng kết quả giả. Cấu hình agent đã chốt =
  **hybrid, k=5, budget 4000**.

### Kết quả agent (nhãn do người chấm, đối chiếu evidence bản gốc)

| Cấu hình / split | Tool use | Số lượt gọi search | Grounded 0/1/2 · TB (/2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Đã chọn — hybrid / dev** | 17/17 | 12×1, 5×4 | 3/0/12 · **1.60** | 13/13 | 12/12 | 2/2 | 3/15 |
| **Đã chọn — hybrid / test** | 7/7 | 4×1, 3×4 | 1/0/4 · **1.60** | 4/4 | 4/4 | 2/2 | 1/5 |
| Baseline (BM25, budget 800) / test | 7/7 | 7×4 | 5/0/0 · **0.00** | 0/0 (n/a) | 0/0 | 2/2 | 5/5 |

Chi phí: **0 USD** (Groq free tier; embedding/vector store cục bộ). Token:
dev hybrid 46.3k prompt + 5.8k completion (32 lượt gọi model); test hybrid
22.4k + 3.0k (16 lượt); test baseline 16.5k + 3.0k (28 lượt). Độ trễ bị chi
phối bởi backoff theo rate-limit free-tier (TPM 8000) nên không báo như chất
lượng.

**Tương phản baseline:** BM25 budget 800 **từ chối toàn bộ 5 câu test có đáp
án** (grounded 0.00) dù Hit@3 của BM25 cao — vì chunk ~1.4k ký tự không vừa
context 800 → context rỗng → agent không thể trích dẫn. Đây là bằng chứng
agent-level cho phát hiện budget ở phần retrieval; cấu hình chọn (budget 4000)
đạt grounded **1.60/2** trên cả dev và test.

**Ba ví dụ (có bằng chứng · thiếu bằng chứng · lỗi):**

| Query | Loại | Tool call & kết quả | Câu trả lời & citation | Nhãn, lý do |
|---|---|---|---|---|
| T01 | Có đáp án | 1 search → pg11_u042_c1 (đúng đoạn) | "…Reeling and Writhing… Ambition, Distraction, Uglification, and Derision [pg11_u042_c1]" | Grounded **2**; citation hỗ trợ (đoạn chứa nguyên văn đáp án). |
| U03 | Thiếu evidence | 4 search → đoạn về White Rabbit, không có giá | "Không đủ bằng chứng để trả lời." | Abstention **đúng**: không đoạn nào nêu giá đồng hồ. |
| D13 | Lỗi (từ chối nhầm) | 4 search → pg11_u053, pg12_u013/u056 (sai trang) | "Không đủ bằng chứng…" | Grounded **0**: retrieval trượt (gold pg12 trang 25 không vào context); filter chương-5 sẽ giúp. |

Ngoài ra, trong một lần chạy mạng rớt giữa chừng, controller ném
`httpx.ConnectError` và runner ghi `status:"error"` cho các câu (không ngụy
tạo kết quả); sau khi mạng phục hồi đã chạy lại sạch — minh hoạ xử lý lỗi thật.

**Trước/sau (cải thiện agent trên dev):** bản đầu (model tự viết query,
budget 3000) từ chối nhầm **{D04, D09, D15}**; sau khi **seed lượt search đầu
bằng câu hỏi gốc** + budget 4000, khôi phục **D09 và D15** (D09 là câu dense
đơn thuần không thể tìm), nhưng **D06, D13** regress sang từ chối. Grounded
trung bình giữ **1.60** — đánh đổi trung thực: cải thiện khôi phục đúng các câu
retrieval khó nhất. **D04/D06** là **từ chối thận trọng** khi chỉ một trong hai
đoạn evidence vào context (artifact do chunk nhỏ) — model không trả lời nửa vời.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** Trace retrieval ở
  `results/retrieval/{dev,test}/`; trace agent ở `results/agent/{split}/{config}/`
  (`full_traces.json`, `traces.json`, `cases.json`, `reviews.json`,
  `summary.json`, `grading_packet.md`). Lệnh chạy ở `src/README.md`.

- **Đánh đổi và giới hạn:**
  - *Đánh đổi:* nâng context budget (800→3000 cho retrieval, →4000 cho agent) là
    thay đổi hiệu quả nhất — baseline BM25 budget 800 từ chối 100% vì context
    rỗng. Thêm lexical (BM25/Hybrid) sửa D09, D13 mà dense trượt ở retrieval;
    seed câu hỏi gốc giúp agent khôi phục D09, D15 (đánh đổi D06, D13).
  - *Giới hạn:* (1) corpus tiếng Anh, tokenizer đơn giản; (2) tập test nhỏ (5 câu
    có đáp án) nên metric phương sai cao; (3) `hit()` khớp nguyên văn — đã bổ sung
    đọc đối chiếu thủ công cho grounded-answer; (4) đáp án hai phần nằm ở hai
    chunk (D04, D06) gây từ chối thận trọng — cần schema multi-evidence hoặc
    chunk lớn hơn; (5) model Groq phụ thuộc mạng và hạn mức free-tier; đặt
    `temperature=0` nhưng chạy lại vẫn có thể lệch nhẹ do LLM.

## Thử thách nâng cao

Không thực hiện (tuỳ chọn). Phần cơ bản là trọng tâm bài nộp.

---

### Trạng thái tái lập

- Test: `python -m unittest discover -s tests -v` (đặt `PYTHONUTF8=1` trên
  Windows) — 15 test starter + 8 test retrieval Day 7 đều PASS.
- Pipeline: `prepare_data.py` → `dense_index.py` → `eval_retrieval.py` →
  `agent_runner.py` → chấm thủ công (`reviews.json`) → `agent_metrics.py` (xem
  `src/README.md`). Toàn bộ trace, nhãn và summary ở `results/`.
- Đã chạy: retrieval dev (6 cấu hình) + test (3); agent dev hybrid (17 câu),
  test hybrid và test baseline (7 câu mỗi loại); tất cả 0 provider error ở lần
  chạy cuối. Kết quả agent dev trước cải thiện lưu ở
  `results/agent/dev/hybrid_v1_beforeimprove/`.
