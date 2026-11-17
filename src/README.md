# Code — Day 7 Agentic RAG

Agent truy xuất bằng chứng trên kho **hai cuốn Alice** của Lewis Carroll (Project
Gutenberg #11 và #12), tái sử dụng chunk, embedding và vector store của **Day 6**.

- Embedding: `BAAI/bge-small-en-v1.5` (FastEmbed ONNX, cục bộ, 384 chiều) — đúng
  model Day 6.
- Vector store: Qdrant cục bộ (`.local/qdrant`), cosine, exact search.
- Agent controller: **model thật có tool-calling** qua Groq
  (`llama-3.3-70b-versatile`, API OpenAI-compatible).

## Cấu trúc

| File | Vai trò |
|---|---|
| `day7/config.py` | Đường dẫn, model, đọc `.env` |
| `day7/prepare_data.py` | Dựng `data/day7/{chunks,questions,agent_cases}.json` từ Day 6 |
| `day7/dense_index.py` | Nhúng chunk + dựng/truy vấn Qdrant (dense) |
| `day7/retrievers.py` | Dense, BM25, Hybrid-RRF, filter-infer, rerank |
| `day7/eval_retrieval.py` | Hit@1, Evidence Hit@3, Context Evidence Rate, MRR, latency |
| `day7/groq_controller.py` | Controller gọi Groq với tool `search`/`provide_answer` |
| `day7/agent_runner.py` | Chạy vòng agent (`starter/agentic.run_agent`), lưu trace |
| `day7/grade.py` | Khung `reviews.json` để chấm thủ công; gộp bằng `starter/agent_metrics.py` |

## Ba phương pháp retrieval (≥3, gồm dense trên vector store Day 6)

1. **Dense** — query embedding bge-small, tìm vector gần nhất trên Qdrant.
2. **BM25** — Okapi BM25 thuần stdlib trên cùng chunk.
3. **Hybrid RRF** — hợp nhất dense + BM25 bằng Reciprocal Rank Fusion.

Cấu hình cải thiện: `dense_filter` (suy luận metadata filter từ câu hỏi),
`dense_rerank_bm25` (over-fetch rồi rerank lexical), và nâng context budget
800 → 3000.

## Cách chạy

Dùng Python 3.11+ có `fastembed`, `qdrant-client`, `numpy`, `httpx`
(xem `requirements.txt`). Trong lab này dùng venv của Day 6 vì đã có sẵn model
cache; có thể tạo venv riêng.

```bash
# 1. Dựng benchmark từ Day 6 (đặt DAY6_ROOT nếu Day 6 ở nơi khác)
python src/day7/prepare_data.py

# 2. Dựng index dense (nhúng 273 chunk, ~1-2 phút CPU)
python src/day7/dense_index.py

# 3. So sánh retrieval trên dev (6 cấu hình)
python src/day7/eval_retrieval.py --split dev

# 4. Cấu hình đã chốt + baseline trên test
python src/day7/eval_retrieval.py --split test --configs hybrid_wide_ctx bm25

# 5. Chạy agent (cần GROQ_API_KEY trong .env). dev để kiểm thử, test để báo cáo
python src/day7/agent_runner.py --split dev  --config hybrid
python src/day7/agent_runner.py --split test --config hybrid
python src/day7/agent_runner.py --split test --config baseline_bm25

# 6. Chấm thủ công -> reviews.json, rồi gộp
python starter/agent_metrics.py --cases results/agent/test/hybrid/cases.json \
    --traces results/agent/test/hybrid/traces.json \
    --reviews results/agent/test/hybrid/reviews.json \
    --split test --output results/agent/test/hybrid/summary.json
```

### Khóa bí mật

`GROQ_API_KEY` đặt trong `.env` ở gốc repo (đã `.gitignore`, không commit). Xem
`.env.example`. Nội dung tài liệu và kết quả tool là **dữ liệu**, không phải chỉ dẫn.

## Tái lập

`data/day7/*.json` và `results/**` được commit (corpus là public domain). Index
Qdrant và model cache không commit; dựng lại bằng bước 1–2. `DAY6_ROOT` trỏ tới
repo Day 6 để lấy chunk gốc và model cache.
