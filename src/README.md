# Retrieval pipeline

Pipeline đầu tiên của repo nằm ở `src/retrieval_pipeline.py` và hiện thực phần
PyMuPDF PDF extraction → chunk → OpenRouter embedding → ChromaDB → search. Đặt PDF được phép sử dụng trong
`src/data/`; Chroma index lưu cục bộ tại `src/index/chroma/`, còn chunk chuẩn
hóa lưu ở `src/data/chunks.json` để tiện audit và chạy lexical baseline.

Từ thư mục gốc repo, cài dependency, copy `.env.example` thành `.env`, điền
`OPENROUTER_API_KEY`, rồi tạo index:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python src/retrieval_pipeline.py ingest
```

Pipeline dùng `qwen/qwen3-embedding-8b` qua OpenRouter cho cả vector tài liệu
(`search_document`) và vector câu hỏi (`search_query`). Chroma lưu vector cục bộ.
Sau khi ingest, thử ba cách retrieval trên cùng dữ liệu:

```powershell
python src/retrieval_pipeline.py search "câu hỏi thử nghiệm" --method dense
python src/retrieval_pipeline.py search "câu hỏi thử nghiệm" --method tfidf
python src/retrieval_pipeline.py search "câu hỏi thử nghiệm" --method hybrid
```

Mỗi kết quả kèm chunk ID, tên file, số trang và score. Đây là bước kiểm tra
retrieval. Benchmark trong `src/data/retrieval_benchmark.json` có 15 câu trả lời
được + 2 câu thiếu evidence cho dev, và 5 câu trả lời được + 2 câu thiếu
evidence cho test. Chạy dev trước khi chọn cấu hình:

```powershell
python src/retrieval_pipeline.py benchmark --split dev
```

Gold chỉ được benchmark dùng để chấm nguồn/trang/evidence; search không nhận
gold. Chốt cấu hình trước khi chạy test:

```powershell
python src/retrieval_pipeline.py benchmark --split test
```

Kết quả ghi vào `src/results/retrieval-<split>.json`, gồm Hit@1, Evidence
Hit@3 và top-3 trace cho dense, TF-IDF và hybrid RRF. Không commit PDF, chunks
sinh ra hoặc Chroma index.

## Agentic RAG

`src/agent.py` dùng OpenRouter chat tool calling để điều phối tối đa 3 lượt tìm.
Model mặc định là `qwen/qwen3.8-flash`; có thể đổi bằng
`OPENROUTER_CHAT_MODEL` trong `.env` hoặc `--model`. Agent được phép chọn dense,
TF-IDF hoặc hybrid RRF cho từng lượt. Mặc định mỗi lượt lấy top-5, giữ tối đa
6000 ký tự context, và ghi trace JSONL gồm tool call, kết quả, context, câu trả
lời, citation, lỗi và thời gian.

Chạy một câu hỏi:

```powershell
python src/agent.py --question "Khóa luận đề xuất phương pháp biểu diễn văn bản dựa trên lý thuyết nào?"
```

Trace được thêm vào `src/results/agent-traces.jsonl`. Agent chỉ chấp nhận citation
có ID trong context; người chấm vẫn cần đọc nguồn để xác nhận citation thực sự
hỗ trợ nội dung câu trả lời. Lệnh trên gọi model chat và có thể phát sinh phí API.

Chạy Agent trên dev và lưu trace từng câu cùng summary trạng thái:

```powershell
python src/agent_benchmark.py --split dev
```

Runner chỉ truyền câu hỏi vào Agent; gold và nhãn answerable được giữ ở evaluator.
Summary là status metrics, chưa thay cho nhãn thủ công về độ đúng, độ đủ và citation
support. Không chạy test cho đến khi chốt cấu hình trên dev.
