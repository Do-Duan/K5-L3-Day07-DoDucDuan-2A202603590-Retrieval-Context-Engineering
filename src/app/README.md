# Chatbot UI Day 7

Giao diện web cục bộ cho Agentic RAG hiện có. Backend dùng Python standard
library (`http.server`) và gọi `src.agent.Agent`; không cần cài framework web.

## Chạy

Từ thư mục gốc repo Day 7:

```powershell
python src/app/backend.py
```

Mở `http://127.0.0.1:8000`. Backend đọc `OPENROUTER_API_KEY` từ `.env` ở thư
mục gốc repo. Model chat lấy từ `OPENROUTER_CHAT_MODEL` hoặc dùng mặc định
`qwen/qwen3.8-flash`. Mỗi câu hỏi gọi model chat và retrieval embedding, có thể
phát sinh phí API.

## Cách dùng

- Nhập câu hỏi và nhấn Enter hoặc nút **Gửi**.
- Câu trả lời hiển thị cùng trạng thái trả lời/abstain.
- Mở thẻ nguồn để xem PDF tại đúng số trang.
- **Làm mới** chỉ xóa nội dung hiển thị trong trình duyệt; trace backend vẫn
  được Agent lưu theo cơ chế hiện tại.

Server mặc định chỉ lắng nghe `127.0.0.1`, không mở ra mạng ngoài.
