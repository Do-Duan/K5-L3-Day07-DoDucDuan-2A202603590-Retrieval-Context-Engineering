"""Minimal local web backend for the Day 7 Agentic RAG chatbot."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT / "src" / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.agent import Agent

load_dotenv(ROOT / ".env")
MAX_REQUEST_BYTES = 16_000


class ChatHandler(SimpleHTTPRequestHandler):
    server_version = "Day7LocalChat/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path.startswith("/api/pdf/"):
            filename = Path(unquote(path.removeprefix("/api/pdf/"))).name
            if not filename.lower().endswith(".pdf"):
                self._send_json(404, {"error": "Không tìm thấy PDF."})
                return
            pdf_path = (PDF_DIR / filename).resolve()
            if pdf_path.parent != PDF_DIR.resolve() or not pdf_path.is_file():
                self._send_json(404, {"error": "Không tìm thấy PDF."})
                return
            content = pdf_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "Endpoint không tồn tại."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Request không hợp lệ."})
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Câu hỏi quá dài hoặc request rỗng."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Nội dung phải là JSON UTF-8 hợp lệ."})
            return
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            self._send_json(400, {"error": "Nhập câu hỏi từ 1 đến 4000 ký tự."})
            return

        try:
            trace = Agent(max_searches=3, top_k=5, context_budget_chars=6000,
                          default_method="hybrid").run(question.strip())
        except Exception as exc:
            print(f"Chat request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            self._send_json(502, {"error": "Agent chưa xử lý được câu hỏi. Xem lỗi ở cửa sổ backend."})
            return

        context = {chunk["id"]: chunk for chunk in trace.get("context", [])}
        sources = []
        for chunk_id in trace.get("citations", []):
            chunk = context.get(chunk_id)
            if chunk:
                sources.append({"id": chunk_id, "source": chunk["source"],
                                "page": chunk["page"], "text": chunk["text"]})
        searches = [event for event in trace.get("events", []) if event.get("event") == "search"]
        self._send_json(200, {
            "status": trace["status"],
            "answer": trace["answer"],
            "citations": trace.get("citations", []),
            "sources": sources,
            "details": {"model": trace.get("model"),
                        "search_count": trace.get("search_count", 0),
                        "methods": [event.get("method") for event in searches],
                        "elapsed_ms": trace.get("elapsed_ms")},
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Local chatbot UI for the Day 7 RAG agent")
    parser.add_argument("--host", default="127.0.0.1", help="Mặc định chỉ nghe trên máy local")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Chatbot đang chạy tại http://{args.host}:{args.port}")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng chatbot...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
