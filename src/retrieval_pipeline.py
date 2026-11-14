"""PDF ingestion and retrieval for the Day 7 lab.

Builds a local Chroma dense index and exposes dense, TF-IDF lexical, and RRF
hybrid retrieval. Run from the repository root with `python src/retrieval_pipeline.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pymupdf
    import chromadb
    from dotenv import load_dotenv
    from src.embeddings import OpenRouterEmbedder
except ImportError as exc:  # report the missing runtime dependency clearly
    raise SystemExit(
        f"Thiếu dependency: {exc.name}. Cài bằng: python -m pip install -r requirements.txt"
    ) from exc

DATA_DIR = ROOT / "src" / "data"
INDEX_DIR = ROOT / "src" / "index" / "chroma"
CHUNKS_PATH = ROOT / "src" / "data" / "chunks.json"
COLLECTION_NAME = "day7_pdf_chunks"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"


def normalize_text(text: str) -> str:
    """Normalize PDF line breaks/spacing while preserving Vietnamese text."""
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def split_into_chunks(text: str, size: int = 1100, overlap: int = 160) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size phải > overlap >= 0")
    text = normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph or sentence boundary near the target size.
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if boundary > start + size * 0.60:
                end = boundary + (1 if text[boundary] == "\n" else 2)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_pdf_chunks(pdf_dir: Path, size: int = 1100, overlap: int = 160) -> list[dict[str, Any]]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Không tìm thấy PDF trong {pdf_dir}")
    output: list[dict[str, Any]] = []
    for pdf in pdfs:
        source_id = hashlib.sha1(str(pdf.resolve()).encode("utf-8")).hexdigest()[:12]
        with pymupdf.open(str(pdf)) as document:
            for page_num, page in enumerate(document, start=1):
                # PyMuPDF preserves Vietnamese words more faithfully than the
                # earlier pypdf extractor for this thesis's embedded fonts.
                page_text = page.get_text("text", sort=True) or ""
                for part_num, part in enumerate(split_into_chunks(page_text, size, overlap), start=1):
                    chunk_id = f"{source_id}:p{page_num}:c{part_num}"
                    output.append({
                        "id": chunk_id,
                        "text": part,
                        "source_id": source_id,
                        "source": pdf.name,
                        "page": page_num,
                        "metadata": {"chunk_index": len(output), "page_chunk": part_num},
                    })
    if not output:
        raise ValueError("PDF không trích xuất được text; có thể tài liệu là scan cần OCR.")
    return output


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def open_collection(db_path: Path = INDEX_DIR, collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    return client, collection


def ingest(pdf_dir: Path = DATA_DIR, chunks_path: Path = CHUNKS_PATH, db_path: Path = INDEX_DIR) -> dict[str, Any]:
    chunks = load_pdf_chunks(pdf_dir)
    save_chunks(chunks, chunks_path)
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    embedder = OpenRouterEmbedder(EMBEDDING_MODEL)
    # Keep requests small enough for local embedding/model memory.
    batch_size = 32
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        vectors = embedder.embed_documents([c["text"] for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[{"source_id": c["source_id"], "source": c["source"], "page": c["page"], **c["metadata"]} for c in batch],
        )
    return {"pdf_count": len(list(pdf_dir.glob("*.pdf"))), "chunks": len(chunks),
            "pages_with_text": len({(c["source_id"], c["page"]) for c in chunks}),
            "embedding_model": embedder.model_name, "embedding_dimensions": embedder.dimensions,
            "index_path": str(db_path), "chunks_path": str(chunks_path)}


def _format_hit(item: dict[str, Any], score: float, method: str) -> dict[str, Any]:
    return {"id": item["id"], "text": item["text"], "source_id": item["source_id"],
            "source": item["source"], "page": item["page"], "metadata": item["metadata"],
            "score": score, "method": method}


def dense_search(query: str, top_k: int = 5, db_path: Path = INDEX_DIR) -> list[dict[str, Any]]:
    _, collection = open_collection(db_path)
    if collection.count() == 0:
        raise RuntimeError("Chroma index rỗng. Chạy lệnh ingest trước.")
    query_vector = OpenRouterEmbedder(EMBEDDING_MODEL).embed_query(query)
    raw = collection.query(query_embeddings=[query_vector], n_results=min(top_k, collection.count()), include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, distance in zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        metadata = dict(meta)
        hits.append(_format_hit({"id": cid, "text": doc, "source_id": metadata.pop("source_id"),
                                 "source": metadata.pop("source"), "page": metadata.pop("page"), "metadata": metadata},
                                1.0 - float(distance), "dense"))
    return hits


def lexical_search(query: str, top_k: int = 5, chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    # Reuse the lab's offline TF-IDF implementation.
    sys.path.insert(0, str(ROOT))
    from starter.baseline import TfidfIndex
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [_format_hit(hit, float(hit["score"]), "tfidf") for hit in TfidfIndex(chunks).search(query, top_k)]


def hybrid_search(query: str, top_k: int = 5, candidate_k: int = 20,
                  db_path: Path = INDEX_DIR, chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    dense = dense_search(query, candidate_k, db_path)
    lexical = lexical_search(query, candidate_k, chunks_path)
    return rrf_fuse(dense, lexical, top_k)


def rrf_fuse(dense: list[dict[str, Any]], lexical: list[dict[str, Any]], top_k: int = 5,
             rrf_constant: int = 60) -> list[dict[str, Any]]:
    # Reciprocal Rank Fusion avoids adding incomparable dense and TF-IDF scores.
    fused: dict[str, dict[str, Any]] = {}
    for ranking in (dense, lexical):
        for rank, hit in enumerate(ranking, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (rrf_constant + rank)
    ordered = sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["id"]))[:top_k]
    return [{**x, "score": x["rrf_score"], "method": "hybrid_rrf"} for x in ordered]


def benchmark(questions_path: Path, split: str, output_path: Path,
              candidate_k: int = 20, db_path: Path = INDEX_DIR,
              chunks_path: Path = CHUNKS_PATH) -> dict[str, Any]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    cases = [q for q in questions if q.get("split") == split]
    if not cases:
        raise ValueError(f"Không có câu hỏi split={split} trong {questions_path}")
    answerable = [q for q in cases if q.get("answerable") is True]
    if len({q.get("id") for q in cases}) != len(cases):
        raise ValueError("ID câu hỏi bị trùng trong benchmark")
    if any(not isinstance(q.get("gold"), list) or not q["gold"] for q in answerable):
        raise ValueError("Mỗi câu answerable cần ít nhất một evidence trong gold")

    traces = []
    counts = {name: {"hit_at_1": 0, "evidence_hit_at_3": 0} for name in ("dense", "tfidf", "hybrid_rrf")}
    errors = 0
    for question in answerable:
        trace: dict[str, Any] = {"id": question["id"], "split": split,
                                 "question": question["question"], "status": "ok", "methods": {}}
        try:
            dense = dense_search(question["question"], candidate_k, db_path)
            lexical = lexical_search(question["question"], candidate_k, chunks_path)
            rankings = {"dense": dense[:3], "tfidf": lexical[:3],
                        "hybrid_rrf": rrf_fuse(dense, lexical, 3)}
            for method, hits in rankings.items():
                def is_hit(hit: dict[str, Any]) -> bool:
                    return any(hit["source_id"] == gold["source_id"]
                               and int(hit["page"]) == int(gold["page"])
                               and gold["evidence"] in hit["text"]
                               for gold in question["gold"])
                flags = [is_hit(hit) for hit in hits]
                counts[method]["hit_at_1"] += bool(flags and flags[0])
                counts[method]["evidence_hit_at_3"] += any(flags)
                trace["methods"][method] = {
                    "hit_at_1": bool(flags and flags[0]),
                    "evidence_hit_at_3": any(flags),
                    "results": [{"id": hit["id"], "source": hit["source"],
                                 "page": hit["page"], "score": hit["score"], "hit": flag}
                                for hit, flag in zip(hits, flags)],
                }
        except Exception as exc:
            errors += 1
            trace["status"] = "error"
            trace["error"] = f"{type(exc).__name__}: {exc}"
        traces.append(trace)

    n = len(answerable)
    summary = {"split": split, "answerable_n": n, "unanswerable_excluded": len(cases) - n,
               "errors": errors, "embedding_model": EMBEDDING_MODEL,
               "methods": {method: {metric: {"n": value, "N": n,
                                                        "rate": (value / n if n else None)}
                                     for metric, value in metrics.items()}
                           for method, metrics in counts.items()}}
    report = {"summary": summary, "traces": traces}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="PDF → chunks → ChromaDB retrieval (dense, TF-IDF, hybrid)")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Trích xuất PDF, chia chunk và lập Chroma index")
    ing.add_argument("--pdf-dir", type=Path, default=DATA_DIR)
    ing.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ing.add_argument("--db", type=Path, default=INDEX_DIR)
    sea = sub.add_parser("search", help="Tìm và in kết quả có nguồn/trang")
    sea.add_argument("query")
    sea.add_argument("--method", choices=("dense", "tfidf", "hybrid"), default="dense")
    sea.add_argument("--top-k", type=int, default=5)
    sea.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    sea.add_argument("--db", type=Path, default=INDEX_DIR)
    bench = sub.add_parser("benchmark", help="So sánh retrieval trên một split; gold chỉ dùng để chấm")
    bench.add_argument("--questions", type=Path, default=DATA_DIR / "retrieval_benchmark.json")
    bench.add_argument("--split", choices=("dev", "test"), required=True)
    bench.add_argument("--output", type=Path)
    bench.add_argument("--candidate-k", type=int, default=20)
    bench.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    bench.add_argument("--db", type=Path, default=INDEX_DIR)
    args = parser.parse_args()
    if args.command == "ingest":
        print(json.dumps(ingest(args.pdf_dir, args.chunks, args.db), ensure_ascii=False, indent=2))
        return
    if args.command == "benchmark":
        if args.candidate_k < 3:
            parser.error("--candidate-k phải >= 3 để bảo đảm có top-3 so sánh")
        output = args.output or ROOT / "src" / "results" / f"retrieval-{args.split}.json"
        print(json.dumps(benchmark(args.questions, args.split, output, args.candidate_k,
                                   args.db, args.chunks), ensure_ascii=False, indent=2))
        print(f"Trace saved: {output}")
        return
    if args.top_k < 1:
        parser.error("--top-k phải >= 1")
    if args.method == "dense":
        hits = dense_search(args.query, args.top_k, args.db)
    elif args.method == "tfidf":
        hits = lexical_search(args.query, args.top_k, args.chunks)
    else:
        hits = hybrid_search(args.query, args.top_k, db_path=args.db, chunks_path=args.chunks)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
