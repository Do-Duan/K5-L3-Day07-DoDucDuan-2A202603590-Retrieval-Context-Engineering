"""Ingest the DOCX with real OpenRouter embeddings into pgvector."""
import os
import json
import hashlib
import time
from pathlib import Path
from dotenv import load_dotenv
from .ingestion import load_docx
from .chunking import by_paragraph
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    root = Path(__file__).resolve().parents[2]
    doc_path = Path(os.getenv("DOCX_PATH", str(root / "starter" / "data" / "Capstone Project Document.docx")))
    model = os.getenv("EMBEDDING_MODEL")
    db_url = os.getenv("DATABASE_URL")
    if not model or not db_url:
        raise RuntimeError("Set EMBEDDING_MODEL and DATABASE_URL in starter/.env")
    source = str(doc_path.resolve().relative_to(root.resolve()))
    units = load_docx(doc_path, "capstone-project-document", source)
    questions_path = root / "data" / "questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks_by_strategy = {
        "fixed-v1": [c for unit in units for c in __import__("src.chunking", fromlist=["fixed_size"]).fixed_size(unit, 500, 50)],
        "paragraph-v1": [c for unit in units for c in by_paragraph(unit, 900)],
        "paragraph-filter-v1": [c for unit in units for c in by_paragraph(unit, 900)],
    }
    embedder = MyEmbedder(model)
    all_chunks = list({c.chunk_id: c for group in chunks_by_strategy.values() for c in group}.values())
    vectors = embedder.embed_documents([c.text for c in all_chunks])
    vector_by_id = {c.chunk_id: v for c, v in zip(all_chunks, vectors)}
    store = PgVectorStore(db_url, embedder.dimensions)
    store.create_schema()
    for strategy, chunks in chunks_by_strategy.items():
        store.upsert(strategy, chunks, [vector_by_id[c.chunk_id] for c in chunks])
    traces = []
    for strategy in chunks_by_strategy:
        for q in questions:
            query_vector = embedder.embed_query(q["question"])
            hits = store.search(strategy, query_vector, 3, q.get("filter") if strategy.endswith("filter-v1") else None)
            traces.append({"strategy": strategy, **evaluate_query(q, hits)})
    results = root / "starter" / "results"
    results.mkdir(exist_ok=True)
    (results / "benchmark_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (results / "benchmark_summary.json").write_text(json.dumps(summarize(traces), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ingested {len(units)} source units across {len(chunks_by_strategy)} strategies")
    print(f"Embedding model: {model}; dimensions: {embedder.dimensions}")
    print(f"Saved traces and summary under {results}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--filter", action="append", default=[], help="metadata key=value")
    args = parser.parse_args()
    if args.query:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        model, db_url = os.getenv("EMBEDDING_MODEL"), os.getenv("DATABASE_URL")
        if not model or not db_url:
            raise RuntimeError("Set EMBEDDING_MODEL and DATABASE_URL in starter/.env")
        embedder = MyEmbedder(model)
        vector = embedder.embed_query(args.query)
        filters = dict(item.split("=", 1) for item in args.filter)
        hits = PgVectorStore(db_url, embedder.dimensions).search("paragraph-v1", vector, 3, filters)
        for hit in hits:
            print(hit)
    else:
        main()
