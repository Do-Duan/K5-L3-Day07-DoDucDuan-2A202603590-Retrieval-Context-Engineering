"""Persistent cosine vector search with JSONB metadata filtering."""
from typing import Any
import json
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """TODO: retain connection settings; validate model dimensions."""
        self.database_url, self.dimensions = database_url, dimensions
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be known before creating the store")

    def _connect(self):
        conn = psycopg.connect(self.database_url, row_factory=dict_row)
        register_vector(conn)
        return conn

    def create_schema(self) -> None:
        """TODO: create table with vector column and source/page/filter metadata."""
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS ingestion_chunks (
                strategy text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
                source text NOT NULL, position integer NOT NULL, content text NOT NULL,
                metadata jsonb NOT NULL, embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY (strategy, chunk_id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS ingestion_chunks_metadata_idx ON ingestion_chunks USING gin (metadata)")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """TODO: validate lengths/dimensions and write stable IDs without duplicates.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("Embedding dimension does not match vector table")
        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page, c.text,
                 json.dumps(c.metadata), v) for c, v in zip(chunks, vectors)]
        with self._connect() as conn:
            conn.execute("DELETE FROM ingestion_chunks WHERE strategy=%s", (strategy,))
            with conn.cursor() as cur:
                cur.executemany("""INSERT INTO ingestion_chunks
                    (strategy,chunk_id,document_id,source,position,content,metadata,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT (strategy,chunk_id) DO UPDATE SET content=EXCLUDED.content,
                    metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding""", rows)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """TODO: filter permitted metadata in SQL, rank by cosine distance, return top-k.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        filters = filters or {}
        allowed = {"document_type", "section", "chapter", "language", "document_id"}
        if set(filters) - allowed:
            raise ValueError(f"Unsupported metadata filter keys: {set(filters)-allowed}")
        clauses, params = ["strategy=%s"], [strategy]
        for key, value in filters.items():
            clauses.append("metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        params.extend([query_vector, top_k])
        sql = f"""SELECT chunk_id,document_id,source,position,content,metadata,
            1-(embedding <=> %s::vector) AS cosine_similarity FROM ingestion_chunks
            WHERE {' AND '.join(clauses)} ORDER BY embedding <=> %s::vector LIMIT %s"""
        # Placeholder order follows SELECT expression before WHERE clauses.
        params = [query_vector, strategy] + [json.dumps({k:v}) for k,v in filters.items()] + [query_vector, top_k]
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()
