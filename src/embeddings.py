"""OpenRouter embeddings adapter used by both ingestion and query retrieval."""
from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenRouterEmbedder:
    """OpenAI-compatible OpenRouter adapter with Qwen retrieval input types."""

    def __init__(self, model_name: str = "qwen/qwen3-embedding-8b", api_key: str | None = None) -> None:
        self.model_name = model_name
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Thiếu OPENROUTER_API_KEY; đặt key trong .env hoặc environment.")
        self.client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        self.dimensions = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            extra_body={"input_type": "search_document"},
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        self._validate(vectors, len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
            extra_body={"input_type": "search_query"},
        )
        vectors = [item.embedding for item in response.data]
        self._validate(vectors, 1)
        return vectors[0]

    def _validate(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected or not vectors or not vectors[0]:
            raise RuntimeError("OpenRouter trả về embedding rỗng hoặc sai số lượng.")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("Số chiều embedding không nhất quán trong cùng batch.")
        if self.dimensions not in (0, dimensions):
            raise RuntimeError("Số chiều embedding thay đổi giữa các request.")
        self.dimensions = dimensions


class MyEmbedder(OpenRouterEmbedder):
    """Compatibility name retained for the existing Day 7 starter code."""
