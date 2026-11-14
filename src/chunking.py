"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage
import hashlib


def _make(page: DocumentPage, text: str, index: int, start: int, end: int) -> Chunk:
    identity = f"{page.document_id}:{page.page}:{index}:{text}"
    return Chunk(hashlib.sha256(identity.encode("utf-8")).hexdigest(), page.document_id,
                 page.source, page.page, text,
                 {**page.metadata, "position_start": start, "position_end": end,
                  "document_type": "capstone_project"})


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a source unit into overlapping character windows."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    out, start = [], 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        out.append(_make(page, page.text[start:end], len(out), start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return out


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Keep source paragraphs intact where possible; split only oversized units."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    if len(page.text) <= max_size:
        return [_make(page, page.text, 0, 0, len(page.text))]
    return fixed_size(page, max_size, min(80, max_size // 5))
