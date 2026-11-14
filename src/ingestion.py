"""Read DOCX paragraphs, preserving stable paragraph positions."""
from pathlib import Path
import re
import unicodedata
from docx import Document
from .models import DocumentPage


def load_docx(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Represent each non-empty paragraph as a traceable source unit."""
    if not path.is_file():
        raise FileNotFoundError(f"DOCX not found: {path}")
    doc = Document(path)
    units = []
    active_chapter, active_section = None, None
    content_index = 0
    for paragraph_index, paragraph in enumerate(doc.paragraphs, start=1):
        text = normalize_text(paragraph.text)
        if text:
            content_index += 1
            import re
            if re.match(r"^(chapter\s+\d+|\d+\s*\.)", text, re.I):
                active_chapter = int(re.search(r"\d+", text).group()) if re.search(r"\d+", text) else "appendices"
                active_section = text.split()[0].rstrip(".")
            elif appendix := re.match(r"^appendix\s+([a-z])\b", text, re.I):
                active_chapter, active_section = "appendices", f"Appendix {appendix.group(1).upper()}"
            elif re.match(r"^\d+(?:\.\d+)+\b", text):
                active_section = re.match(r"^\d+(?:\.\d+)+", text).group()
            elif re.fullmatch(r"(?:0\.0|[135]\.0)", text):
                # Standalone rubric score rows inherit Appendix metadata.
                pass
            units.append(DocumentPage(
                document_id=document_id, source=source, page=content_index, text=text,
                metadata={"position_type": "paragraph", "paragraph_index": content_index,
                          "docx_paragraph_index": paragraph_index,
                          "section": active_section or (paragraph.style.name if paragraph.style else "Normal"),
                          "chapter": active_chapter},
            ))
    # Include table cells, which python-docx exposes separately from paragraph flow.
    for table_index, table in enumerate(doc.tables, start=1):
        for row_index, row in enumerate(table.rows, start=1):
            text = normalize_text(" | ".join(cell.text for cell in row.cells))
            if text:
                position = len(doc.paragraphs) + (table_index * 100000) + row_index
                units.append(DocumentPage(document_id, source, position, text,
                    {"position_type": "table_row", "table_index": table_index,
                     "row_index": row_index, "section": "table"}))
    if not units:
        raise ValueError(f"No readable paragraph or table text found in {path}")
    return units


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining paragraph content."""
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()
