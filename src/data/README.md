# PDF retrieval input

Place permitted PDF files in this folder. The pipeline uses PyMuPDF to read
`*.pdf`, records source filename and one-based PDF page number, and writes generated chunks to
`chunks.json`. ChromaDB's persistent index is stored under `src/index/chroma/`.

The PDF and generated index/chunks are local data artifacts and are excluded
from Git. Do not upload confidential documents to external services.
