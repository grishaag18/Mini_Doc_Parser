from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    idx: int


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 600) -> List[Chunk]:
    """
    Simple char-based chunking that works across models.
    max_chars ~ safe for many local LLM contexts; adjust if needed.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, idx=idx))
            idx += 1
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks
