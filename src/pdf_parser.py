from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class ParsedPDF:
    path: Path
    text: str


def _clean_text(text: str) -> str:
    # Normalize whitespace
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Drop very common header/footer patterns (best-effort)
    # You can add more patterns as you see recurring artifacts.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        # Remove page numbers like "12" or "Page 12"
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.fullmatch(r"Page\s+\d{1,4}", stripped, flags=re.IGNORECASE):
            continue

        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def parse_pdf(pdf_path: str | Path) -> ParsedPDF:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text_parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                full_text_parts.append(page_text)

    raw = "\n\n".join(full_text_parts)
    cleaned = _clean_text(raw)
    return ParsedPDF(path=pdf_path, text=cleaned)
