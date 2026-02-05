from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SectionExtraction:
    sections: Dict[str, str]
    found: bool


_ITEM_PATTERNS = {
    "item_1_business": r"\bitem\s+1[\.\:\s]+(business)\b",
    "item_1a_risks": r"\bitem\s+1a[\.\:\s]+(risk\s+factors)\b",
    "item_7_mdna": r"\bitem\s+7[\.\:\s]+(management['’]s\s+discussion|md&a|discussion\s+and\s+analysis)\b",
}


def _find_heading_positions(text: str) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for key, pat in _ITEM_PATTERNS.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            positions[key] = m.start()
    return positions


def extract_core_sections(text: str) -> SectionExtraction:
    """
    Best-effort extraction of 10-K sections. 10-K formatting varies, so this may miss.
    """
    pos = _find_heading_positions(text)
    if not pos:
        return SectionExtraction(sections={}, found=False)

    # Sort headings by position
    ordered = sorted(pos.items(), key=lambda kv: kv[1])

    # Slice each section until next heading
    sections: Dict[str, str] = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        chunk = text[start:end].strip()
        sections[key] = chunk

    return SectionExtraction(sections=sections, found=True)


def get_section_or_fallback(ex: SectionExtraction, text: str, key: str) -> str:
    if ex.found and key in ex.sections:
        return ex.sections[key]
    # fallback: return entire doc if section not found
    return text
