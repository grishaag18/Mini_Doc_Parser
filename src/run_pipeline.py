from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .pdf_parser import parse_pdf
from .sectioner import extract_core_sections, get_section_or_fallback
from .chunker import chunk_text
from .llm_client import LLMClient
from .llm_tasks import (
    SYSTEM_FINANCE,
    prompt_company_snapshot,
    prompt_risks,
)


@dataclass
class CompanyDoc:
    name: str
    pdf_path: Path


def _combine_first_n_chunks(chunks: List[str], n: int = 1) -> str:
    return "\n\n".join(chunks[:n]).strip()


def run_for_company(client: LLMClient, company: CompanyDoc, out_dir: Path) -> Dict:
    print(f"\n--- Processing {company.name} ---", flush=True)

    parsed = parse_pdf(company.pdf_path)
    extraction = extract_core_sections(parsed.text)

    business_text = get_section_or_fallback(
        extraction, parsed.text, "item_1_business"
    )
    risks_text = get_section_or_fallback(
        extraction, parsed.text, "item_1a_risks"
    )

    business_chunks = [
        c.text for c in chunk_text(business_text, max_chars=2000, overlap=200)
    ]
    risks_chunks = [
        c.text for c in chunk_text(risks_text, max_chars=2000, overlap=200)
    ]

    business_ctx = _combine_first_n_chunks(business_chunks) or parsed.text[:2000]
    risks_ctx = _combine_first_n_chunks(risks_chunks) or parsed.text[:2000]

    snapshot = client.generate(
        prompt_company_snapshot(company.name, business_ctx),
        system=SYSTEM_FINANCE,
        temperature=0.2,
    ).text

    risks_json = client.generate_json(
        prompt_risks(company.name, risks_ctx),
        system=SYSTEM_FINANCE,
        temperature=0.0,
    )

    result = {
        "company": company.name,
        "snapshot_text": snapshot,
        "risks_structured": risks_json,
    }

    out_path = out_dir / f"{company.name.lower()}_insights.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Saved → {out_path}", flush=True)
    return result


def build_comparison_table(results: List[Dict]) -> List[Dict]:
    table = []

    for r in results:
        risks = r.get("risks_structured", {})
        key_risks = risks.get("key_risks", [])
        risk_count = len(key_risks) if isinstance(key_risks, list) else None

        table.append(
            {
                "company": r["company"],
                "num_risks_identified": risk_count,
                "risk_categories": sorted(
                    {rk.get("category", "Unknown") for rk in key_risks}
                ) if isinstance(key_risks, list) else [],
                "snapshot_preview": r["snapshot_text"][:180] + "..."
            }
        )

    return table


def main():
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data" / "10k_reports"
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    companies = [
        CompanyDoc("Apple", data_dir / "apple_10k.pdf"),
        CompanyDoc("Microsoft", data_dir / "microsoft_10k.pdf"),
        CompanyDoc("Tesla", data_dir / "tesla_10k.pdf"),
    ]

    client = LLMClient(model="qwen2.5:3b")

    results = []
    for c in tqdm(companies, desc="Processing 10-Ks"):
        results.append(run_for_company(client, c, out_dir))

    comparison = build_comparison_table(results)
    comparison_path = out_dir / "comparison_table.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    print("\n=== DONE ===")
    print("Company outputs:")
    for r in results:
        print(f" - {r['company'].lower()}_insights.json")
    print("Comparison table:")
    print(f" - {comparison_path}")


if __name__ == "__main__":
    main()
