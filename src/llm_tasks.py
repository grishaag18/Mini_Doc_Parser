from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RiskItem(BaseModel):
    category: str = Field(..., description="Market / Regulatory / Operational / Financial / Competitive / Other")
    risk: str = Field(..., description="Concise risk description")
    evidence_quote: str = Field(..., description="Short phrase from document supporting the risk (<= 20 words)")


class CompanyInsights(BaseModel):
    company: str
    business_model: str
    revenue_drivers: List[str]
    management_tone: str = Field(..., description="Positive / Neutral / Cautious")
    outlook: str
    key_risks: List[RiskItem]


SYSTEM_FINANCE = """You are an investment research assistant.
Be conservative: if unclear, say "Not enough information".
No hallucinations. Use only provided text context.
Return concise, analyst-style outputs.
"""


def prompt_company_snapshot(company_name: str, context: str) -> str:
    return f"""
Company: {company_name}

TASK:
1) Summarize the business model in ~120 words for an investor.
2) List 3-6 revenue drivers (bullet list).

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_risks(company_name: str, context: str) -> str:
    schema = CompanyInsights.model_json_schema()
    return f"""
Company: {company_name}

TASK:
Extract top 5 investment risks from the text and categorize them.
For each risk, include a short evidence quote (<= 20 words).
Return JSON that follows this schema (JSON only):

SCHEMA:
{schema}

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_tone_outlook(company_name: str, context: str) -> str:
    return f"""
Company: {company_name}

TASK:
Determine management tone (Positive/Neutral/Cautious) and explain why in 3 bullets.
Then write a 2-3 sentence outlook summary for an investor.

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_qa(company_name: str, question: str, context: str) -> str:
    return f"""
Company: {company_name}

QUESTION:
{question}

INSTRUCTIONS:
Answer strictly using the context. If missing, say "Not enough information in provided text."

CONTEXT (10-K excerpt):
{context}
""".strip()
