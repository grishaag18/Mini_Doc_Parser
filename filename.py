cat
cat > 


ls -la src


cat > src/llm_tasks.py <<'PY'
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


# System message: keeps the model conservative (important in finance)
SYSTEM_FINANCE = """You are an investment research assistant.
Rules:
- Use ONLY the provided context. Do not invent facts.
- If the context is insufficient, say: "Not enough information in provided text."
- Be concise and analyst-style.
- Prefer bullet points where helpful.
"""


# --- Structured schema for risk extraction (so output is comparable across companies) ---

class RiskItem(BaseModel):
    category: str = Field(
        ...,
        description="One of: Market, Regulatory, Operational, Financial, Competitive, Legal, Other"
    )
    risk: str = Field(..., description="Concise risk statement")
    evidence_quote: str = Field(..., description="Short supporting phrase from context (<= 20 words)")


class CompanyRiskOutput(BaseModel):
    company: str
    key_risks: List[RiskItem]


# --- Prompts ---

def prompt_company_snapshot(company_name: str, context: str) -> str:
    return f"""
Company: {company_name}

TASK:
1) Summarize the business model in ~120 words (investor style).
2) List 3–6 revenue drivers as bullet points.

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_risks(company_name: str, context: str) -> str:
    schema = CompanyRiskOutput.model_json_schema()
    return f"""
Company: {company_name}

TASK:
Extract the top 5 investment risks from the context.
- Categorize each risk (Market, Regulatory, Operational, Financial, Competitive, Legal, Other).
- Provide ONE short evidence quote (<= 20 words) from the context for each risk.
Return VALID JSON ONLY matching the schema below. No markdown, no commentary.

SCHEMA:
{schema}

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_tone_outlook(company_name: str, context: str) -> str:
    return f"""
Company: {company_name}

TASK:
1) Classify management tone as one of: Positive, Neutral, Cautious.
2) Justify tone with 3 bullet points using phrases from the context.
3) Write a 2–3 sentence investor outlook summary based ONLY on the context.

CONTEXT (10-K excerpt):
{context}
""".strip()


def prompt_qa(company_name: str, question: str, context: str) -> str:
    return f"""
Company: {company_name}

QUESTION:
{question}

INSTRUCTIONS:
Answer using ONLY the context. If missing, respond exactly:
"Not enough information in provided text."

CONTEXT (10-K excerpt):
{context}
""".strip()
