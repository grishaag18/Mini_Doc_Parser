
# Financial 10-K Analysis using Local LLMs

A lightweight, local-first pipeline for parsing SEC 10-K filings and extracting structured business and risk insights using Large Language Models (LLMs).

This project is designed for **investment research and asset management workflows** and runs entirely on **consumer hardware**, without relying on paid APIs or external services.

---

## Overview

The pipeline processes public company 10-K filings (Apple, Microsoft, Tesla) to:

- Extract key sections (Business Overview and Risk Factors)
- Summarise company business models
- Identify and structure investment-relevant risks
- Generate a cross-company comparison table

The output is designed to support faster qualitative analysis and downstream quantitative workflows.

---

## Key design choices

- **Local LLM inference** via Ollama (no API keys, no billing)
- Hardware-aware constraints (small context windows, minimal calls)
- Structured JSON outputs for auditability and comparison
- Modular, production-style pipeline design

---

## Architecture (high level)

10-K PDF
→ Text extraction
→ Section detection (Item 1 / 1A)
→ Chunking
→ Local LLM inference
→ Structured JSON outputs
→ Cross-company comparison


---

## Tech stack

- Python
- Ollama (local LLM runtime)
- Lightweight LLMs (`qwen2.5:3b`)
- PDF parsing and text processing utilities

---


