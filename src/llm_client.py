from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]


class LLMClient:
    """
    Lightweight local Ollama client.
    Requires: ollama serve + pulled model.
    """

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> LLMResponse:
        full_prompt = prompt if not system else f"{system}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": False,
        }

        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            text=(data.get("response") or "").strip(),
            raw=data,
        )

    def generate_json(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> Dict[str, Any]:
        resp = self.generate(prompt, system=system, temperature=temperature)
        txt = resp.text

        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(txt[start:end+1])
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "raw_text": txt}
