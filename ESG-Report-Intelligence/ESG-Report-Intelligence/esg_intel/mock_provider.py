"""
Offline test double for the LLM providers.

It answers every pipeline stage with simple rules (keyword + number
matching), so the whole application can be tested without an API key or
network access. It is NOT used for real analyses: the UI only offers it
when the environment variable ESG_ENABLE_MOCK=1 is set.

Tests can also script failures, e.g. MockProvider(script=[RateLimitError("429"), ...])
to check that retries and error handling work.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter

from .config import METRICS_BY_ID
from .llm_client import BaseProvider, LLMRequest, LLMResponse
from .schemas import parse_number

_NUM = r"(\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, model: str = "mock-extractor", script: list | None = None):
        super().__init__(model)
        self.script = list(script or [])
        self.calls: list[LLMRequest] = []

    # -------------------------------------------------------------
    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        started = time.perf_counter()
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                return self._response(item, request, started)

        system, prompt = request.system.lower(), request.prompt
        if "document metadata" in system:
            text = self._profile(prompt)
        elif "assurance auditor" in system:
            text = self._verify(prompt)
        elif "sustainability analyst" in system:
            text = self._summary(prompt)
        elif "could not be parsed" in prompt:
            text = json.dumps({"metrics": []})
        else:
            text = self._extract(prompt)
        return self._response(text, request, started)

    def _response(self, text: str, request: LLMRequest, started: float) -> LLMResponse:
        return LLMResponse(text=text, model=self.model,
                           input_tokens=len(request.prompt) // 4, output_tokens=len(text) // 4,
                           latency_s=time.perf_counter() - started, finish_reason="STOP")

    # -------------------------------------------------------------
    @staticmethod
    def _profile(prompt: str) -> str:
        prompt = prompt.split("TASK:")[0]   # only the report pages, not the instructions
        years = Counter(re.findall(r"\b(20[12]\d)\b", prompt))
        year = years.most_common(1)[0][0] if years else None
        company = "Unknown"
        match = re.search(r"([A-Z][\w&.\- ]{2,40}?\s(?:AG|SE|SA|S\.A\.|plc|PLC|Inc\.?|Group|Oyj|AB|SpA|N\.V\.))\b",
                          prompt)
        if match:
            company = match.group(1).strip()
        frameworks = [f for f in ("GRI", "ESRS", "TCFD", "SASB", "CDP") if f in prompt]
        return json.dumps({"company_name": company, "reporting_year": int(year) if year else None,
                           "report_type": "Sustainability report", "sector": "Unknown",
                           "country": "Unknown", "frameworks": frameworks,
                           "assurance": "None stated", "language": "English",
                           "one_line_description": "Profile produced by the offline test provider."})

    # -------------------------------------------------------------
    @staticmethod
    def _pages(prompt: str) -> list[tuple[int, str]]:
        parts = re.split(r"===== PAGE (\d+)[^\n]*=====", prompt)
        pages = []
        for i in range(1, len(parts) - 1, 2):
            pages.append((int(parts[i]), parts[i + 1].split("=" * 60)[0]))
        return pages

    def _extract(self, prompt: str) -> str:
        ids = [int(x) for x in re.findall(r"^(\d+) \| ", prompt.split("METRICS TO EXTRACT")[-1], re.M)]
        year_match = re.search(r"Reporting year: (\d{4})", prompt)
        year = int(year_match.group(1)) if year_match else None
        pages = self._pages(prompt.split("TASK:")[0])   # report excerpts only, not the few-shot examples
        results = []
        for metric_id in ids:
            metric = METRICS_BY_ID.get(metric_id)
            found = None
            if metric:
                key = metric.keywords[0]
                for number, text in pages:
                    for line in text.splitlines():
                        low = line.lower()
                        if key not in low or "...." in line or \
                                any(w in low for w in ("target", "by 2030", "goal", "gri ")):
                            continue
                        if line.strip().startswith("|"):
                            cells = [c.strip() for c in line.strip().strip("|").split("|")]
                            numbers = [c for c in cells[1:] if parse_number(c) is not None]
                            if numbers:
                                found = (number, parse_number(numbers[-1]), line.strip()[:200], "table")
                                break
                        else:
                            m = re.search(_NUM + r"\s*(%|t\s?CO2e|tCO2e|m³|m3|t\b)?", line[low.index(key):])
                            if m and parse_number(m.group(1)) not in (None, 1.0, 2.0, 3.0):
                                found = (number, parse_number(m.group(1)), line.strip()[:200], "text")
                                break
                    if found:
                        break
            if found:
                page, value, evidence, source = found
                results.append({"metric_id": metric_id, "status": "found", "value": value,
                                "unit": metric.standard_unit, "year": year, "page": page,
                                "source_type": source, "evidence": evidence,
                                "scope2_method": "market-based" if metric_id == 9 else "",
                                "reasoning": "Keyword and number matched on the page (test provider).",
                                "confidence": 0.7, "history": []})
            else:
                results.append({"metric_id": metric_id, "status": "not_found", "value": None, "unit": "",
                                "year": None, "page": None, "source_type": "none", "evidence": "",
                                "scope2_method": "", "reasoning": "No matching line.", "confidence": 0.0,
                                "history": []})
        return json.dumps({"metrics": results})

    # -------------------------------------------------------------
    @staticmethod
    def _verify(prompt: str) -> str:
        items = re.split(r"\n\[\d+\] ", prompt)
        out = []
        for block in items[1:]:
            id_match = re.search(r"metric_id (\d+)", block)
            value_match = re.search(r"extracted value: ([\d.,\-]+)", block)
            if not id_match:
                continue
            source = block.split("SOURCE", 1)[-1]
            value = value_match.group(1) if value_match else ""
            present = bool(value) and value.rstrip(".0") in source.replace(",", "").replace(" ", "")
            out.append({"metric_id": int(id_match.group(1)),
                        "verdict": "VALID" if present else "UNCERTAIN",
                        "checks": {"metric_matches": True, "year_matches": True, "value_matches": present,
                                   "unit_matches": True, "company_scope": True},
                        "reason": "Number found in the source text." if present
                        else "Number not visible in the source text.",
                        "corrected_value": None})
        return json.dumps({"verifications": out})

    @staticmethod
    def _summary(prompt: str) -> str:
        n_points = prompt.count("- [")
        return json.dumps({
            "executive_summary": f"Offline test summary: {n_points} data points were provided. "
                                 "Connect a real model (Gemini or Groq) for an AI-written briefing.",
            "environmental": "Test provider – no narrative generated.",
            "social": "Test provider – no narrative generated.",
            "governance": "Test provider – no narrative generated.",
            "strengths": [], "disclosure_gaps": [], "red_flags": [], "recommended_follow_up": []})
