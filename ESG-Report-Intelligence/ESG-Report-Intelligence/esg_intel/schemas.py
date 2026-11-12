"""
Pydantic schemas for every structured LLM output.

LLMs are not perfectly consistent: a number may come back as "12,345",
confidence as "high" and a page as "p. 85". The validators below coerce
these common variations instead of failing, and anything that cannot be
coerced raises a ValidationError that the pipeline turns into a
self-repair request.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

_CONFIDENCE_WORDS = {"very high": 0.95, "high": 0.9, "medium": 0.6, "moderate": 0.6,
                     "low": 0.3, "very low": 0.15}


def parse_number(value) -> Optional[float]:
    """Turn '12,345.6', '12 345', '1.234.567' or '45%' into a float. None if impossible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "null", "none", "n/a", "na", "not found", "not_found", "-", "–"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text or not re.search(r"\d", text):
        return None
    # "1.234.567" (European thousands) -> 1234567
    if text.count(".") > 1 and "," not in text:
        text = text.replace(".", "")
    # "1,234,567.8" -> remove thousands commas
    elif "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):          # 1.234,5 (European)
            text = text.replace(".", "").replace(",", ".")
        else:                                             # 1,234.5
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        # "12,345" -> thousands separator; "12,5" -> decimal comma
        if all(len(p) == 3 for p in parts[1:]):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def parse_year(value) -> Optional[int]:
    """'2024' -> 2024, 'FY2023/24' -> 2024 (fiscal years use the end year)."""
    if value is None:
        return None
    text = str(value)
    fiscal = re.search(r"((?:19|20)\d{2})\s*[/\-–]\s*(\d{2})\b(?!\d)", text)
    if fiscal:
        return int(fiscal.group(1)[:2] + fiscal.group(2))
    years = re.findall(r"(?:19|20)\d{2}", text)
    return int(years[-1]) if years else None


def parse_page(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


class YearValue(BaseModel):
    year: int
    value: float

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        year = parse_year(v)
        if year is None:
            raise ValueError(f"not a year: {v!r}")
        return year

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, v):
        number = parse_number(v)
        if number is None:
            raise ValueError(f"not a number: {v!r}")
        return number


# ------------------------------------------------------------------
# Stage 1: report profile
# ------------------------------------------------------------------
class ReportProfile(BaseModel):
    company_name: str = "Unknown"
    reporting_year: Optional[int] = None
    report_type: str = "Unknown"
    sector: str = "Unknown"
    country: str = "Unknown"
    frameworks: list[str] = Field(default_factory=list)
    assurance: str = "Not stated"
    language: str = "Unknown"
    one_line_description: str = ""

    @field_validator("reporting_year", mode="before")
    @classmethod
    def _year(cls, v):
        return parse_year(v)

    @field_validator("frameworks", mode="before")
    @classmethod
    def _frameworks(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [part.strip() for part in re.split(r"[,;/]", v) if part.strip()]
        return v


# ------------------------------------------------------------------
# Stage 2: metric extraction
# ------------------------------------------------------------------
SourceType = Literal["text", "table", "chart", "image", "none"]


class ExtractedMetric(BaseModel):
    metric_id: int
    status: Literal["found", "not_found"]
    value: Optional[float] = None
    unit: str = ""
    year: Optional[int] = None
    page: Optional[int] = None
    source_type: SourceType = "none"
    evidence: str = ""
    scope2_method: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    history: list[YearValue] = Field(default_factory=list)

    @field_validator("metric_id", mode="before")
    @classmethod
    def _metric_id(cls, v):
        match = re.search(r"\d+", str(v))
        if not match:
            raise ValueError(f"metric_id must contain a number, got {v!r}")
        return int(match.group())

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        text = str(v).strip().lower().replace(" ", "_")
        return "found" if text in {"found", "reported", "yes", "true"} else "not_found"

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, v):
        return parse_number(v)

    @field_validator("page", mode="before")
    @classmethod
    def _page(cls, v):
        return parse_page(v)

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        return parse_year(v)

    @field_validator("source_type", mode="before")
    @classmethod
    def _source_type(cls, v):
        text = str(v or "none").strip().lower()
        for option in ("table", "chart", "image", "text"):
            if option in text:
                return option
        if "graph" in text or "figure" in text:
            return "chart"
        return "none"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v):
        if isinstance(v, str) and v.strip().lower() in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[v.strip().lower()]
        number = parse_number(v)
        if number is None:
            return 0.0
        if number > 1:          # "85" or "85%" -> 0.85
            number /= 100
        return max(0.0, min(1.0, number))

    @field_validator("history", mode="before")
    @classmethod
    def _history(cls, v):
        if not v:
            return []
        clean = []
        for item in v if isinstance(v, list) else []:
            try:
                clean.append(YearValue.model_validate(item))
            except Exception:
                continue  # drop unusable history points, keep the metric
        return clean

    @field_validator("unit", "evidence", "scope2_method", "reasoning", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else str(v)


class ExtractionOutput(BaseModel):
    metrics: list[ExtractedMetric]


# ------------------------------------------------------------------
# Stage 3: AI verification
# ------------------------------------------------------------------
class VerificationChecks(BaseModel):
    metric_matches: bool = False
    year_matches: bool = False
    value_matches: bool = False
    unit_matches: bool = False
    company_scope: bool = False   # refers to the company, not a target / supplier / site


class VerificationItem(BaseModel):
    metric_id: int
    verdict: Literal["VALID", "UNCERTAIN", "INVALID"]
    checks: VerificationChecks = Field(default_factory=VerificationChecks)
    reason: str = ""
    corrected_value: Optional[float] = None

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        text = str(v).strip().upper()
        if text.startswith("VALID") or text == "CORRECT":
            return "VALID"
        if text.startswith("INVALID") or text in {"WRONG", "INCORRECT"}:
            return "INVALID"
        return "UNCERTAIN"

    @field_validator("metric_id", mode="before")
    @classmethod
    def _metric_id(cls, v):
        match = re.search(r"\d+", str(v))
        if not match:
            raise ValueError("metric_id missing")
        return int(match.group())

    @field_validator("corrected_value", mode="before")
    @classmethod
    def _corrected(cls, v):
        return parse_number(v)


class VerificationOutput(BaseModel):
    verifications: list[VerificationItem]


# ------------------------------------------------------------------
# Stage 4: insight summary
# ------------------------------------------------------------------
class SummaryOutput(BaseModel):
    executive_summary: str
    environmental: str = ""
    social: str = ""
    governance: str = ""
    strengths: list[str] = Field(default_factory=list)
    disclosure_gaps: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    recommended_follow_up: list[str] = Field(default_factory=list)

    @field_validator("strengths", "disclosure_gaps", "red_flags", "recommended_follow_up", mode="before")
    @classmethod
    def _list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(item) for item in v]
