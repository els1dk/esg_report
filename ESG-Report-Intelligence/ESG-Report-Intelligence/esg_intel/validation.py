"""
Rule-based validation and result fusion (no LLM involved).

LLM output is never trusted blindly. Every extracted value is checked with
deterministic rules against the PDF itself:

  * page_in_range     – the cited page exists
  * value_on_page     – the number really appears on the cited page
                        (in any common print format: 12,345 / 12 345 / 12.345)
  * evidence_grounded – the quoted evidence really is on the page
  * unit_plausible    – the unit fits the metric (tCO2e for emissions, % for shares...)
  * range_ok          – percentages 0–100, ratios 0–3, no negative emissions, sane years
  * not_a_target      – the evidence does not contain target / forecast language

The rule results are then combined with the AI verifier's verdict into the
final status: VALID, UNCERTAIN, INVALID or NOT FOUND.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .config import METRICS_BY_ID, TARGET_WORDS, PILLAR_NAMES
from .pdf_processor import ParsedDocument, number_variants
from .schemas import ExtractedMetric, VerificationItem

CURRENT_YEAR = date.today().year


# ------------------------------------------------------------------
# Unit normalisation
# ------------------------------------------------------------------
def normalise_unit(value: Optional[float], unit: str, family: str) -> tuple[Optional[float], str, bool]:
    """
    Convert a value to the metric's standard unit.
    Returns (normalised_value, standard_unit, unit_recognised).
    """
    if value is None:
        return None, "", False
    raw = unit or ""
    u = raw.lower().replace("₂", "2").replace("³", "3")

    if "%" in u or "percent" in u:
        return value, "%", family in ("percent", "mass", "count")

    if family == "emissions":
        if not any(k in u for k in ("co2", "ghg", "t", "ton")):
            return value, raw, False
        if re.search(r"\bmt\b|mtco2|million\s*(metric\s*)?t", u):
            return value * 1e6, "t CO2e", True
        if re.search(r"\bkt\b|ktco2|thousand\s*(metric\s*)?t|kilot|'000\s*t|000\s*t", u):
            return value * 1e3, "t CO2e", True
        return value, "t CO2e", True

    if family == "volume":
        if raw.strip() == "ML" or "megalit" in u or "mega lit" in u:
            return value * 1e3, "m³", True
        if re.search(r"\bgl\b|gigalit", u):
            return value * 1e6, "m³", True
        if re.search(r"million\s*m|mio\.?\s*m|\bmm3\b|hm3", u):
            return value * 1e6, "m³", True
        if re.search(r"thousand\s*m|'000\s*m|000\s*m|\bkm3\b|tsd\.?\s*m|1,000\s*m", u):
            return value * 1e3, "m³", True
        if "m3" in u or "cubic" in u:
            return value, "m³", True
        if re.search(r"\blit(re|er)s?\b|\bl\b", u):
            return value / 1000, "m³", True
        return value, raw, False

    if family == "mass":
        if re.search(r"million\s*t", u):
            return value * 1e6, "t", True
        if re.search(r"\bkt\b|thousand\s*t|'000\s*t", u):
            return value * 1e3, "t", True
        if re.search(r"\bkg\b", u):
            return value / 1e3, "t", True
        if re.search(r"\bt\b|tonne|ton", u):
            return value, "t", True
        return value, raw, False

    if family == "energy":
        factors = (("gwh", 1e3), ("mwh", 1.0), ("kwh", 1e-3), ("pj", 277_777.8),
                   ("tj", 277.78), ("gj", 0.27778))
        for key, factor in factors:
            if key in u:
                return value * factor, "MWh", True
        return value, raw, False

    if family == "ratio":
        return value, "ratio", True
    if family == "percent":
        # A percentage metric reported as a fraction (0.36) is converted.
        if value <= 1 and ("ratio" in u or u.strip() in ("", "share", "fraction")):
            return value * 100, "%", True
        return value, "%", False
    return value, raw or "count", True


# ------------------------------------------------------------------
# Rule checks
# ------------------------------------------------------------------
@dataclass
class RuleChecks:
    page_in_range: Optional[bool] = None
    value_on_page: Optional[bool] = None       # None = not applicable (chart / scan)
    evidence_grounded: Optional[bool] = None
    unit_plausible: Optional[bool] = None
    range_ok: Optional[bool] = None
    not_a_target: Optional[bool] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "notes"}

    @property
    def hard_failure(self) -> bool:
        return self.page_in_range is False or self.range_ok is False

    @property
    def pass_ratio(self) -> float:
        values = [v for v in self.as_dict().values() if v is not None]
        return sum(values) / len(values) if values else 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _page_haystack(parsed: ParsedDocument, page_number: int) -> str:
    page = parsed.page(page_number)
    if page is None:
        return ""
    return page.text + "\n" + "\n".join(page.tables)


def value_appears(value: float, haystack: str) -> bool:
    if value is None or not haystack:
        return False
    squeezed = haystack.replace(" ", " ").replace(" ", " ")
    for variant in number_variants(value):
        # Require non-digit boundaries so 1,234 does not match inside 11,234.
        pattern = r"(?<![\d.,])" + re.escape(variant) + r"(?![\d])"
        if re.search(pattern, squeezed):
            return True
    return False


def evidence_on_page(evidence: str, haystack: str, threshold: float = 0.7) -> bool:
    """Share of the evidence's words found on the page (robust to line breaks / table pipes)."""
    words = re.findall(r"[a-z0-9]+", _norm(evidence))
    if len(words) < 3:
        return False
    page_words = set(re.findall(r"[a-z0-9]+", _norm(haystack)))
    hits = sum(1 for w in words if w in page_words)
    return hits / len(words) >= threshold


def run_rule_checks(item: ExtractedMetric, parsed: ParsedDocument,
                    reporting_year: Optional[int]) -> RuleChecks:
    checks = RuleChecks()
    metric = METRICS_BY_ID.get(item.metric_id)
    if item.status != "found" or metric is None:
        return checks

    # 1. page exists
    checks.page_in_range = item.page is not None and 1 <= item.page <= parsed.page_count
    if not checks.page_in_range:
        checks.notes.append(f"Cited page {item.page} does not exist in the PDF.")

    # 2./3. value and evidence on the page (not applicable to chart/scan sources)
    visual_source = item.source_type in ("chart", "image")
    if checks.page_in_range:
        haystack = _page_haystack(parsed, item.page)
        if visual_source:
            checks.notes.append("Value read from a chart/image: text-layer checks not applicable.")
        else:
            checks.value_on_page = value_appears(item.value, haystack)
            if not checks.value_on_page:
                checks.notes.append("The number was not found in the text of the cited page.")
            checks.evidence_grounded = evidence_on_page(item.evidence, haystack)
            if not checks.evidence_grounded:
                checks.notes.append("The quoted evidence does not match the page text.")

    # 4. unit
    _, _, recognised = normalise_unit(item.value, item.unit, metric.unit_family)
    checks.unit_plausible = recognised or metric.unit_family in ("count", "ratio")
    if not checks.unit_plausible:
        checks.notes.append(f"Unit '{item.unit}' is unusual for {metric.name}.")

    # 5. ranges
    ok = True
    if item.value is None:
        ok = False
        checks.notes.append("Status is 'found' but no value was returned.")
    else:
        if "%" in (item.unit or "") and not 0 <= item.value <= 100:
            ok = False
            checks.notes.append("Percentage outside 0–100.")
        if metric.unit_family == "ratio" and not 0 < item.value <= 3:
            ok = False
            checks.notes.append("Pay ratio outside the plausible range 0–3.")
        if metric.unit_family in ("emissions", "volume", "mass", "energy") and item.value < 0:
            ok = False
            checks.notes.append("Negative value for a quantity that cannot be negative.")
    if item.year is not None and not 1990 <= item.year <= CURRENT_YEAR:
        ok = False
        checks.notes.append(f"Year {item.year} is outside the plausible range.")
    checks.range_ok = ok
    if reporting_year and item.year and item.year != reporting_year:
        checks.notes.append(f"Value is for {item.year}, the report covers {reporting_year}.")

    # 6. target language
    evidence = _norm(item.evidence)
    checks.not_a_target = not any(word in evidence for word in TARGET_WORDS)
    if not checks.not_a_target:
        checks.notes.append("Evidence contains target/forecast language – check it is an actual result.")
    return checks


# ------------------------------------------------------------------
# Fusion of extractor, rules and verifier into the final record
# ------------------------------------------------------------------
_VERDICT_SCORE = {"VALID": 1.0, "UNCERTAIN": 0.5, "INVALID": 0.0}


def fuse(item: ExtractedMetric, checks: RuleChecks, verification: Optional[VerificationItem],
         parsed: ParsedDocument) -> dict:
    """Build the final result row shown in the UI and exported."""
    metric = METRICS_BY_ID[item.metric_id]
    notes = list(checks.notes)
    value = item.value

    if item.status != "found":
        status = "NOT FOUND"
    else:
        verdict = verification.verdict if verification else None
        # A verifier correction is accepted only if the corrected number is on the page.
        if verification and verification.corrected_value is not None and item.page \
                and value_appears(verification.corrected_value, _page_haystack(parsed, item.page)):
            notes.append(f"Value corrected by the verifier from {value:g} to "
                         f"{verification.corrected_value:g}.")
            value = verification.corrected_value
            verdict = "UNCERTAIN"

        text_supported = checks.value_on_page is not False and checks.evidence_grounded is not False
        if checks.hard_failure or verdict == "INVALID":
            status = "INVALID"
        elif verdict == "VALID" and text_supported and checks.not_a_target is not False:
            status = "VALID"
        elif verdict is None and checks.value_on_page and checks.evidence_grounded \
                and checks.not_a_target is not False:
            status = "VALID"          # rules-only mode (AI verification switched off/failed)
            notes.append("Validated by rules only (no AI verification).")
        else:
            status = "UNCERTAIN"
        if verdict == "VALID" and not text_supported:
            notes.append("Verifier accepted the value but it is not in the page text: marked UNCERTAIN.")

    normalised, standard_unit, _ = normalise_unit(value, item.unit, metric.unit_family)
    verdict_score = _VERDICT_SCORE.get(verification.verdict, 0.5) if verification else 0.5
    final_conf = 0.0 if item.status != "found" else round(
        0.4 * item.confidence + 0.35 * verdict_score + 0.25 * checks.pass_ratio, 2)

    return {
        "metric_id": metric.metric_id,
        "metric": metric.name,
        "pillar": metric.pillar,
        "pillar_name": PILLAR_NAMES[metric.pillar],
        "category": metric.category,
        "status": status,
        "value": value,
        "unit": item.unit,
        "year": item.year,
        "normalized_value": normalised,
        "standard_unit": standard_unit or metric.standard_unit,
        "page": item.page,
        "source_type": item.source_type,
        "evidence": item.evidence,
        "scope2_method": item.scope2_method,
        "reasoning": item.reasoning,
        "extractor_confidence": round(item.confidence, 2),
        "verifier_verdict": verification.verdict if verification else ("—" if item.status != "found" else "not run"),
        "verifier_reason": verification.reason if verification else "",
        "verifier_checks": verification.checks.model_dump() if verification else {},
        "rule_checks": checks.as_dict(),
        "final_confidence": final_conf,
        "history": [h.model_dump() for h in item.history],
        "notes": notes,
    }


def disclosure_completeness(rows: list[dict]) -> dict[str, dict]:
    """Share of requested metrics that were found (VALID or UNCERTAIN), per pillar."""
    summary: dict[str, dict] = {}
    for pillar in ("E", "S", "G"):
        subset = [r for r in rows if r["pillar"] == pillar]
        if not subset:
            continue
        disclosed = sum(1 for r in subset if r["status"] in ("VALID", "UNCERTAIN"))
        valid = sum(1 for r in subset if r["status"] == "VALID")
        summary[pillar] = {"requested": len(subset), "disclosed": disclosed, "valid": valid,
                           "share": round(disclosed / len(subset), 3)}
    total = len(rows)
    disclosed = sum(1 for r in rows if r["status"] in ("VALID", "UNCERTAIN"))
    summary["ALL"] = {"requested": total, "disclosed": disclosed,
                      "valid": sum(1 for r in rows if r["status"] == "VALID"),
                      "share": round(disclosed / total, 3) if total else 0.0}
    return summary
