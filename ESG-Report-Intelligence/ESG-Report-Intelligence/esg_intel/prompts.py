"""
Prompt templates and prompt-engineering strategies.

Techniques used (and switchable, so they can be compared in the Prompt Lab):

  * Role prompting      – a domain persona in the system instruction
                          (ESG data specialist, ESG auditor, ESG analyst).
  * Zero-shot           – task description + output format only (baseline,
                          also used for the simple report-profiling step).
  * Few-shot            – worked input→output examples, including a
                          negative example (a target that must NOT be extracted).
  * Chain-of-thought    – the task is decomposed into explicit reasoning steps
    (decomposition)       (locate → filter → period → value/unit → scope →
                          decide). The model reasons internally and returns only
                          a one-sentence justification, not its full reasoning.
  * Structured output   – JSON mode + a fixed schema validated with Pydantic.
  * Self-repair         – if the JSON is invalid, the error is fed back to the
                          model in a follow-up prompt (iterative prompting).
  * Self-verification   – a second, independent "auditor" prompt checks every
                          value against the source text (optionally with a
                          different model: cross-model verification).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import METRICS_BY_ID
from .schemas import ReportProfile


@dataclass(frozen=True)
class PromptStrategy:
    key: str
    label: str
    role: bool
    few_shot: bool
    cot: bool
    description: str


STRATEGIES: dict[str, PromptStrategy] = {
    s.key: s for s in (
        PromptStrategy("zero_shot", "Zero-shot (baseline)", role=False, few_shot=False, cot=False,
                       description="Plain instructions and output format. No persona, no examples."),
        PromptStrategy("role", "Role prompting", role=True, few_shot=False, cot=False,
                       description="Adds an ESG data-specialist persona with domain rules."),
        PromptStrategy("role_few_shot", "Role + few-shot", role=True, few_shot=True, cot=False,
                       description="Persona plus four worked examples (text, table, target trap, chart)."),
        PromptStrategy("role_few_shot_cot", "Role + few-shot + chain-of-thought (recommended)",
                       role=True, few_shot=True, cot=True,
                       description="Persona, examples and a six-step reasoning procedure per metric."),
    )
}
DEFAULT_STRATEGY = "role_few_shot_cot"


# ------------------------------------------------------------------
# System instructions (role prompting)
# ------------------------------------------------------------------
NEUTRAL_SYSTEM = "You are a helpful assistant. Answer with JSON only."

EXTRACTOR_ROLE = (
    "You are a senior ESG data extraction specialist who has prepared data for CDP, "
    "GRI and ESRS/CSRD disclosures. You read sustainability reports the way an auditor "
    "does: you only record a figure when the report states it clearly for the company "
    "as a whole and for a specific year. You know that reports mix actual results with "
    "targets, baselines, intensities and value-chain categories, and you never confuse them. "
    "You never estimate, calculate or invent numbers. Answer with JSON only."
)

AUDITOR_ROLE = (
    "You are a meticulous ESG assurance auditor (ISAE 3000 experience). Your job is to "
    "check extracted data points against the source text of the report. You are sceptical: "
    "if the source does not clearly support a value, you say so. Answer with JSON only."
)

ANALYST_ROLE = (
    "You are a senior sustainability analyst who writes briefings for investors and ESG "
    "rating committees. You write clear, neutral, evidence-based English. You only use the "
    "figures you are given, always with their unit and year, and you point out when data "
    "is missing or uncertain. Answer with JSON only."
)

PROFILER_ROLE = (
    "You identify document metadata from the first pages of corporate reports. "
    "Answer with JSON only."
)


# ------------------------------------------------------------------
# Few-shot examples for extraction
# ------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """\
EXAMPLES (fictional companies; they show the expected behaviour, not real data)

Example 1 – value in running text
Excerpt:
===== PAGE 14 | content: text =====
"In 2023 our direct (Scope 1) greenhouse gas emissions fell to 42,321 tCO2e, 6% below 2022."
Requested: 8 (Scope 1 GHG emissions total), reporting year 2023
Output item:
{"metric_id": 8, "status": "found", "value": 42321, "unit": "tCO2e", "year": 2023, "page": 14,
 "source_type": "text", "evidence": "our direct (Scope 1) greenhouse gas emissions fell to 42,321 tCO2e",
 "scope2_method": "", "reasoning": "Explicit group Scope 1 total for 2023.", "confidence": 0.95, "history": []}

Example 2 – multi-year table, scale in the header
Excerpt:
===== PAGE 58 | content: text, table =====
[TABLE 1 on page 58]
| GHG emissions (kt CO2e)      | 2021 | 2022 | 2023 |
| Scope 2 (market-based)       | 18.4 | 15.1 | 12.9 |
| Scope 2 (location-based)     | 21.0 | 20.2 | 19.8 |
Requested: 9 (Scope 2 GHG emissions total), reporting year 2023
Output item:
{"metric_id": 9, "status": "found", "value": 12.9, "unit": "kt CO2e", "year": 2023, "page": 58,
 "source_type": "table", "evidence": "Scope 2 (market-based) | 18.4 | 15.1 | 12.9",
 "scope2_method": "market-based", "reasoning": "Market-based row of the GHG table; unit kt from the header.",
 "confidence": 0.9, "history": [{"year": 2021, "value": 18.4}, {"year": 2022, "value": 15.1}]}

Example 3 – a TARGET, which must NOT be extracted
Excerpt:
===== PAGE 9 | content: text =====
"We aim to have 40% women in management positions by 2030 (2023: 31%)."
Requested: 75 (Women in management), reporting year 2023
Output item:
{"metric_id": 75, "status": "found", "value": 31, "unit": "%", "year": 2023, "page": 9,
 "source_type": "text", "evidence": "(2023: 31%)", "scope2_method": "",
 "reasoning": "40% is a 2030 target; the actual 2023 value is 31%.", "confidence": 0.85, "history": []}
(If only the 40% target were given, the correct output would be status "not_found" with value null.)

Example 4 – value read from a chart image
Excerpt:
===== PAGE 33 | content: chart/figure =====
[IMAGE] A rendered image of page 33 is attached.
(The bar chart "Water consumption (thousand m³)" shows data labels 1,140 for 2022 and 1,085 for 2023.)
Requested: 50 (Water consumption: total), reporting year 2023
Output item:
{"metric_id": 50, "status": "found", "value": 1085, "unit": "thousand m³", "year": 2023, "page": 33,
 "source_type": "chart", "evidence": "Bar chart 'Water consumption (thousand m³)', data label 1,085 for 2023",
 "scope2_method": "", "reasoning": "Read from the data label of the 2023 bar.", "confidence": 0.75,
 "history": [{"year": 2022, "value": 1140}]}
"""

COT_PROCEDURE = """\
HOW TO WORK – follow these steps for every requested metric (reason silently; write only the JSON):
Step 1 – Locate: find every passage, table row or chart label that could contain the metric.
Step 2 – Filter: discard targets, goals, forecasts, scenarios, intensity ratios (per revenue,
         per employee), single sites, subsidiaries, other companies and single Scope 3 categories.
Step 3 – Period: determine which reporting year each remaining candidate belongs to.
Step 4 – Value and unit: read the number exactly; check thousands separators, decimal commas and
         scale words in table headers (thousand, million, kt, Mt, ML).
Step 5 – Scope: confirm the figure covers the whole company or group.
Step 6 – Decide: pick the best candidate for the reporting year, or return not_found.
         Summarise your decision in ONE sentence in "reasoning" (no step-by-step text).
"""

OUTPUT_SCHEMA_EXTRACTION = """\
OUTPUT FORMAT – return exactly one JSON object, no Markdown, no text outside the JSON:
{"metrics": [
  {"metric_id": <int>, "status": "found" | "not_found", "value": <number or null>,
   "unit": "<unit exactly as printed>", "year": <int or null>, "page": <int or null>,
   "source_type": "text" | "table" | "chart" | "image" | "none",
   "evidence": "<verbatim quote, max 200 characters>", "scope2_method": "<market-based | location-based | not specified | empty>",
   "reasoning": "<one sentence>", "confidence": <0.0-1.0>,
   "history": [{"year": <int>, "value": <number>}]}
]}
Return exactly one item per requested metric_id, in the order listed.
"""


def _metric_table(metric_ids: list[int]) -> str:
    lines = ["id | metric | standard unit | definition"]
    for metric_id in metric_ids:
        m = METRICS_BY_ID[metric_id]
        lines.append(f"{m.metric_id} | {m.name} | {m.standard_unit} | {m.definition}")
    return "\n".join(lines)


def _profile_line(profile: ReportProfile | None) -> str:
    if profile is None:
        return "Company: unknown | Reporting year: unknown"
    year = profile.reporting_year or "unknown"
    return (f"Company: {profile.company_name} | Reporting year: {year} | "
            f"Report type: {profile.report_type} | Sector: {profile.sector}")


# ------------------------------------------------------------------
# Stage 1 – profiling (zero-shot + light role)
# ------------------------------------------------------------------
def build_profile_prompt(filename: str, metadata: dict, head_text: str) -> tuple[str, str]:
    meta = "; ".join(f"{k}: {v}" for k, v in metadata.items() if k in {"title", "author", "subject", "creationDate"})
    user = f"""\
Below are the first pages of a corporate report.
File name: {filename}
PDF metadata: {meta or "none"}

{head_text}

TASK: Identify the report's metadata.
Return one JSON object with these keys:
{{"company_name": "...", "reporting_year": <int or null>,
  "report_type": "Sustainability report | Integrated report | Annual report | ESG data book | Climate/TCFD report | Other",
  "sector": "...", "country": "...",
  "frameworks": ["GRI", "ESRS/CSRD", "SASB", "TCFD", "ISSB", "CDP", "UN SDGs", ...only those mentioned],
  "assurance": "Limited | Reasonable | None stated", "language": "...",
  "one_line_description": "<max 25 words>"}}
The reporting year is the fiscal year the report covers (not the publication year).
Use "Unknown" for anything that is not stated.
"""
    return PROFILER_ROLE, user


# ------------------------------------------------------------------
# Stage 2 – extraction (strategy-dependent)
# ------------------------------------------------------------------
def build_extraction_prompt(strategy: PromptStrategy, metric_ids: list[int], context: str,
                            profile: ReportProfile | None, n_images: int) -> tuple[str, str]:
    year = profile.reporting_year if profile and profile.reporting_year else "the most recent year reported"
    images = (f"{n_images} page image(s) are attached; each one is labelled in the excerpts with [IMAGE]."
              if n_images else "No page images are attached.")

    rules = f"""\
RULES
1. Use ONLY the excerpts and attached page images. Do not use outside knowledge.
2. Report the value for the reporting year ({year}). If a table also shows earlier years, put them in "history".
3. Never extract targets, goals, forecasts or intensity ratios.
4. Scope 3: only an explicit total, never a single category. Scope 2: prefer market-based and say which
   method in "scope2_method".
5. Copy the number as printed (convert words: "1.2 million" -> 1200000) and the unit as printed.
6. "page" is the number in the ===== PAGE n ===== header where the value appears.
7. "evidence" is a short verbatim quote from that page containing the number (for charts: the chart title
   and data label).
8. If a metric is not clearly reported, return status "not_found" and value null.
   A missing value is better than a guessed value.
9. "confidence" is your probability (0.0-1.0) that the value is correct.
"""

    parts = [
        "REPORT EXCERPTS (retrieved pages; text, tables and chart/scan markers)",
        context,
        "",
        "=" * 60,
        "TASK: Extract the ESG metrics listed below from the report excerpts above.",
        _profile_line(profile),
        images,
        "",
        "METRICS TO EXTRACT",
        _metric_table(metric_ids),
        "",
        rules if strategy.role or strategy.cot or strategy.few_shot else
        "Extract each metric's value for the reporting year. Use not_found if it is missing.",
    ]
    if strategy.cot:
        parts += [COT_PROCEDURE]
    if strategy.few_shot:
        parts += [FEW_SHOT_EXAMPLES]
    parts += [OUTPUT_SCHEMA_EXTRACTION]

    system = EXTRACTOR_ROLE if strategy.role else NEUTRAL_SYSTEM
    return system, "\n".join(parts)


# ------------------------------------------------------------------
# Stage 2b – JSON self-repair (iterative prompting)
# ------------------------------------------------------------------
def build_repair_prompt(raw_output: str, error: str, truncated: bool, schema_hint: str) -> str:
    shorten = ""
    if truncated:
        shorten = ("Your previous answer was cut off by the output limit. Return a SHORTER answer: "
                   "evidence max 100 characters, reasoning max 12 words, and an empty history list.\n")
    return f"""\
Your previous answer could not be parsed as JSON.
Problem: {error}
{shorten}
Previous answer (may be incomplete):
<<<
{raw_output[:6000]}
>>>

Return the same information as ONE valid JSON object with this structure:
{schema_hint}
Return only the JSON.
"""


# ------------------------------------------------------------------
# Stage 3 – verification (role + chain-of-thought checklist)
# ------------------------------------------------------------------
VERIFICATION_SCHEMA = """\
{"verifications": [
  {"metric_id": <int>, "verdict": "VALID" | "UNCERTAIN" | "INVALID",
   "checks": {"metric_matches": <bool>, "year_matches": <bool>, "value_matches": <bool>,
              "unit_matches": <bool>, "company_scope": <bool>},
   "reason": "<one or two sentences>", "corrected_value": <number or null>}
]}"""


def build_verification_prompt(items_block: str, n_images: int) -> tuple[str, str]:
    images = (f"\n{n_images} page image(s) are attached for items that come from charts.\n"
              if n_images else "")
    user = f"""\
TASK: Verify each extracted ESG data point against the SOURCE text taken directly from the PDF.
{images}
For each item, work through this checklist:
1. metric_matches – the source describes exactly this metric (Scope 1 alone, not Scope 1+2; a total,
   not one category; the whole workforce, not one country).
2. year_matches – the value belongs to the stated year.
3. value_matches – the number is identical to the source, including scale (thousand / million / kt).
4. unit_matches – the unit is correct.
5. company_scope – it is the company's actual reported result, not a target, forecast, supplier or site.

Verdict rules:
- VALID: all five checks are true.
- INVALID: the metric or value is clearly wrong, is a target/forecast, or does not appear in the source.
- UNCERTAIN: the source is ambiguous or incomplete (for example the value is only in a chart, or the
  table header is missing).
If the value is wrong but the correct value is visible in the source, put it in "corrected_value".

ITEMS TO VERIFY
{items_block}

Return exactly one JSON object with this structure (one entry per item):
{VERIFICATION_SCHEMA}
"""
    return AUDITOR_ROLE, user


# ------------------------------------------------------------------
# Stage 4 – insight summary (role, higher temperature)
# ------------------------------------------------------------------
SUMMARY_SCHEMA = """\
{"executive_summary": "<120-180 words>",
 "environmental": "<2-4 sentences>", "social": "<2-4 sentences>", "governance": "<2-4 sentences>",
 "strengths": ["<max 4 items>"], "disclosure_gaps": ["<max 5 items>"],
 "red_flags": ["<max 3 items, empty if none>"], "recommended_follow_up": ["<max 3 items>"]}"""


def build_summary_prompt(profile: ReportProfile | None, metrics_block: str, missing_block: str,
                         completeness_block: str) -> tuple[str, str]:
    user = f"""\
Write an ESG briefing for the report described below.

REPORT
{_profile_line(profile)}
Frameworks: {", ".join(profile.frameworks) if profile and profile.frameworks else "not stated"}
Assurance: {profile.assurance if profile else "not stated"}

VERIFIED DATA POINTS (status VALID or UNCERTAIN; history = earlier years from the same report)
{metrics_block or "none"}

METRICS NOT DISCLOSED IN THE REPORT
{missing_block or "none"}

DISCLOSURE COMPLETENESS (computed by the application)
{completeness_block}

Guidelines:
- Use only the figures above, always with unit and year. Do not invent numbers or trends.
- Describe a trend only when "history" gives at least two years; give the percentage change.
- Say explicitly when a figure is marked UNCERTAIN.
- Treat missing core metrics as disclosure gaps, not as poor performance.
- Red flags are rising emissions, missing Scope 3, values that failed verification, or similar.

Return one JSON object with this structure:
{SUMMARY_SCHEMA}
"""
    return ANALYST_ROLE, user
