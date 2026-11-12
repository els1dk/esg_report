"""
The analysis pipeline: PDF -> structured, verified ESG data -> insights.

    1. Parse & validate PDF ............ local (PyMuPDF)
    2. Profile the report .............. LLM call #1 (zero-shot + role)
    3. Retrieve relevant pages ......... local BM25 + table detection + chart rendering
    4. Extract metrics ................. LLM call #2 (multimodal, few-shot + CoT)
       4b. JSON self-repair ............ extra LLM call only if the JSON is broken
    5. Rule-based validation ........... local (value on page, units, ranges ...)
    6. AI verification ................. LLM call #3 (auditor persona, may use a 2nd model)
    7. Fuse results + completeness ..... local
    8. Insight summary ................. LLM call #4 (analyst persona, higher temperature)

Failures degrade gracefully: if profiling, verification or the summary fail,
the run continues and the result is marked "partial" with a warning. Only
problems that make the core extraction impossible stop the run.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Optional

from pydantic import ValidationError

from . import prompts
from .config import CORE_METRIC_IDS, DEFAULT_STAGE_PARAMS, METRICS_BY_ID, PROVIDERS, StageParams, find_model
from .json_utils import JSONParseError, as_list_payload, parse_llm_json
from .llm_client import (AuthenticationError, BadRequestError, BaseProvider, ContextTooLargeError, ImageInput,
                         LLMError, LLMRequest, LLMResponse, ModelNotFoundError, QuotaExhaustedError,
                         call_with_retries, make_provider)
from .pdf_processor import ParsedDocument, enrich_pages, number_variants, parse_pdf, render_page_for_llm
from .retrieval import RetrievalResult, build_context, choose_image_pages, retrieve_pages
from .schemas import (ExtractedMetric, ExtractionOutput, ReportProfile, SummaryOutput, VerificationItem,
                      VerificationOutput)
from .validation import disclosure_completeness, fuse, run_rule_checks

# Errors after which further LLM calls are pointless (stop the run).
FATAL_ERRORS = (AuthenticationError, QuotaExhaustedError, ModelNotFoundError)


# ------------------------------------------------------------------
# Configuration and result containers
# ------------------------------------------------------------------
@dataclass
class PipelineConfig:
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    api_key: str = ""
    strategy: str = prompts.DEFAULT_STRATEGY
    metric_ids: list[int] = field(default_factory=lambda: list(CORE_METRIC_IDS))
    params: dict[str, StageParams] = field(
        default_factory=lambda: {k: StageParams(**asdict(v)) for k, v in DEFAULT_STAGE_PARAMS.items()})
    max_pages_to_model: int = 12
    use_images: bool = True
    max_images: int = 3
    context_chars: Optional[int] = None          # None = model default
    verify: bool = True
    verifier_provider: Optional[str] = None      # None = same as extractor
    verifier_model: Optional[str] = None
    verifier_api_key: str = ""
    summarize: bool = True
    max_attempts: int = 4
    timeout_s: float = 180.0

    def public_dict(self) -> dict:
        """Settings without secrets, stored with each analysis."""
        data = asdict(self)
        data.pop("api_key", None)
        data.pop("verifier_api_key", None)
        return data


@dataclass
class CallRecord:
    stage: str
    provider: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    reasoning: str
    n_images: int
    prompt_chars: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 0
    status: str = "ok"
    error: str = ""


@dataclass
class PreparedDocument:
    """Everything computed before the extraction call (reused by the Prompt Lab)."""
    parsed: ParsedDocument
    pdf_bytes: bytes
    profile: ReportProfile
    retrieval: RetrievalResult
    context: str
    images: list[ImageInput]
    image_pages: list[int]


@dataclass
class AnalysisResult:
    analysis_id: str
    created_at: str
    filename: str
    file_hash: str
    page_count: int
    profile: dict
    page_stats: dict
    retrieval: list[dict]
    rows: list[dict]
    completeness: dict
    summary: Optional[dict]
    calls: list[dict]
    warnings: list[str]
    config: dict
    status: str = "complete"           # complete | partial
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        return cls(**data)


class PipelineError(RuntimeError):
    """A failure that stops the analysis; ``user_message`` is shown in the UI."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def format_number(value: Optional[float]) -> str:
    if value is None:
        return "null"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def looks_like_loop(text: str) -> bool:
    """Detect degenerate repetition (a known failure mode at very low temperature)."""
    if len(text) < 2000:
        return False
    tail = text[-1500:]
    chunk = tail[-60:]
    return tail.count(chunk) >= 4


def _source_window(parsed: ParsedDocument, page_number: Optional[int], value: Optional[float],
                   evidence: str, width: int) -> str:
    """Text around the value on the cited page, taken from the PDF (not from the model)."""
    page = parsed.page(page_number) if page_number else None
    if page is None:
        return "(cited page does not exist)"
    text = page.text
    position = -1
    for variant in number_variants(value) if value is not None else []:
        position = text.find(variant)
        if position != -1:
            break
    if position == -1 and evidence:
        position = text.lower().find(evidence.lower()[:30])
    if position == -1:
        window = text[:width]
        note = "\n(the extracted number does not appear in this page's text layer)"
    else:
        start = max(0, position - width // 2)
        window = text[start:start + width]
        note = ""
    for table in page.tables:
        if value is not None and any(v in table for v in number_variants(value)):
            window += "\n[TABLE ON THIS PAGE]\n" + table[: max(400, width)]
            break
    return window.strip() + note


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class ESGPipeline:
    def __init__(self, config: PipelineConfig,
                 progress: Optional[Callable[[float, str], None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 provider: Optional[BaseProvider] = None,
                 verifier: Optional[BaseProvider] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self._progress = progress or (lambda fraction, message: None)
        self._log = log or (lambda message: None)
        self._sleep = sleep
        self.calls: list[CallRecord] = []
        self.warnings: list[str] = []
        self.model_info = find_model(config.provider, config.model)
        self.provider = provider or make_provider(
            config.provider, config.model, config.api_key,
            max_images=self.model_info.max_images, vision=self.model_info.vision)
        self.verifier = verifier
        if self.verifier is None and config.verify:
            if config.verifier_provider and config.verifier_model and (
                    config.verifier_provider, config.verifier_model) != (config.provider, config.model):
                info = find_model(config.verifier_provider, config.verifier_model)
                self.verifier = make_provider(config.verifier_provider, config.verifier_model,
                                              config.verifier_api_key, max_images=info.max_images,
                                              vision=info.vision)
            else:
                self.verifier = self.provider

    # -------------------------------------------------------------
    # LLM call wrapper: retries, logging and call records
    # -------------------------------------------------------------
    def _call(self, stage: str, provider: BaseProvider, system: str, prompt: str,
              params: StageParams, images: Optional[list[ImageInput]] = None,
              temperature_override: Optional[float] = None) -> LLMResponse:
        temperature = params.temperature if temperature_override is None else temperature_override
        # Respect the model's output limit (Groq's free tier counts it against 8k tokens/minute).
        info = find_model(provider.name if provider.name in PROVIDERS else "mock", provider.model)
        max_tokens = min(params.max_output_tokens, info.max_output_tokens)
        request = LLMRequest(system=system, prompt=prompt, images=images or [],
                             temperature=temperature, top_p=params.top_p,
                             max_output_tokens=max_tokens, reasoning=params.reasoning,
                             timeout_s=self.config.timeout_s)
        record = CallRecord(stage=stage, provider=provider.name, model=provider.model,
                            temperature=temperature, top_p=params.top_p,
                            max_output_tokens=max_tokens, reasoning=params.reasoning,
                            n_images=len(request.images), prompt_chars=len(system) + len(prompt))
        try:
            response = call_with_retries(provider, request, max_attempts=self.config.max_attempts,
                                         on_event=self._log, sleep=self._sleep)
        except LLMError as error:
            record.status, record.error = "error", f"{type(error).__name__}: {str(error)[:300]}"
            self.calls.append(record)
            raise
        record.input_tokens, record.output_tokens = response.input_tokens, response.output_tokens
        record.latency_s, record.attempts = round(response.latency_s, 2), response.attempts
        if response.truncated:
            record.status = "truncated"
        self.calls.append(record)
        for warning in response.warnings:
            self.warnings.append(f"[{stage}] {warning}")
        return response

    def _parse_with_repair(self, stage: str, provider: BaseProvider, response: LLMResponse,
                           model_cls, list_key: str, schema_hint: str, params: StageParams):
        """Parse + validate JSON; on failure ask the model once to repair its own output."""
        try:
            data = as_list_payload(parse_llm_json(response.text), list_key) if list_key else \
                parse_llm_json(response.text)
            return model_cls.model_validate(data)
        except (JSONParseError, ValidationError) as error:
            truncated = response.truncated or getattr(error, "truncated", False)
            message = str(error).splitlines()[0][:300]
            self._log(f"{stage}: output was not valid ({message}) → asking the model to repair it")
            self.warnings.append(f"[{stage}] Invalid JSON on first try ({message}); self-repair used.")
            repair_prompt = prompts.build_repair_prompt(response.text, message, truncated, schema_hint)
            repair_params = StageParams(temperature=0.0, top_p=params.top_p,
                                        max_output_tokens=params.max_output_tokens, reasoning=params.reasoning)
            repaired = self._call("repair", provider, prompts.NEUTRAL_SYSTEM, repair_prompt, repair_params)
            try:
                data = as_list_payload(parse_llm_json(repaired.text), list_key) if list_key else \
                    parse_llm_json(repaired.text)
                return model_cls.model_validate(data)
            except (JSONParseError, ValidationError) as error2:
                raise PipelineError(
                    f"The model's {stage} output was not valid JSON, even after a repair attempt.",
                    str(error2)[:500]) from error2

    # -------------------------------------------------------------
    # Stage 1-3: preparation (parse, profile, retrieve)
    # -------------------------------------------------------------
    def prepare(self, pdf_bytes: bytes, filename: str) -> PreparedDocument:
        cfg = self.config
        self._progress(0.03, "Validating and parsing the PDF …")
        parsed = parse_pdf(pdf_bytes, filename)
        self.warnings.extend(parsed.warnings)

        self._progress(0.12, "Profiling the report (company, year, frameworks) …")
        profile = self._profile(parsed)

        self._progress(0.25, "Retrieving the most relevant pages …")
        retrieval = retrieve_pages(parsed, cfg.metric_ids, max_pages=cfg.max_pages_to_model,
                                   reporting_year=profile.reporting_year)
        enrich_pages(pdf_bytes, parsed, retrieval.page_numbers)

        images, image_pages = [], []
        if cfg.use_images and self.model_info.vision:
            image_pages = choose_image_pages(parsed, retrieval, min(cfg.max_images, self.model_info.max_images))
            for number in image_pages:
                try:
                    data, mime = render_page_for_llm(pdf_bytes, number)
                    images.append(ImageInput(data=data, mime_type=mime, label=f"Image of PAGE {number}:"))
                except Exception as error:
                    self.warnings.append(f"Page {number} could not be rendered as an image: {error}")
        elif cfg.use_images and not self.model_info.vision:
            self.warnings.append(f"{cfg.model} is text-only, so charts are not analysed visually. "
                                 "Choose a vision model for multimodal analysis.")

        budget = cfg.context_chars or self.model_info.context_chars
        context = build_context(parsed, retrieval, cfg.metric_ids, budget)
        return PreparedDocument(parsed, pdf_bytes, profile, retrieval, context, images, image_pages)

    def _profile(self, parsed: ParsedDocument) -> ReportProfile:
        head = parsed.head_text()
        fallback = ReportProfile(company_name=parsed.metadata.get("author") or parsed.filename.rsplit(".", 1)[0],
                                 one_line_description="Profile not available (AI profiling failed).")
        years = re.findall(r"(?:19|20)\d{2}", parsed.filename)
        if years:
            fallback.reporting_year = int(years[-1])
        if not head:
            self.warnings.append("The first pages have no text; the report profile is based on the file name.")
            return fallback
        system, prompt = prompts.build_profile_prompt(parsed.filename, parsed.metadata, head)
        params = self.config.params["profile"]
        try:
            response = self._call("profile", self.provider, system, prompt, params)
            return self._parse_with_repair("profile", self.provider, response, ReportProfile, "",
                                           "the profile object described above", params)
        except FATAL_ERRORS:
            raise
        except (LLMError, PipelineError) as error:
            self.warnings.append(f"[profile] Profiling failed ({error}); continuing with file-name defaults.")
            return fallback

    # -------------------------------------------------------------
    # Stage 4: extraction
    # -------------------------------------------------------------
    def extract(self, prepared: PreparedDocument, strategy_key: Optional[str] = None,
                params: Optional[StageParams] = None, stage_name: str = "extraction"
                ) -> tuple[list[ExtractedMetric], LLMResponse]:
        cfg = self.config
        strategy = prompts.STRATEGIES[strategy_key or cfg.strategy]
        params = params or cfg.params["extraction"]
        context, images = prepared.context, prepared.images
        budget = cfg.context_chars or self.model_info.context_chars

        for shrink in range(3):
            system, prompt = prompts.build_extraction_prompt(strategy, cfg.metric_ids, context,
                                                             prepared.profile, len(images))
            try:
                response = self._call(stage_name, self.provider, system, prompt, params, images)
                break
            except ContextTooLargeError:
                if shrink == 2:
                    raise
                budget //= 2
                images = images[: max(0, len(images) - 1)]
                self._log(f"Prompt too large → retrying with a smaller context ({budget:,} characters, "
                          f"{len(images)} image(s))")
                self.warnings.append(f"[{stage_name}] Context reduced to {budget:,} characters to fit the model.")
                context = build_context(prepared.parsed, prepared.retrieval, cfg.metric_ids, budget)
            except BadRequestError as error:
                if error.raw:   # Groq JSON-mode failure: repair the failed generation
                    response = LLMResponse(text=error.raw, model=self.provider.model)
                    break
                raise

        # Very low temperatures can make reasoning models loop; retry once at the
        # vendor-recommended default temperature (1.0) if that happens.
        if looks_like_loop(response.text) and params.temperature < 0.5:
            self._log("Repetitive output detected → retrying extraction at temperature 1.0")
            self.warnings.append(f"[{stage_name}] Degenerate repetition at temperature "
                                 f"{params.temperature}; retried at 1.0.")
            response = self._call(stage_name, self.provider, system, prompt, params, images,
                                  temperature_override=1.0)

        output = self._parse_with_repair(stage_name, self.provider, response, ExtractionOutput, "metrics",
                                         prompts.OUTPUT_SCHEMA_EXTRACTION, params)
        return self._complete_metric_list(output.metrics), response

    def _complete_metric_list(self, items: list[ExtractedMetric]) -> list[ExtractedMetric]:
        """Keep one item per requested metric, in catalogue order; fill gaps with not_found."""
        by_id: dict[int, ExtractedMetric] = {}
        for item in items:
            if item.metric_id not in self.config.metric_ids:
                continue
            current = by_id.get(item.metric_id)
            # If the model returned a metric twice, keep the found / more confident one.
            if current is None or (item.status == "found" and
                                   (current.status != "found" or item.confidence > current.confidence)):
                by_id[item.metric_id] = item
        result = []
        for metric_id in self.config.metric_ids:
            if metric_id in by_id:
                result.append(by_id[metric_id])
            else:
                result.append(ExtractedMetric(metric_id=metric_id, status="not_found",
                                              reasoning="The model did not return this metric."))
        return result

    # -------------------------------------------------------------
    # Stage 6: AI verification
    # -------------------------------------------------------------
    def verify(self, prepared: PreparedDocument, items: list[ExtractedMetric]) -> dict[int, VerificationItem]:
        found = [i for i in items if i.status == "found"]
        if not found or self.verifier is None:
            return {}
        verifier_info = find_model(self.verifier.name if self.verifier.name in PROVIDERS else "mock",
                                   self.verifier.model)
        budget = (self.config.context_chars or verifier_info.context_chars) * 0.8
        width = int(max(300, min(1400, budget / max(1, len(found)) - 300)))

        blocks, images = [], []
        for n, item in enumerate(found, start=1):
            metric = METRICS_BY_ID[item.metric_id]
            window = _source_window(prepared.parsed, item.page, item.value, item.evidence, width)
            blocks.append(
                f"\n[{n}] metric_id {item.metric_id} — {metric.name} (standard unit {metric.standard_unit})\n"
                f"    extracted value: {format_number(item.value)} | unit: {item.unit} | year: {item.year} | "
                f"page: {item.page} | source: {item.source_type}\n"
                f"    extractor evidence: \"{item.evidence[:240]}\"\n"
                f"    SOURCE (page {item.page}, taken directly from the PDF):\n\"\"\"\n{window}\n\"\"\"")
            if item.source_type in ("chart", "image") and verifier_info.vision and item.page \
                    and len(images) < verifier_info.max_images:
                image = next((im for im, p in zip(prepared.images, prepared.image_pages) if p == item.page), None)
                if image is None:
                    try:
                        data, mime = render_page_for_llm(prepared.pdf_bytes, item.page)
                        image = ImageInput(data, mime, f"Image of PAGE {item.page}:")
                    except Exception:
                        image = None
                if image:
                    images.append(image)

        system, prompt = prompts.build_verification_prompt("".join(blocks), len(images))
        params = self.config.params["verification"]
        response = self._call("verification", self.verifier, system, prompt, params, images)
        output = self._parse_with_repair("verification", self.verifier, response, VerificationOutput,
                                         "verifications", prompts.VERIFICATION_SCHEMA, params)
        return {v.metric_id: v for v in output.verifications}

    # -------------------------------------------------------------
    # Stage 8: summary
    # -------------------------------------------------------------
    def summarize(self, profile: ReportProfile, rows: list[dict], completeness: dict) -> SummaryOutput:
        lines = []
        for row in rows:
            if row["status"] not in ("VALID", "UNCERTAIN"):
                continue
            history = ", ".join(f"{h['year']}={format_number(h['value'])}" for h in row["history"])
            lines.append(f"- [{row['status']}] {row['metric']} ({row['pillar_name']}): "
                         f"{format_number(row['value'])} {row['unit']} for {row['year'] or 'year n/a'} "
                         f"(page {row['page']}, {row['source_type']})"
                         + (f"; history: {history}" if history else ""))
        missing = "\n".join(f"- {r['metric']} ({r['pillar_name']})" for r in rows if r["status"] == "NOT FOUND")
        invalid = [r for r in rows if r["status"] == "INVALID"]
        if invalid:
            missing += "\n" + "\n".join(f"- {r['metric']}: extracted value failed verification"
                                        for r in invalid)
        comp = "\n".join(f"- {k}: {v['disclosed']}/{v['requested']} metrics disclosed ({v['share']:.0%})"
                         for k, v in completeness.items())
        system, prompt = prompts.build_summary_prompt(profile, "\n".join(lines), missing, comp)
        params = self.config.params["summary"]
        response = self._call("summary", self.provider, system, prompt, params)
        return self._parse_with_repair("summary", self.provider, response, SummaryOutput, "",
                                       prompts.SUMMARY_SCHEMA, params)

    # -------------------------------------------------------------
    # Full run
    # -------------------------------------------------------------
    def run(self, pdf_bytes: bytes, filename: str) -> AnalysisResult:
        started = time.perf_counter()
        cfg = self.config
        partial = False

        prepared = self.prepare(pdf_bytes, filename)
        parsed, profile = prepared.parsed, prepared.profile

        self._progress(0.35, f"Extracting {len(cfg.metric_ids)} ESG metrics from "
                             f"{len(prepared.retrieval.selected)} pages and {len(prepared.images)} image(s) …")
        try:
            items, _ = self.extract(prepared)
        except FATAL_ERRORS:
            raise
        except LLMError as error:
            raise PipelineError(error.user_message, str(error)) from error

        self._progress(0.6, "Running rule-based checks against the PDF …")
        checks = {i.metric_id: run_rule_checks(i, parsed, profile.reporting_year) for i in items}

        verifications: dict[int, VerificationItem] = {}
        if cfg.verify and any(i.status == "found" for i in items):
            self._progress(0.68, "AI verification of every extracted value …")
            try:
                verifications = self.verify(prepared, items)
            except FATAL_ERRORS as error:
                partial = True
                self.warnings.append(f"[verification] Skipped: {error.user_message}")
            except (LLMError, PipelineError) as error:
                partial = True
                self.warnings.append(f"[verification] Failed ({error}); results are validated by rules only.")

        rows = [fuse(i, checks[i.metric_id], verifications.get(i.metric_id), parsed) for i in items]
        completeness = disclosure_completeness(rows)

        summary = None
        if cfg.summarize:
            self._progress(0.85, "Writing the ESG insight summary …")
            try:
                summary = self.summarize(profile, rows, completeness).model_dump()
            except FATAL_ERRORS as error:
                partial = True
                self.warnings.append(f"[summary] Skipped: {error.user_message}")
            except (LLMError, PipelineError) as error:
                partial = True
                self.warnings.append(f"[summary] Failed ({error}).")

        self._progress(1.0, "Done.")
        page_stats = {
            "pages": parsed.page_count,
            "text_pages": parsed.text_pages,
            "scanned_pages": sum(1 for p in parsed.pages if p.is_scanned),
            "pages_with_images": sum(1 for p in parsed.pages if p.image_count),
            "selected_pages": len(prepared.retrieval.selected),
            "tables_found": sum(len(parsed.page(p).tables) for p in prepared.retrieval.page_numbers),
            "chart_pages_sent": len(prepared.images),
            "context_chars": len(prepared.context),
        }
        retrieval = [{"page": p.number, "score": p.score,
                      "metrics": [METRICS_BY_ID[m].name for m in p.matched_metrics],
                      "content": ", ".join(parsed.page(p.number).content_types),
                      "image_sent": p.send_image, "why": "; ".join(p.reasons)}
                     for p in prepared.retrieval.selected]
        return AnalysisResult(
            analysis_id=uuid.uuid4().hex[:12],
            created_at=datetime.now().isoformat(timespec="seconds"),
            filename=filename,
            file_hash=parsed.file_hash,
            page_count=parsed.page_count,
            profile=profile.model_dump(),
            page_stats=page_stats,
            retrieval=retrieval,
            rows=rows,
            completeness=completeness,
            summary=summary,
            calls=[asdict(c) for c in self.calls],
            warnings=self.warnings,
            config=cfg.public_dict(),
            status="partial" if partial else "complete",
            duration_s=round(time.perf_counter() - started, 1),
        )
