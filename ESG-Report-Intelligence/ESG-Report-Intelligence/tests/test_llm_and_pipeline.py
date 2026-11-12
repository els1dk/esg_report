"""Retry logic, error classification and the full pipeline with the offline provider."""

import json

import httpx
import pytest

from esg_intel.config import CORE_METRIC_IDS
from esg_intel.experiments import run_prompt_lab, summarise_runs, to_latex_table
from esg_intel.export import to_csv, to_excel, to_markdown
from esg_intel.llm_client import (AuthenticationError, ContextTooLargeError, GeminiProvider, LLMRequest,
                                  QuotaExhaustedError, RateLimitError, ServiceTimeoutError,
                                  ServiceUnavailableError, call_with_retries, classify_http_error)
from esg_intel.mock_provider import MockProvider
from esg_intel.pipeline import ESGPipeline, PipelineConfig, PipelineError
from esg_intel.storage import Storage

NO_SLEEP = lambda seconds: None  # noqa: E731


# ---------------- error classification ----------------
def test_classify_errors():
    assert isinstance(classify_http_error(401, "API key not valid"), AuthenticationError)
    minute = classify_http_error(429, "Quota exceeded ... PerMinute ... Please retry in 17.5s.")
    assert isinstance(minute, RateLimitError) and minute.retry_after == 17.5
    assert isinstance(classify_http_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
                      QuotaExhaustedError)
    assert isinstance(classify_http_error(413, "Request too large ... tokens per minute"), ContextTooLargeError)
    assert isinstance(classify_http_error(503, "The model is overloaded"), ServiceUnavailableError)
    assert isinstance(classify_http_error(504, "Deadline exceeded"), ServiceTimeoutError)


def test_gemini_sdk_errors_are_translated():
    from google.genai import errors
    error = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                               "message": "Please retry in 3s."}})
    translated = GeminiProvider._translate(error)
    assert isinstance(translated, RateLimitError) and translated.retry_after == 3
    assert isinstance(GeminiProvider._translate(errors.ClientError(
        400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}})),
        AuthenticationError)


def test_groq_sdk_errors_are_translated():
    import groq
    from esg_intel.llm_client import GroqProvider
    provider = GroqProvider("qwen/qwen3.8-27b", api_key="test-key")
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": "7"}, request=request)
    error = groq.RateLimitError("Rate limit reached for model (RPM)", response=response, body=None)
    translated = provider._translate(error)
    assert isinstance(translated, RateLimitError) and translated.retry_after == 7
    failed = groq.BadRequestError("json_validate_failed", response=httpx.Response(400, request=request),
                                  body={"error": {"code": "json_validate_failed", "failed_generation": "{bad"}})
    assert provider._translate(failed).raw == "{bad"


def test_missing_key_raises():
    with pytest.raises(AuthenticationError):
        GeminiProvider("gemini-3.5-flash-lite", api_key="")


# ---------------- retries ----------------
def test_retry_then_success():
    provider = MockProvider(script=[RateLimitError("429", retry_after=1), ServiceTimeoutError("slow"),
                                    '{"ok": true}'])
    waits, events = [], []
    response = call_with_retries(provider, LLMRequest(system="s", prompt="p"), max_attempts=4,
                                 on_event=events.append, sleep=waits.append)
    assert json.loads(response.text) == {"ok": True}
    assert response.attempts == 3 and waits[0] == 1 and len(events) == 2


def test_non_retryable_errors_fail_fast():
    provider = MockProvider(script=[AuthenticationError("bad key"), '{"ok": true}'])
    with pytest.raises(AuthenticationError):
        call_with_retries(provider, LLMRequest(system="s", prompt="p"), sleep=NO_SLEEP)
    assert len(provider.calls) == 1


def test_retries_exhausted():
    provider = MockProvider(script=[ServiceUnavailableError("503")] * 3)
    with pytest.raises(ServiceUnavailableError):
        call_with_retries(provider, LLMRequest(system="s", prompt="p"), max_attempts=3, sleep=NO_SLEEP)


# ---------------- pipeline ----------------
def _pipeline(provider=None, **overrides):
    cfg = PipelineConfig(provider="mock", model="mock-extractor", **overrides)
    return ESGPipeline(cfg, provider=provider or MockProvider(), sleep=NO_SLEEP)


def test_full_pipeline(sample_bytes):
    result = _pipeline().run(sample_bytes, "Northwind.pdf").to_dict()
    assert result["profile"]["company_name"] == "Northwind Renewables AG"
    assert result["profile"]["reporting_year"] == 2024
    rows = {r["metric_id"]: r for r in result["rows"]}
    assert len(rows) == len(CORE_METRIC_IDS)
    assert rows[8]["value"] == 41250 and rows[8]["status"] == "VALID" and rows[8]["page"] == 4
    assert [c["stage"] for c in result["calls"]] == ["profile", "extraction", "verification", "summary"]
    assert result["status"] == "complete" and result["summary"]


def test_malformed_json_triggers_self_repair(sample_bytes):
    profile = '{"company_name": "Northwind Renewables AG", "reporting_year": 2024}'
    repaired = json.dumps({"metrics": [{"metric_id": 8, "status": "found", "value": 41250, "unit": "tCO2e",
                                        "year": 2024, "page": 4, "source_type": "text",
                                        "evidence": "direct (Scope 1) GHG emissions amounted to 41,250 tCO2e",
                                        "confidence": 0.9}]})
    provider = MockProvider(script=[profile, '{"metrics": [{"metric_id": 8, "value": 41', repaired])
    pipeline = _pipeline(provider, verify=False, summarize=False)
    result = pipeline.run(sample_bytes, "x.pdf").to_dict()
    assert [c["stage"] for c in result["calls"]] == ["profile", "extraction", "repair"]
    assert any("self-repair" in w for w in result["warnings"])
    assert next(r for r in result["rows"] if r["metric_id"] == 8)["value"] == 41250


def test_context_too_large_shrinks_and_retries(sample_bytes):
    provider = MockProvider(script=['{"company_name": "N", "reporting_year": 2024}',
                                    ContextTooLargeError("413 Request too large")])
    result = _pipeline(provider, verify=False, summarize=False).run(sample_bytes, "x.pdf").to_dict()
    assert any("Context reduced" in w for w in result["warnings"])
    assert [c["status"] for c in result["calls"]] == ["ok", "error", "ok"]


def test_auth_error_stops_run(sample_bytes):
    provider = MockProvider(script=[AuthenticationError("API key not valid")])
    with pytest.raises(AuthenticationError):
        _pipeline(provider).run(sample_bytes, "x.pdf")


def test_profile_failure_degrades_gracefully(sample_bytes):
    provider = MockProvider(script=[ServiceUnavailableError("503")] * 4)
    result = _pipeline(provider, verify=False, summarize=False).run(sample_bytes, "Report_2024.pdf").to_dict()
    assert result["profile"]["reporting_year"] == 2024             # from the file name
    assert any("Profiling failed" in w for w in result["warnings"])


def test_verification_failure_marks_partial(sample_bytes):
    class FailingVerifier(MockProvider):
        def generate(self, request):
            raise ServiceUnavailableError("verifier down")
    result = ESGPipeline(PipelineConfig(provider="mock", model="mock-extractor", summarize=False),
                         provider=MockProvider(), verifier=FailingVerifier(), sleep=NO_SLEEP
                         ).run(sample_bytes, "x.pdf").to_dict()
    assert result["status"] == "partial"
    assert all(r["verifier_verdict"] in ("not run", "—") for r in result["rows"])


def test_unparseable_after_repair_raises(sample_bytes):
    provider = MockProvider(script=['{"company_name": "N"}', "no json at all", "still no json"])
    with pytest.raises(PipelineError):
        _pipeline(provider, verify=False, summarize=False).run(sample_bytes, "x.pdf")


# ---------------- storage, exports, prompt lab ----------------
def test_storage_and_exports(sample_bytes, tmp_path):
    result = _pipeline().run(sample_bytes, "Northwind.pdf").to_dict()
    store = Storage(tmp_path / "t.db", tmp_path / "uploads")
    store.save_pdf(result["file_hash"], sample_bytes)
    store.save_analysis(result)
    assert store.load_analysis(result["analysis_id"])["rows"] == result["rows"]
    assert store.find_by_hash(result["file_hash"]) and store.load_pdf(result["file_hash"]) == sample_bytes
    assert store.usage_by_model()
    assert to_excel(result)[:2] == b"PK"          # xlsx is a zip file
    assert b"Scope 1 GHG emissions total" in to_csv(result)
    assert "| Metric |" in to_markdown(result)
    store.delete_analysis(result["analysis_id"])
    assert store.list_analyses() == []


def test_prompt_lab(sample_bytes):
    pipeline = _pipeline()
    prepared = pipeline.prepare(sample_bytes, "Northwind.pdf")
    runs = run_prompt_lab(pipeline, prepared, ["zero_shot", "role_few_shot_cot"], [0.0, 1.0], repeats=2,
                          ground_truth={8: 41250, 9: 12430}, pause_s=0, sleep=NO_SLEEP)
    assert len(runs) == 8 and all(r.ok for r in runs)
    summary = summarise_runs(runs)
    assert len(summary) == 4 and summary["Stability %"].notna().all()
    assert (summary["Accuracy %"] == 1.0).all()
    latex = to_latex_table(summary, "test")
    assert "\\toprule" in latex and "Accuracy" in latex


def test_sample_ground_truth_file_loads():
    from pathlib import Path
    from esg_intel.experiments import ground_truth_for, load_ground_truth
    path = Path(__file__).resolve().parent.parent / "samples" / "northwind_ground_truth.csv"
    frame = load_ground_truth(path.read_bytes(), path.name)
    truth = ground_truth_for(frame, 2024, "Northwind Renewables AG")
    assert len(truth) == 15 and truth[8] == 41250 and truth[50] == 1_180_000
