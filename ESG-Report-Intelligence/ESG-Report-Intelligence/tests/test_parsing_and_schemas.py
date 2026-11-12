"""JSON parsing, schema coercion and PDF validation."""

import pymupdf
import pytest

from esg_intel.json_utils import JSONParseError, as_list_payload, parse_llm_json
from esg_intel.pdf_processor import PDFValidationError, number_variants, parse_pdf, validate_pdf_bytes
from esg_intel.schemas import ExtractedMetric, VerificationItem, parse_number, parse_year


# ---------------- JSON ----------------
def test_json_plain_and_fenced():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}


def test_json_think_tags_and_trailing_commas():
    raw = '<think>let me reason {not json}</think>{"metrics": [{"x": 1,},],}'
    assert parse_llm_json(raw) == {"metrics": [{"x": 1}]}


def test_json_python_literals():
    assert parse_llm_json("{'a': None}".replace("'", '"')) == {"a": None}


def test_json_truncated_is_flagged():
    with pytest.raises(JSONParseError) as info:
        parse_llm_json('{"metrics": [{"metric_id": 8, "value": 12')
    assert info.value.truncated


def test_json_empty():
    with pytest.raises(JSONParseError):
        parse_llm_json("   ")


def test_bare_list_is_wrapped():
    assert as_list_payload([{"a": 1}], "metrics") == {"metrics": [{"a": 1}]}
    assert as_list_payload({"results": [1]}, "metrics") == {"metrics": [1]}


# ---------------- schemas ----------------
@pytest.mark.parametrize("text, expected", [
    ("12,345", 12345), ("12 345", 12345), ("1.234.567", 1234567), ("1,234.5", 1234.5),
    ("1.234,5", 1234.5), ("36,5", 36.5), ("45%", 45), ("(120)", -120), ("n/a", None), ("Not found", None),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_parse_year_fiscal():
    assert parse_year("FY2023/24") == 2024
    assert parse_year("2022") == 2022
    assert parse_year("unknown") is None


def test_extracted_metric_coercion():
    item = ExtractedMetric.model_validate({
        "metric_id": "8", "status": "Found", "value": "41,250", "unit": "tCO2e", "year": "FY2024",
        "page": "p. 4", "source_type": "Table 1", "confidence": "high",
        "history": [{"year": "2023", "value": "44,870"}, {"year": "bad", "value": 1}]})
    assert item.metric_id == 8 and item.status == "found" and item.value == 41250
    assert item.page == 4 and item.source_type == "table" and item.confidence == 0.9
    assert [h.year for h in item.history] == [2023]       # unusable point dropped


def test_verification_verdict_normalised():
    assert VerificationItem.model_validate({"metric_id": 9, "verdict": "correct"}).verdict == "VALID"
    assert VerificationItem.model_validate({"metric_id": 9, "verdict": "unsure"}).verdict == "UNCERTAIN"


# ---------------- PDF ----------------
def test_invalid_pdf_rejected():
    with pytest.raises(PDFValidationError, match="not a valid PDF"):
        validate_pdf_bytes(b"hello world", "notes.pdf")
    with pytest.raises(PDFValidationError, match="empty"):
        validate_pdf_bytes(b"", "empty.pdf")


def test_encrypted_pdf_rejected(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(PDFValidationError, match="password"):
        validate_pdf_bytes(path.read_bytes(), "locked.pdf")


def test_sample_parses_with_scanned_page(sample_parsed):
    assert sample_parsed.page_count == 10
    assert sample_parsed.page(9).is_scanned                 # assurance page is an image only
    assert "41,250" in sample_parsed.page(4).text


def test_page_limit_warning(sample_bytes):
    parsed = parse_pdf(sample_bytes, "x.pdf", max_pages=3)
    assert len(parsed.pages) == 3 and parsed.pages_scanned_limit_hit
    assert any("only the first 3" in w for w in parsed.warnings)


def test_number_variants():
    assert {"41,250", "41 250", "41.250", "41250"} <= set(number_variants(41250.0))
    assert {"34.2", "34,2"} <= set(number_variants(34.2))
