"""Retrieval ranking, unit normalisation, rule checks and result fusion."""

import pytest

from esg_intel.config import CORE_METRIC_IDS
from esg_intel.retrieval import build_context, choose_image_pages, retrieve_pages, tokenize
from esg_intel.schemas import ExtractedMetric, VerificationItem
from esg_intel.validation import fuse, normalise_unit, run_rule_checks, value_appears


def test_tokenize_bigrams():
    tokens = tokenize("Scope 1 emissions")
    assert "scope_1" in tokens and "emissions" in tokens


def test_emissions_page_ranks_first(sample_parsed):
    result = retrieve_pages(sample_parsed, CORE_METRIC_IDS, max_pages=8, reporting_year=2024)
    for metric_id in (8, 9, 10):
        assert result.per_metric[metric_id][0][0] == 4     # GHG page is the top hit
    assert result.per_metric[71][0][0] == 7                 # board diversity on the people page


def test_chart_pages_selected_for_images(sample_bytes, sample_parsed):
    from esg_intel.pdf_processor import enrich_pages
    result = retrieve_pages(sample_parsed, CORE_METRIC_IDS, max_pages=10, reporting_year=2024)
    enrich_pages(sample_bytes, sample_parsed, result.page_numbers)
    assert 5 in choose_image_pages(sample_parsed, result, 3)   # water chart page
    context = build_context(sample_parsed, result, CORE_METRIC_IDS, 20_000)
    assert "===== PAGE 4" in context and "[TABLE 1 on page 4]" in context
    assert len(context) <= 20_000 + 200


@pytest.mark.parametrize("value, unit, family, expected", [
    (12.9, "kt CO2e", "emissions", (12900, "t CO2e")),
    (1.2, "Mt CO2e", "emissions", (1_200_000, "t CO2e")),
    (41250, "tCO2e", "emissions", (41250, "t CO2e")),
    (1180, "thousand m³", "volume", (1_180_000, "m³")),
    (350, "ML", "volume", (350_000, "m³")),
    (2.5, "GWh", "energy", (2500, "MWh")),
    (34.2, "%", "percent", (34.2, "%")),
])
def test_normalise_unit(value, unit, family, expected):
    normalised, standard, recognised = normalise_unit(value, unit, family)
    assert recognised and standard == expected[1] and normalised == pytest.approx(expected[0])


def test_value_appears_boundaries():
    assert value_appears(41250, "amounted to 41,250 tCO2e")
    assert not value_appears(1250, "amounted to 41,250 tCO2e")   # no partial matches


def _item(**kw):
    base = dict(metric_id=8, status="found", value=41250, unit="tCO2e", year=2024, page=4, source_type="text",
                evidence="direct (Scope 1) GHG emissions amounted to 41,250 tCO2e", confidence=0.9)
    base.update(kw)
    return ExtractedMetric(**base)


def test_rule_checks_pass_for_true_value(sample_parsed):
    checks = run_rule_checks(_item(), sample_parsed, 2024)
    assert checks.value_on_page and checks.evidence_grounded and checks.page_in_range and checks.not_a_target


def test_hallucinated_value_is_caught(sample_parsed):
    checks = run_rule_checks(_item(value=39999, evidence="Scope 1 emissions were 39,999 t"), sample_parsed, 2024)
    assert checks.value_on_page is False
    row = fuse(_item(value=39999), checks, VerificationItem(metric_id=8, verdict="VALID"), sample_parsed)
    assert row["status"] == "UNCERTAIN"      # verifier said VALID, but the PDF disagrees


def test_target_and_bad_page(sample_parsed):
    checks = run_rule_checks(_item(page=99), sample_parsed, 2024)
    assert checks.page_in_range is False and checks.hard_failure
    target = run_rule_checks(_item(metric_id=75, value=40, unit="%", page=3,
                                   evidence="aim to have 40% women in management positions by 2028"),
                             sample_parsed, 2024)
    assert target.not_a_target is False


def test_fusion_statuses(sample_parsed):
    good = _item()
    checks = run_rule_checks(good, sample_parsed, 2024)
    assert fuse(good, checks, VerificationItem(metric_id=8, verdict="VALID"), sample_parsed)["status"] == "VALID"
    assert fuse(good, checks, VerificationItem(metric_id=8, verdict="INVALID"), sample_parsed)["status"] == "INVALID"
    assert fuse(good, checks, None, sample_parsed)["status"] == "VALID"            # rules-only mode
    missing = ExtractedMetric(metric_id=48, status="not_found")
    assert fuse(missing, run_rule_checks(missing, sample_parsed, 2024), None, sample_parsed)["status"] == "NOT FOUND"


def test_verifier_correction_applied_only_if_on_page(sample_parsed):
    wrong = _item(value=44870, evidence="Scope 1 (direct) | 47,310 | 44,870 | 41,250", source_type="table")
    checks = run_rule_checks(wrong, sample_parsed, 2024)
    row = fuse(wrong, checks, VerificationItem(metric_id=8, verdict="INVALID", corrected_value=41250), sample_parsed)
    assert row["value"] == 41250 and row["status"] == "UNCERTAIN"
    row = fuse(wrong, checks, VerificationItem(metric_id=8, verdict="INVALID", corrected_value=11111), sample_parsed)
    assert row["value"] == 44870 and row["status"] == "INVALID"
