import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "samples" / "Northwind_Renewables_Sustainability_Report_2024.pdf"


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    return SAMPLE.read_bytes()


@pytest.fixture(scope="session")
def sample_parsed(sample_bytes):
    from esg_intel.pdf_processor import parse_pdf
    return parse_pdf(sample_bytes, SAMPLE.name)
