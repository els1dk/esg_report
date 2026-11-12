"""
Multimodal PDF processing with PyMuPDF.

For every page the processor records three kinds of content:
  * text      – the page's text layer
  * tables    – tables detected with PyMuPDF's table finder, as Markdown
  * visuals   – embedded images and vector drawings; pages that look like
                charts (or scanned pages without text) are rendered to
                images and sent to a vision model.

Expensive work (table detection, drawing analysis) is done lazily, only for
the pages the retriever selects, so a 300-page report stays fast.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pymupdf  # PyMuPDF

from .config import MAX_PAGES_TO_SCAN, MAX_UPLOAD_MB


class PDFValidationError(ValueError):
    """The uploaded file cannot be analysed. The message is shown to the user."""


@dataclass
class PageContent:
    number: int                     # 1-based page number, as shown in PDF viewers
    text: str = ""
    image_count: int = 0
    image_area_ratio: float = 0.0   # share of the page covered by raster images
    drawing_count: int = -1         # vector paths; -1 = not analysed yet
    tables: list[str] = field(default_factory=list)  # Markdown tables
    tables_done: bool = False
    error: Optional[str] = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_scanned(self) -> bool:
        """Little or no text layer but a large image: probably a scanned page."""
        return self.char_count < 80 and self.image_area_ratio > 0.4

    @property
    def chart_candidate(self) -> bool:
        """Heuristic: many vector paths or a large image usually means a chart/figure."""
        many_paths = self.drawing_count >= 60
        big_image = self.image_area_ratio >= 0.12
        return many_paths or big_image or self.is_scanned

    @property
    def content_types(self) -> list[str]:
        kinds = []
        if self.char_count >= 80:
            kinds.append("text")
        if self.tables:
            kinds.append("table")
        if self.is_scanned:
            kinds.append("scanned")
        elif self.chart_candidate:
            # Graphics next to detected tables are usually styled table cells.
            kinds.append("graphics" if self.tables else "chart/figure")
        return kinds or ["empty"]


@dataclass
class ParsedDocument:
    filename: str
    file_hash: str
    page_count: int
    pages: list[PageContent]
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    pages_scanned_limit_hit: bool = False

    def page(self, number: int) -> Optional[PageContent]:
        if 1 <= number <= len(self.pages):
            return self.pages[number - 1]
        return None

    @property
    def text_pages(self) -> int:
        return sum(1 for p in self.pages if p.char_count >= 80)

    @property
    def scanned_ratio(self) -> float:
        if not self.pages:
            return 0.0
        return sum(1 for p in self.pages if p.is_scanned) / len(self.pages)

    def head_text(self, max_pages: int = 3, max_chars: int = 6000) -> str:
        """First pages' text, used to profile the report (company, year, ...)."""
        chunks = []
        for page in self.pages[:max_pages]:
            if page.text.strip():
                chunks.append(f"--- PAGE {page.number} ---\n{page.text.strip()}")
        return "\n\n".join(chunks)[:max_chars]


# ------------------------------------------------------------------
# Validation and parsing
# ------------------------------------------------------------------
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_bytes(data: bytes, filename: str = "upload.pdf") -> None:
    """Raise PDFValidationError with a user-friendly message if the file is unusable."""
    if not data:
        raise PDFValidationError("The uploaded file is empty.")
    size_mb = len(data) / 1_000_000
    if size_mb > MAX_UPLOAD_MB:
        raise PDFValidationError(
            f"The file is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB. "
            "Split the report or upload the sustainability section only."
        )
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise PDFValidationError(
            f"'{filename}' is not a valid PDF (the file does not start with a PDF header)."
        )
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass and not document.authenticate(""):
                raise PDFValidationError("This PDF is password-protected. Upload an unlocked copy.")
            if document.page_count < 1:
                raise PDFValidationError("The PDF has no pages.")
    except PDFValidationError:
        raise
    except Exception as error:  # corrupt file, unsupported structure, ...
        raise PDFValidationError(f"The PDF could not be opened: {error}") from error


def parse_pdf(data: bytes, filename: str, max_pages: int = MAX_PAGES_TO_SCAN) -> ParsedDocument:
    """
    Read text and image statistics for every page.

    A page that fails to parse is recorded with an error and skipped; the
    analysis continues with the remaining pages.
    """
    validate_pdf_bytes(data, filename)
    warnings: list[str] = []

    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            document.authenticate("")
        metadata = {k: v for k, v in (document.metadata or {}).items() if v}
        total = document.page_count
        limit = min(total, max_pages)
        if total > max_pages:
            warnings.append(
                f"The report has {total} pages; only the first {max_pages} were scanned."
            )

        pages: list[PageContent] = []
        failed: list[int] = []
        for index in range(limit):
            content = PageContent(number=index + 1)
            try:
                page = document[index]
                content.text = _clean_text(page.get_text("text", sort=True))
                content.image_count, content.image_area_ratio = _image_stats(page)
            except Exception as error:
                content.error = f"{type(error).__name__}: {error}"
                failed.append(index + 1)
            pages.append(content)

    if failed:
        shown = ", ".join(map(str, failed[:10])) + (" ..." if len(failed) > 10 else "")
        warnings.append(
            f"{len(failed)} page(s) could not be processed ({shown}). "
            "Analysis continued with the successfully extracted pages."
        )

    parsed = ParsedDocument(
        filename=filename,
        file_hash=file_sha256(data),
        page_count=total,
        pages=pages,
        metadata=metadata,
        warnings=warnings,
        pages_scanned_limit_hit=total > max_pages,
    )
    if parsed.text_pages == 0:
        warnings.append(
            "No text layer was found (scanned document). The app will rely on page "
            "images and a vision model, which covers fewer pages."
        )
    elif parsed.scanned_ratio > 0.3:
        warnings.append(
            f"{parsed.scanned_ratio:.0%} of pages look scanned; their content is only "
            "available to the model as images."
        )
    return parsed


def enrich_pages(data: bytes, parsed: ParsedDocument, page_numbers: Iterable[int],
                 max_tables_per_page: int = 4) -> None:
    """Detect tables and vector drawings on the selected pages (lazy, in place)."""
    with pymupdf.open(stream=data, filetype="pdf") as document:
        for number in page_numbers:
            content = parsed.page(number)
            if content is None or content.tables_done or content.error:
                continue
            page = document[number - 1]
            try:
                content.drawing_count = len(page.get_drawings())
            except Exception:
                content.drawing_count = 0
            try:
                found = page.find_tables()
                for table in list(found.tables)[:max_tables_per_page]:
                    markdown = _clean_markdown(table.to_markdown(clean=True))
                    # Skip "tables" that are really layout boxes with no numbers.
                    if markdown and re.search(r"\d", markdown):
                        content.tables.append(markdown[:4000])
            except Exception as error:
                parsed.warnings.append(f"Table detection failed on page {number}: {error}")
            content.tables_done = True


# ------------------------------------------------------------------
# Rendering (for the vision model and the evidence viewer)
# ------------------------------------------------------------------
def render_page_for_llm(data: bytes, page_number: int, max_side: int = 1600) -> tuple[bytes, str]:
    """Render a page as JPEG (smaller than PNG) for multimodal prompts."""
    with pymupdf.open(stream=data, filetype="pdf") as document:
        page = document[page_number - 1]
        longest = max(page.rect.width, page.rect.height) or 1
        zoom = max(0.5, min(2.5, max_side / longest))
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        try:
            return pixmap.tobytes("jpeg", jpg_quality=80), "image/jpeg"
        except Exception:
            return pixmap.tobytes("png"), "image/png"


def render_page_with_highlights(data: bytes, page_number: int, terms: Iterable[str],
                                zoom: float = 1.6) -> tuple[bytes, int]:
    """
    Render a page as PNG with yellow highlights on every occurrence of the
    given terms (the extracted value, evidence fragments, ...).
    Returns (png_bytes, number_of_highlights).
    """
    hits = 0
    with pymupdf.open(stream=data, filetype="pdf") as document:
        page = document[page_number - 1]
        for term in terms:
            term = (term or "").strip()
            if len(term) < 2:
                continue
            try:
                rects = page.search_for(term)[:12]
            except Exception:
                rects = []
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 0.85, 0))
                annot.update()
                hits += 1
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, annots=True)
        return pixmap.tobytes("png"), hits


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown(markdown: str) -> str:
    """Remove HTML entities, <br> tags, bold markers and 'ColN' placeholders from table Markdown."""
    text = html.unescape(html.unescape(markdown or ""))
    text = text.replace("<br>", " ").replace("**", "")
    text = re.sub(r"\bCol\d+\b", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _image_stats(page) -> tuple[int, float]:
    """Number of raster images and the share of the page area they cover."""
    page_area = abs(page.rect) or 1.0
    infos = page.get_image_info()
    covered = 0.0
    for info in infos:
        x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
        covered += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    return len(infos), min(1.0, covered / page_area)


def number_variants(value: float) -> list[str]:
    """
    Ways a number can be printed in a report, used to find it on the page:
    12345.0 -> ["12,345", "12 345", "12.345", "12345"]; 36.5 -> ["36.5", "36,5"].
    """
    variants: list[str] = []
    if value is None:
        return variants
    if float(value).is_integer():
        integer = int(round(value))
        plain = str(integer)
        variants += [f"{integer:,}", f"{integer:,}".replace(",", " "),
                     f"{integer:,}".replace(",", "."), f"{integer:,}".replace(",", " "), plain]
    else:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        variants += [text, text.replace(".", ",")]
        whole, _, decimals = text.partition(".")
        if len(whole) > 3:
            grouped = f"{int(whole):,}"
            variants += [f"{grouped}.{decimals}", f"{grouped.replace(',', '.')},{decimals}",
                         f"{grouped.replace(',', ' ')},{decimals}"]
    seen, unique = set(), []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return unique
