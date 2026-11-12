"""
Retrieval step (the "R" in RAG) – choose the pages worth sending to the model.

Sustainability reports are long (100–300 pages) and a free-tier model only
gets a limited context, so the app ranks pages with BM25, a classic
lexical-search scoring function, using one query per ESG metric built from
its keywords. It needs no embedding API, so it costs no quota and is fully
explainable: the UI shows which metrics each selected page matched.

Score of page p for metric query q (BM25):
    sum over terms t in q of  IDF(t) * f(t,p) * (k1 + 1) / (f(t,p) + k1 * (1 - b + b * |p| / avgdl))
plus small bonuses for pages with tables / many numbers, and a penalty for
tables of contents and GRI index pages, which mention every metric but
rarely contain the values.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import METRICS_BY_ID
from .pdf_processor import ParsedDocument

_TOKEN_RE = re.compile(r"[a-z0-9³²₂]+")
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:[ ,. ]\d{3})+(?:[.,]\d+)?\b|\b\d{3,}(?:[.,]\d+)?\b")
_STOPWORDS = {"the", "of", "and", "in", "to", "a", "for", "on", "by", "with", "is", "are", "or", "as"}


def tokenize(text: str) -> list[str]:
    """Lower-case unigrams plus bigrams, so phrases like 'scope 1' match as a unit."""
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
    words = [w.replace("₂", "2").replace("²", "2") for w in words]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


@dataclass
class SelectedPage:
    number: int
    score: float
    matched_metrics: list[int] = field(default_factory=list)
    send_image: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    selected: list[SelectedPage]
    per_metric: dict[int, list[tuple[int, float]]]   # metric -> [(page, score)] best first

    @property
    def page_numbers(self) -> list[int]:
        return [p.number for p in self.selected]


class BM25Index:
    """Minimal BM25 index over document pages (k1=1.5, b=0.75)."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        doc_freq: Counter = Counter()
        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())
        n_docs = len(documents)
        self.idf = {t: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def score(self, query: list[str], index: int) -> float:
        freqs = self.term_freqs[index]
        length = self.lengths[index] or 1
        total = 0.0
        for term in set(query):
            tf = freqs.get(term, 0)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
        return total


def _is_navigation_page(text: str) -> bool:
    """Table of contents or GRI/ESRS index: many references, few real values."""
    head = text[:400].lower()
    if "contents" in head or "content index" in text.lower()[:2000]:
        return True
    dotted = len(re.findall(r"\.{4,}", text))
    gri_refs = len(re.findall(r"\b(?:gri|esrs)\s?\d", text.lower()))
    return dotted > 8 or gri_refs > 12


def retrieve_pages(parsed: ParsedDocument, metric_ids: list[int], max_pages: int = 12,
                   per_metric_k: int = 3, reporting_year: int | None = None) -> RetrievalResult:
    """Rank pages for the requested metrics and return the best ``max_pages``."""
    tokenized = [tokenize(page.text) for page in parsed.pages]
    index = BM25Index(tokenized)

    per_metric: dict[int, list[tuple[int, float]]] = {}
    page_scores: dict[int, float] = {}
    page_metrics: dict[int, set[int]] = {}

    for metric_id in metric_ids:
        metric = METRICS_BY_ID[metric_id]
        query = tokenize(" ".join(metric.keywords) + " " + metric.name)
        scored = []
        for i, page in enumerate(parsed.pages):
            if page.char_count < 40:
                continue
            raw = index.score(query, i)
            if raw <= 0:
                continue
            numbers = len(_NUMBER_RE.findall(page.text))
            bonus = 1.0 + min(numbers, 40) / 80          # pages with figures, up to +50%
            if reporting_year and str(reporting_year) in page.text:
                bonus += 0.15
            if _is_navigation_page(page.text):
                bonus *= 0.35
            scored.append((page.number, raw * bonus))
        scored.sort(key=lambda item: item[1], reverse=True)
        per_metric[metric_id] = scored[:5]
        for number, score in scored[:per_metric_k]:
            page_scores[number] = page_scores.get(number, 0.0) + score
            page_metrics.setdefault(number, set()).add(metric_id)

    # Pages relevant to several metrics (ESG data tables) are worth more.
    ranked = sorted(page_scores, key=lambda n: page_scores[n] * (1 + 0.25 * (len(page_metrics[n]) - 1)),
                    reverse=True)

    selected: list[SelectedPage] = []
    for number in ranked[:max_pages]:
        metrics = sorted(page_metrics[number])
        names = ", ".join(METRICS_BY_ID[m].name.split(":")[0] for m in metrics[:3])
        more = f" +{len(metrics) - 3} more" if len(metrics) > 3 else ""
        selected.append(SelectedPage(number=number, score=round(page_scores[number], 2),
                                     matched_metrics=metrics, reasons=[f"matches {names}{more}"]))

    # Scanned pages have no text, so BM25 cannot see them. If the document is
    # mostly scanned, fall back to the first pages so the vision model gets something.
    if not selected:
        for page in parsed.pages[:max_pages]:
            selected.append(SelectedPage(number=page.number, score=0.0,
                                         reasons=["fallback: no text layer to rank"]))

    selected.sort(key=lambda p: p.number)   # keep reading order in the prompt
    return RetrievalResult(selected=selected, per_metric=per_metric)


def choose_image_pages(parsed: ParsedDocument, result: RetrievalResult, max_images: int) -> list[int]:
    """Pick the highest-scoring selected pages that look like charts or scans."""
    if max_images <= 0:
        return []
    candidates = [p for p in result.selected
                  if (page := parsed.page(p.number)) is not None and page.chart_candidate]
    # Pure charts first (their numbers are not in the text layer), then
    # visually complex table pages, each group by relevance score.
    candidates.sort(key=lambda p: (not parsed.page(p.number).tables, p.score), reverse=True)
    chosen = [p.number for p in candidates[:max_images]]
    for page in result.selected:
        if page.number in chosen:
            page.send_image = True
            page.reasons.append("chart/graphics detected → page image sent to the vision model")
    return chosen


def _focused_text(text: str, keywords: list[str], limit: int) -> str:
    """
    Shorten a long page to the lines around keyword hits and numbers, keeping
    order. Plain truncation would often cut off the data table at the bottom.
    """
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    keep: set[int] = set()
    lowered = [line.lower() for line in lines]
    for i, line in enumerate(lowered):
        if any(k in line for k in keywords) or _NUMBER_RE.search(line):
            keep.update(range(max(0, i - 1), min(len(lines), i + 2)))
    out, size = [], 0
    for i in sorted(keep):
        if size + len(lines[i]) + 1 > limit:
            break
        out.append(lines[i])
        size += len(lines[i]) + 1
    return "\n".join(out) + "\n[... page shortened to the relevant lines ...]"


def build_context(parsed: ParsedDocument, result: RetrievalResult, metric_ids: list[int],
                  char_budget: int) -> str:
    """
    Assemble the retrieved pages into one prompt block, labelled by page and
    content type, within ``char_budget`` characters.
    """
    if not result.selected:
        return ""
    keywords = sorted({k for m in metric_ids for k in METRICS_BY_ID[m].keywords})
    per_page = max(1200, char_budget // len(result.selected))
    blocks: list[str] = []
    used = 0
    for selected in result.selected:
        page = parsed.page(selected.number)
        if page is None:
            continue
        kinds = ", ".join(page.content_types)
        parts = [f"===== PAGE {page.number} | content: {kinds} ====="]
        table_text = "\n\n".join(f"[TABLE {i + 1} on page {page.number}]\n{t}"
                                 for i, t in enumerate(page.tables))
        text_limit = max(600, per_page - len(table_text))
        if page.text:
            parts.append("[TEXT]\n" + _focused_text(page.text, keywords, text_limit))
        if table_text:
            parts.append(table_text[: per_page])
        if selected.send_image:
            parts.append(f"[IMAGE] A rendered image of page {page.number} is attached; "
                         "read chart labels and data labels from it.")
        block = "\n".join(parts)
        if used + len(block) > char_budget:
            remaining = char_budget - used
            if remaining < 500:
                break
            block = block[:remaining] + "\n[... truncated to fit the context budget ...]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
