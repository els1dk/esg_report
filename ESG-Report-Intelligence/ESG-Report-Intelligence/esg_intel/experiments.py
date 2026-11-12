"""
Prompt Lab: controlled experiments with prompt techniques and parameters.

The report is parsed and retrieved ONCE; then only the extraction call is
repeated under different configurations (prompt strategy × temperature ×
repeats), so differences in the results come from the prompt/parameters
alone. Each run is scored with metrics that need no human labels:

  found            – metrics reported as found
  grounded %       – found values whose number really appears on the cited page
  evidence %       – found values whose quote really appears on the cited page
  JSON first try   – output was valid JSON without the self-repair step
  stability %      – metrics with identical values across repeated runs
  agreement %      – metrics with the same value as the reference configuration

If ground-truth values are supplied (e.g. the validated ESG database from the
original project) the lab also reports accuracy: extracted values within 1 %
of the validated value.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Callable, Optional

import pandas as pd

from . import prompts
from .config import METRICS_BY_ID, StageParams
from .llm_client import LLMError
from .pipeline import FATAL_ERRORS, ESGPipeline, PipelineError, PreparedDocument
from .validation import normalise_unit, run_rule_checks


@dataclass
class LabRun:
    config_label: str
    strategy: str
    temperature: float
    repeat: int
    ok: bool = True
    error: str = ""
    n_found: int = 0
    grounded_rate: Optional[float] = None
    evidence_rate: Optional[float] = None
    json_first_try: bool = True
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    accuracy: Optional[float] = None
    values: dict = field(default_factory=dict)       # metric_id -> value (None = not found)


def _close(a: Optional[float], b: Optional[float], tolerance: float = 0.01) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tolerance


def run_prompt_lab(pipeline: ESGPipeline, prepared: PreparedDocument, strategies: list[str],
                   temperatures: list[float], repeats: int = 1, top_p: float = 0.9,
                   ground_truth: Optional[dict[int, float]] = None,
                   progress: Optional[Callable[[float, str], None]] = None,
                   pause_s: float = 3.0, sleep: Callable[[float], None] = time.sleep) -> list[LabRun]:
    """Run every (strategy, temperature) combination ``repeats`` times."""
    base = pipeline.config.params["extraction"]
    plan = [(s, t, r) for s in strategies for t in temperatures for r in range(1, repeats + 1)]
    runs: list[LabRun] = []
    for index, (strategy_key, temperature, repeat) in enumerate(plan):
        label = f"{prompts.STRATEGIES[strategy_key].label.split(' (')[0]} · T={temperature:g}"
        if progress:
            progress(index / len(plan), f"Run {index + 1}/{len(plan)}: {label} (repeat {repeat})")
        params = StageParams(temperature=temperature, top_p=top_p,
                             max_output_tokens=base.max_output_tokens, reasoning=base.reasoning)
        run = LabRun(config_label=label, strategy=strategy_key, temperature=temperature, repeat=repeat)
        calls_before = len(pipeline.calls)
        try:
            items, _ = pipeline.extract(prepared, strategy_key, params, stage_name="lab")
        except FATAL_ERRORS:
            raise
        except (LLMError, PipelineError) as error:
            run.ok, run.error = False, str(getattr(error, "user_message", error))[:200]
            runs.append(run)
            continue
        new_calls = pipeline.calls[calls_before:]
        run.json_first_try = not any(c.stage == "repair" for c in new_calls)
        run.latency_s = round(sum(c.latency_s for c in new_calls), 2)
        run.input_tokens = sum(c.input_tokens for c in new_calls)
        run.output_tokens = sum(c.output_tokens for c in new_calls)

        found = [i for i in items if i.status == "found"]
        run.n_found = len(found)
        run.values = {i.metric_id: (i.value if i.status == "found" else None) for i in items}
        checks = [run_rule_checks(i, prepared.parsed, prepared.profile.reporting_year) for i in found]
        text_checks = [c for c in checks if c.value_on_page is not None]
        if text_checks:
            run.grounded_rate = sum(bool(c.value_on_page) for c in text_checks) / len(text_checks)
            run.evidence_rate = sum(bool(c.evidence_grounded) for c in text_checks) / len(text_checks)

        if ground_truth:
            correct = 0
            for metric_id, truth in ground_truth.items():
                item = next((i for i in items if i.metric_id == metric_id), None)
                if item is None or item.status != "found":
                    continue
                normalised, _, _ = normalise_unit(item.value, item.unit, METRICS_BY_ID[metric_id].unit_family)
                if _close(item.value, truth) or _close(normalised, truth):
                    correct += 1
            run.accuracy = correct / len(ground_truth)
        runs.append(run)
        if index < len(plan) - 1 and pause_s:
            sleep(pause_s)   # stay under the free tier's requests-per-minute limit
    if progress:
        progress(1.0, "Prompt Lab finished.")
    return runs


def summarise_runs(runs: list[LabRun], reference_label: Optional[str] = None) -> pd.DataFrame:
    """One row per configuration: means over repeats, stability and agreement."""
    if not runs:
        return pd.DataFrame()
    good = [r for r in runs if r.ok]
    reference = None
    if good:
        labels = [r.config_label for r in good]
        reference_label = reference_label if reference_label in labels else labels[-1]
        reference = next(r for r in good if r.config_label == reference_label)

    rows = []
    for label in dict.fromkeys(r.config_label for r in runs):
        group = [r for r in runs if r.config_label == label]
        ok = [r for r in group if r.ok]
        mean = lambda values: (sum(values) / len(values)) if values else None  # noqa: E731
        row = {
            "Configuration": label,
            "Strategy": group[0].strategy,
            "Temperature": group[0].temperature,
            "Runs": f"{len(ok)}/{len(group)}",
            "Found": mean([r.n_found for r in ok]),
            "Grounded %": mean([r.grounded_rate for r in ok if r.grounded_rate is not None]),
            "Evidence %": mean([r.evidence_rate for r in ok if r.evidence_rate is not None]),
            "JSON first try %": mean([1.0 if r.json_first_try else 0.0 for r in ok]),
            "Accuracy %": mean([r.accuracy for r in ok if r.accuracy is not None]),
            "Stability %": None,
            "Agreement %": None,
            "Latency s": mean([r.latency_s for r in ok]),
            "Tokens": mean([r.input_tokens + r.output_tokens for r in ok]),
        }
        if len(ok) > 1:
            keys = ok[0].values.keys()
            stable = sum(1 for k in keys if all(_close(r.values.get(k), ok[0].values.get(k)) for r in ok))
            row["Stability %"] = stable / len(keys) if keys else None
        if ok and reference is not None:
            keys = reference.values.keys()
            agree = sum(1 for k in keys if _close(ok[0].values.get(k), reference.values.get(k)))
            row["Agreement %"] = agree / len(keys) if keys else None
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex_table(summary: pd.DataFrame, caption: str, label: str = "tab:promptlab") -> str:
    """Booktabs table ready to paste into the Overleaf report (results/prompt_lab_results.tex)."""
    def pct(value):
        return "--" if value is None or pd.isna(value) else f"{100 * value:.0f}\\%"

    def num(value, digits=1):
        return "--" if value is None or pd.isna(value) else f"{value:.{digits}f}"

    has_accuracy = summary["Accuracy %"].notna().any()
    header = ["Strategy", "$T$", "Found", "Grounded", "Evidence", "JSON 1st", "Stability", "Agreement"]
    if has_accuracy:
        header.append("Accuracy")
    header += ["Latency (s)", "Tokens"]
    lines = [
        "\\begin{table}[H]", "\\centering", "\\small",
        "\\begin{tabular}{@{}l" + "r" * (len(header) - 1) + "@{}}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
    ]
    for _, row in summary.iterrows():
        name = prompts.STRATEGIES[row["Strategy"]].label.split(" (")[0].replace("&", "\\&")
        cells = [name, f"{row['Temperature']:g}", num(row["Found"]), pct(row["Grounded %"]),
                 pct(row["Evidence %"]), pct(row["JSON first try %"]), pct(row["Stability %"]),
                 pct(row["Agreement %"])]
        if has_accuracy:
            cells.append(pct(row["Accuracy %"]))
        cells += [num(row["Latency s"]), num(row["Tokens"], 0)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\end{table}"]
    return "\n".join(lines)


def new_experiment_record(filename: str, strategies: list[str], temperatures: list[float],
                          repeats: int, model: str, runs: list[LabRun]) -> dict:
    return {
        "experiment_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "config": {"strategies": strategies, "temperatures": temperatures, "repeats": repeats, "model": model},
        "results": [asdict(r) for r in runs],
    }


# ------------------------------------------------------------------
# Ground truth (e.g. the validated database of the original ESG project)
# ------------------------------------------------------------------
_COLUMN_ALIASES = {
    "metric_id": ("metric_id", "metric id", "theme_id", "id"),
    "year": ("year", "reporting year", "reporting_year"),
    "value": ("value", "validated value", "validated_value"),
    "company": ("company", "company_name", "company name"),
}


def load_ground_truth(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV/XLSX with metric id, year and value columns (names are flexible)."""
    if filename.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(BytesIO(file_bytes))
    else:
        frame = pd.read_csv(BytesIO(file_bytes))
    rename = {}
    lower = {c.lower().strip(): c for c in frame.columns}
    for target, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = target
                break
    frame = frame.rename(columns=rename)
    missing = {"metric_id", "year", "value"} - set(frame.columns)
    if missing:
        raise ValueError(f"Ground-truth file is missing column(s): {', '.join(sorted(missing))}. "
                         "Expected e.g. 'Metric ID', 'Reporting Year', 'Validated Value'.")
    frame = frame.dropna(subset=["metric_id", "year", "value"])
    frame["metric_id"] = pd.to_numeric(frame["metric_id"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["metric_id", "year", "value"])


def ground_truth_for(frame: pd.DataFrame, year: int, company: Optional[str] = None,
                     metric_ids: Optional[list[int]] = None) -> dict[int, float]:
    subset = frame[frame["year"] == year]
    if company and "company" in subset.columns:
        subset = subset[subset["company"].astype(str) == company]
    if metric_ids:
        subset = subset[subset["metric_id"].isin(metric_ids)]
    # If a metric appears twice (e.g. tonnes and %), keep the first row.
    subset = subset.drop_duplicates(subset=["metric_id"])
    return {int(r.metric_id): float(r.value) for r in subset.itertuples()}
