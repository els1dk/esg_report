"""Exports: Excel workbook, CSV, JSON and a Markdown briefing."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

EXPORT_COLUMNS = ["metric_id", "metric", "pillar_name", "category", "status", "value", "unit", "year",
                  "normalized_value", "standard_unit", "page", "source_type", "scope2_method", "evidence",
                  "reasoning", "extractor_confidence", "verifier_verdict", "verifier_reason",
                  "final_confidence", "notes"]


def metrics_frame(result: dict) -> pd.DataFrame:
    frame = pd.DataFrame(result["rows"])
    if frame.empty:
        return frame
    frame["notes"] = frame["notes"].apply(lambda notes: " | ".join(notes) if isinstance(notes, list) else notes)
    frame.insert(0, "company", result["profile"].get("company_name"))
    frame.insert(1, "source_file", result["filename"])
    return frame[["company", "source_file"] + [c for c in EXPORT_COLUMNS if c in frame.columns]]


def history_frame(result: dict) -> pd.DataFrame:
    rows = []
    for row in result["rows"]:
        if row["status"] in ("VALID", "UNCERTAIN"):
            if row["year"] is not None and row["value"] is not None:
                rows.append({"metric": row["metric"], "year": row["year"], "value": row["value"],
                             "unit": row["unit"], "status": row["status"]})
            for point in row["history"]:
                rows.append({"metric": row["metric"], "year": point["year"], "value": point["value"],
                             "unit": row["unit"], "status": row["status"]})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["metric", "year"]).sort_values(["metric", "year"])
    return frame


def to_excel(result: dict) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        metrics_frame(result).to_excel(writer, sheet_name="ESG metrics", index=False)
        history_frame(result).to_excel(writer, sheet_name="Time series", index=False)
        pd.DataFrame([result["profile"]]).T.rename(columns={0: "value"}).to_excel(writer, sheet_name="Report profile")
        summary = result.get("summary") or {}
        if summary:
            flat = {k: (" • ".join(v) if isinstance(v, list) else v) for k, v in summary.items()}
            pd.DataFrame([flat]).T.rename(columns={0: "text"}).to_excel(writer, sheet_name="AI summary")
        pd.DataFrame(result["calls"]).to_excel(writer, sheet_name="LLM calls", index=False)
        pd.DataFrame(result["retrieval"]).to_excel(writer, sheet_name="Retrieved pages", index=False)
        pd.DataFrame({"warning": result["warnings"] or ["none"]}).to_excel(writer, sheet_name="Warnings",
                                                                          index=False)
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in column[:200])
                sheet.column_dimensions[column[0].column_letter].width = min(60, max(10, width + 2))
    return buffer.getvalue()


def to_csv(result: dict) -> bytes:
    return metrics_frame(result).to_csv(index=False).encode("utf-8-sig")


def to_json(result: dict) -> bytes:
    return json.dumps(result, indent=2, default=str, ensure_ascii=False).encode("utf-8")


def to_markdown(result: dict) -> str:
    profile, summary = result["profile"], result.get("summary") or {}
    lines = [f"# ESG briefing – {profile.get('company_name', 'Unknown')} "
             f"({profile.get('reporting_year') or 'year n/a'})", "",
             f"*Source: {result['filename']} · {result['page_count']} pages · generated "
             f"{result['created_at']} with {result['config'].get('model')}*", ""]
    if summary:
        lines += ["## Executive summary", summary.get("executive_summary", ""), ""]
        for key, title in (("environmental", "Environmental"), ("social", "Social"),
                           ("governance", "Governance")):
            if summary.get(key):
                lines += [f"## {title}", summary[key], ""]
        for key, title in (("strengths", "Strengths"), ("disclosure_gaps", "Disclosure gaps"),
                           ("red_flags", "Red flags"), ("recommended_follow_up", "Recommended follow-up")):
            if summary.get(key):
                lines += [f"## {title}"] + [f"- {item}" for item in summary[key]] + [""]
    lines += ["## Extracted metrics", "", "| Metric | Value | Year | Page | Status |", "|---|---|---|---|---|"]
    for row in result["rows"]:
        value = "—" if row["value"] is None else f"{row['value']:,.6g} {row['unit']}"
        lines.append(f"| {row['metric']} | {value} | {row['year'] or '—'} | {row['page'] or '—'} | "
                     f"{row['status']} |")
    return "\n".join(lines) + "\n"
