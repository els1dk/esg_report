"""
Local persistence: a small SQLite database plus a folder of uploaded PDFs.

  analyses     – one row per completed analysis (full result as JSON)
  llm_calls    – one row per LLM call (stage, model, parameters, tokens, latency)
  experiments  – Prompt Lab runs (configuration + metrics as JSON)

Uploaded PDFs are stored once under data/uploads/<sha256>.pdf so a past
analysis can be re-opened later with its evidence pages.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .config import DB_PATH, UPLOADS_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id   TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    filename      TEXT NOT NULL,
    file_hash     TEXT NOT NULL,
    company       TEXT,
    reporting_year INTEGER,
    provider      TEXT,
    model         TEXT,
    strategy      TEXT,
    n_requested   INTEGER,
    n_valid       INTEGER,
    n_uncertain   INTEGER,
    n_invalid     INTEGER,
    n_not_found   INTEGER,
    status        TEXT,
    result_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    stage         TEXT, provider TEXT, model TEXT,
    temperature   REAL, top_p REAL, max_output_tokens INTEGER, reasoning TEXT,
    input_tokens  INTEGER, output_tokens INTEGER, latency_s REAL, attempts INTEGER,
    status        TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    filename      TEXT,
    config_json   TEXT,
    results_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_hash ON analyses(file_hash);
"""


class Storage:
    def __init__(self, db_path: Path = DB_PATH, uploads_dir: Path = UPLOADS_DIR):
        self.db_path = Path(db_path)
        self.uploads_dir = Path(uploads_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------- files -------------------------
    def save_pdf(self, file_hash: str, data: bytes) -> Path:
        path = self.uploads_dir / f"{file_hash}.pdf"
        if not path.exists():
            path.write_bytes(data)
        return path

    def load_pdf(self, file_hash: str) -> Optional[bytes]:
        path = self.uploads_dir / f"{file_hash}.pdf"
        return path.read_bytes() if path.exists() else None

    # ----------------------- analyses ------------------------
    def save_analysis(self, result: dict) -> None:
        rows = result["rows"]
        count = lambda status: sum(1 for r in rows if r["status"] == status)  # noqa: E731
        cfg = result.get("config", {})
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO analyses VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result["analysis_id"], result["created_at"], result["filename"], result["file_hash"],
                 result["profile"].get("company_name"), result["profile"].get("reporting_year"),
                 cfg.get("provider"), cfg.get("model"), cfg.get("strategy"), len(rows),
                 count("VALID"), count("UNCERTAIN"), count("INVALID"), count("NOT FOUND"),
                 result.get("status"), json.dumps(result, default=str)))
            conn.execute("DELETE FROM llm_calls WHERE analysis_id = ?", (result["analysis_id"],))
            conn.executemany(
                """INSERT INTO llm_calls (analysis_id, stage, provider, model, temperature, top_p,
                   max_output_tokens, reasoning, input_tokens, output_tokens, latency_s, attempts, status, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(result["analysis_id"], c["stage"], c["provider"], c["model"], c["temperature"], c["top_p"],
                  c["max_output_tokens"], c["reasoning"], c["input_tokens"], c["output_tokens"],
                  c["latency_s"], c["attempts"], c["status"], c["error"]) for c in result["calls"]])

    def list_analyses(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.execute(
                """SELECT analysis_id, created_at, filename, company, reporting_year, provider, model,
                          strategy, n_requested, n_valid, n_uncertain, n_invalid, n_not_found, status
                   FROM analyses ORDER BY created_at DESC LIMIT ?""", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def load_analysis(self, analysis_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT result_json FROM analyses WHERE analysis_id = ?",
                               (analysis_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def find_by_hash(self, file_hash: str) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT analysis_id, created_at, model, strategy FROM analyses WHERE file_hash = ? "
                "ORDER BY created_at DESC", (file_hash,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_analysis(self, analysis_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))

    def usage_by_model(self) -> list[dict]:
        """Token usage and latency per model across all stored analyses."""
        with self._connect() as conn:
            cursor = conn.execute(
                """SELECT provider, model, stage, COUNT(*) AS calls,
                          SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                          ROUND(AVG(latency_s), 2) AS avg_latency_s,
                          SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
                   FROM llm_calls GROUP BY provider, model, stage ORDER BY provider, model, stage""")
            return [dict(row) for row in cursor.fetchall()]

    # ---------------------- experiments ----------------------
    def save_experiment(self, experiment_id: str, created_at: str, filename: str,
                        config: dict, results: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?)",
                         (experiment_id, created_at, filename, json.dumps(config, default=str),
                          json.dumps(results, default=str)))

    def list_experiments(self) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
            rows = []
            for row in cursor.fetchall():
                item = dict(row)
                item["config"] = json.loads(item.pop("config_json") or "{}")
                item["results"] = json.loads(item.pop("results_json") or "[]")
                rows.append(item)
            return rows
