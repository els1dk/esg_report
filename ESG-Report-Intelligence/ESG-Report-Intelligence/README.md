# 🌿 ESG Report Intelligence

A standalone generative-AI application that turns a sustainability / ESG report (PDF) into **structured,
verified ESG data**, with every number traceable to its page, and an analyst-style briefing.

It is *not* a chat-with-your-PDF tool. It runs a multi-step pipeline:

```
PDF ─► parse text, tables, charts ─► profile report (LLM) ─► BM25 page retrieval
    ─► multimodal metric extraction (LLM, few-shot + chain-of-thought) ─► JSON self-repair
    ─► rule-based checks against the PDF ─► AI auditor verification (LLM, optional 2nd model)
    ─► ESG insight summary (LLM) ─► SQLite history + Excel/CSV/JSON/Markdown export
```

The metric catalogue uses the 17 **ConsciESG metric IDs** (Scope 1/2/3, waste, water, parental leave,
gender diversity, pay ratios, supplier labour violations) from the original ESG extraction project, plus
4 optional extended metrics.

---

## Quick start (Windows)

1. Get a **free** Gemini API key: <https://aistudio.google.com/apikey>
   (optional: a free Groq key at <https://console.groq.com/keys>).
2. Double-click **`run_app.bat`**. On the first run it creates a virtual environment, installs the
   packages and opens `.env` in Notepad: paste your key after `GEMINI_API_KEY=` and save.
3. The app opens in your browser at <http://localhost:8501>.
   Click **Load sample report** → **Analyze report** to try it.

### Manual start (any OS)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env  → add your key
streamlit run app.py
```

### Tests (no API key needed)

```bash
python -m pytest -q             # 57 tests, uses the offline test provider
```

To click through the UI without an API key, set `ESG_ENABLE_MOCK=1` in `.env`: the sidebar then offers an
*offline test provider* that answers with simple rules (for testing only, not real AI output).

---

## Free APIs used

| Provider | Default model | Why |
|---|---|---|
| Google Gemini (free tier) | `gemini-3.5-flash-lite` | multimodal (reads chart images), largest free quota |
| Google Gemini (free tier) | `gemini-3.6-flash`, `gemini-3.8-flash` | higher quality, lower daily quota |
| Groq (free tier) | `qwen/qwen3.8-27b` (vision), `openai/gpt-oss-120b` | very fast; good as a second, independent verifier |

Model IDs change often – the sidebar also accepts a **custom model ID**. One analysis uses 4 LLM calls
(profile, extraction, verification, summary), so the free tiers are sufficient.

---

## Features

| Tab | What you see |
|---|---|
| 🚀 Analyze | upload, live progress log (including retries), KPIs |
| 📊 ESG Metrics | status per metric (VALID / UNCERTAIN / INVALID / NOT FOUND), disclosure completeness per E/S/G pillar, time series, exports |
| 🔎 Evidence | *"Where did this number come from?"* – value, quote, reasoning, rule checks, verifier verdict and the **PDF page rendered with the value highlighted** |
| 📝 Summary | executive summary, E/S/G paragraphs, strengths, gaps, red flags |
| 📄 Report & Pipeline | report profile, page statistics, why each page was retrieved, every LLM call with its parameters, tokens and latency |
| 🧪 Prompt Lab | compare prompt strategies × temperatures × repeats; grounded %, evidence %, JSON validity, stability, agreement, accuracy vs. ground truth; **exports a LaTeX table** for the report |
| 🗂️ History | all analyses stored in SQLite; reopen or delete; token usage per model |

Tip: in the Prompt Lab, upload `samples/northwind_ground_truth.csv` as ground truth when you test the sample
report – then the lab also reports **accuracy** (15 known values, incl. one that appears only in a chart).

## Prompt engineering

* **Role prompting** – ESG data specialist (extraction), ISAE 3000 auditor (verification), sustainability analyst (summary).
* **Zero-shot** – report profiling and the Prompt Lab baseline.
* **Few-shot** – four worked examples: value in text, multi-year table with scale in header, a *target that must not be extracted*, and a value read from a chart.
* **Chain-of-thought decomposition** – six explicit steps per metric (locate → filter → period → value/unit → scope → decide); the model returns a one-sentence justification instead of its full reasoning.
* **Structured output** – JSON mode + Pydantic schemas.
* **Iterative self-repair** – invalid JSON is fed back with the parser error.
* **Self-verification / cross-model verification** – a second prompt (optionally a second model) audits every value against text taken directly from the PDF.

## Parameters

Each LLM stage has its own `temperature`, `top_p`, `max_output_tokens` and reasoning level (Gemini
`thinking_level` / Groq `reasoning_effort`), editable in the sidebar:

| Stage | temperature | top_p | max tokens | rationale |
|---|---|---|---|---|
| Profile | 0.0 | 0.9 | 1,024 | deterministic metadata |
| Extraction | 0.1 | 0.9 | 8,192 | reproducible numbers, room for 17–21 JSON items |
| Verification | 0.0 | 0.9 | 4,096 | strict, repeatable judgement |
| Summary | 0.6 | 0.95 | 2,048 | fluent, less repetitive prose |

## Error handling

Invalid / empty / encrypted / oversized PDFs · unreadable pages (skipped, run continues) · scanned pages (sent
as images) · missing or invalid API key · rate limits (exponential backoff, honours `retry-after`) · daily
quota exhausted · timeouts and 5xx (retry) · prompt too large (context automatically halved) · unsupported
parameters (automatic retry without them) · malformed / truncated JSON (local repair → model self-repair) ·
degenerate repetition at low temperature (retry at T = 1.0) · profiling, verification or summary failure
(graceful degradation, result marked *partial*) · any unexpected UI error is contained to one tab.

## Project structure

```
app.py                     Streamlit UI
esg_intel/
  config.py                providers, models, parameters, metric catalogue
  pdf_processor.py         PyMuPDF parsing, tables, chart detection, rendering
  retrieval.py             BM25 retrieval + context builder
  prompts.py               all prompts and prompt strategies
  llm_client.py            Gemini + Groq clients, error classes, retries
  schemas.py               Pydantic output schemas
  json_utils.py            robust JSON parsing
  validation.py            rule checks, unit normalisation, status fusion
  pipeline.py              end-to-end pipeline
  experiments.py           Prompt Lab
  storage.py               SQLite persistence
  export.py                Excel / CSV / JSON / Markdown
  mock_provider.py         offline test double
samples/                   fictional sample report + generator script
tests/                     pytest suite (57 tests)
docs/                      LaTeX documentation (Overleaf-ready)
```

## Put it on GitHub (optional for individual work, a plus for grading)

```bash
git init
git add .
git commit -m "ESG Report Intelligence – initial version"
git branch -M main
git remote add origin https://github.com/<your-user>/esg-report-intelligence.git
git push -u origin main
```

`.env` (your key) and `data/` (database, uploaded PDFs) are excluded by `.gitignore`.

---

*Sample report: "Northwind Renewables AG" is fictional; all of its figures are invented for testing.*
