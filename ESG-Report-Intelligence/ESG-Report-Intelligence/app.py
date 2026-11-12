"""
ESG Report Intelligence – Streamlit front end.

Run with:   streamlit run app.py

The UI only collects settings, starts the pipeline and displays results;
all processing lives in the esg_intel package.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from esg_intel import export, prompts
from esg_intel.config import (APP_NAME, APP_VERSION, BASE_DIR, CORE_METRIC_IDS, DEFAULT_STAGE_PARAMS, METRICS,
                              METRICS_BY_ID, PILLAR_NAMES, PROVIDERS, SAMPLES_DIR, STAGE_LABELS, StageParams,
                              find_model)
from esg_intel.experiments import (LabRun, ground_truth_for, load_ground_truth, new_experiment_record,
                                   run_prompt_lab, summarise_runs, to_latex_table)
from esg_intel.llm_client import LLMError
from esg_intel.pdf_processor import PDFValidationError, file_sha256, number_variants, render_page_with_highlights
from esg_intel.pipeline import ESGPipeline, PipelineConfig, PipelineError
from esg_intel.schemas import ReportProfile
from esg_intel.storage import Storage

load_dotenv(BASE_DIR / ".env")
st.set_page_config(page_title=APP_NAME, page_icon="🌿", layout="wide")

STATUS_ICON = {"VALID": "✅ VALID", "UNCERTAIN": "⚠️ UNCERTAIN", "INVALID": "❌ INVALID", "NOT FOUND": "⬜ NOT FOUND"}
STATUS_COLOR = {"VALID": "#2E8B57", "UNCERTAIN": "#E0A526", "INVALID": "#C8423B", "NOT FOUND": "#B8BDB9"}
SAMPLE_PDF = SAMPLES_DIR / "Northwind_Renewables_Sustainability_Report_2024.pdf"
TAB_NAMES = ["🚀 Analyze", "📊 ESG Metrics", "🔎 Evidence", "📝 Summary", "📄 Report & Pipeline",
             "🧪 Prompt Lab", "🗂️ History", "ℹ️ About"]
METRICS_TAB = TAB_NAMES[1]

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
         background: linear-gradient(120deg, rgba(31,111,80,.12), rgba(58,141,191,.10));
         border: 1px solid rgba(31,111,80,.25);}
  .hero h1 {margin: 0 0 .2rem 0; font-size: 2rem;}
  .hero p {margin: 0; opacity: .85;}
  .valuecard {padding: 1rem 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.3);}
  .valuecard .big {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
  .valuecard .label {opacity: .7; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em;}
  .quote {border-left: 4px solid #1F6F50; padding: .4rem .9rem; margin: .4rem 0;
          background: rgba(31,111,80,.07); border-radius: 0 8px 8px 0; font-style: italic;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_storage() -> Storage:
    return Storage()


storage = get_storage()
state = st.session_state
state.setdefault("result", None)
state.setdefault("pdf_bytes", None)
state.setdefault("pdf_name", None)
state.setdefault("lab", None)
state.setdefault("last_context", None)


# ======================================================================
# Sidebar – provider, prompt strategy and parameters
# ======================================================================
def api_key_input(provider_key: str, widget_key: str) -> str:
    provider = PROVIDERS[provider_key]
    if not provider.env_var:
        return ""
    env_value = os.getenv(provider.env_var, "").strip()
    typed = st.text_input(f"{provider.env_var}", type="password", key=widget_key,
                          placeholder="loaded from .env" if env_value else "paste your free API key",
                          help=f"Get a free key at {provider.key_url}. Stored only in this session.")
    if env_value and not typed:
        st.caption("✓ Key loaded from `.env`")
    elif not env_value and not typed:
        st.caption(f"No key yet → [get a free key]({provider.key_url})")
    return typed.strip() or env_value


def model_select(provider_key: str, widget_key: str, label: str = "Model") -> str:
    provider = PROVIDERS[provider_key]
    options = [m.model_id for m in provider.models] + ["custom"]
    labels = {m.model_id: m.label for m in provider.models} | {"custom": "Custom model ID…"}
    choice = st.selectbox(label, options, format_func=lambda m: labels[m], key=widget_key)
    if choice == "custom":
        return st.text_input("Custom model ID", key=f"{widget_key}_custom",
                             placeholder="e.g. gemini-3.6-flash").strip() or provider.models[0].model_id
    return choice


def stage_param_inputs(stage: str, provider_key: str) -> StageParams:
    default = DEFAULT_STAGE_PARAMS[stage]
    levels = PROVIDERS[provider_key].reasoning_levels or ("low",)
    temperature = st.slider("temperature", 0.0, 1.5, default.temperature, 0.05, key=f"{stage}_t",
                            help="Randomness of token sampling. Low = deterministic facts, higher = varied prose.")
    top_p = st.slider("top_p", 0.1, 1.0, default.top_p, 0.05, key=f"{stage}_p",
                      help="Nucleus sampling: only tokens within this cumulative probability are considered.")
    max_tokens = st.number_input("max_output_tokens", 256, 32768, default.max_output_tokens, 256,
                                 key=f"{stage}_m", help="Upper bound on the answer length (incl. thinking).")
    reasoning_default = default.reasoning if default.reasoning in levels else levels[0]
    reasoning = st.selectbox("reasoning / thinking level", levels, index=levels.index(reasoning_default),
                             key=f"{stage}_r_{provider_key}",
                             help="Gemini thinking_level / Groq reasoning_effort: how much the model "
                                  "reasons before answering.")
    return StageParams(temperature=temperature, top_p=top_p, max_output_tokens=int(max_tokens), reasoning=reasoning)


def sidebar() -> PipelineConfig:
    sb = st.sidebar
    sb.markdown(f"### 🌿 {APP_NAME}")
    sb.caption(f"v{APP_VERSION} · multimodal ESG report analysis")

    providers = ["gemini", "groq"] + (["mock"] if os.getenv("ESG_ENABLE_MOCK") == "1" else [])
    with sb.container(border=True):
        st.markdown("**🤖 AI provider**")
        provider = st.selectbox("Provider", providers, format_func=lambda k: PROVIDERS[k].label, key="provider")
        model = model_select(provider, "model")
        api_key = api_key_input(provider, "api_key")
        info = find_model(provider, model)
        st.caption(("🖼️ vision model – charts are read from page images" if info.vision
                    else "📝 text-only model – charts are not analysed") + (f" · {info.note}" if info.note else ""))

    with sb.container(border=True):
        st.markdown("**🧠 Prompt strategy**")
        strategy = st.selectbox("Extraction prompt", list(prompts.STRATEGIES),
                                index=list(prompts.STRATEGIES).index(prompts.DEFAULT_STRATEGY),
                                format_func=lambda k: prompts.STRATEGIES[k].label, key="strategy")
        st.caption(prompts.STRATEGIES[strategy].description)

    with sb.expander("🎛️ Generation parameters (per stage)"):
        st.caption("Each LLM stage has its own sampling parameters – factual stages use low temperature, "
                   "the summary a higher one.")
        params = {}
        for stage, tab in zip(DEFAULT_STAGE_PARAMS, st.tabs(["Profile", "Extract", "Verify", "Summary"])):
            with tab:
                params[stage] = stage_param_inputs(stage, provider)

    with sb.expander("📄 Document processing"):
        max_pages = st.slider("Pages sent to the model", 4, 30, 12,
                              help="Top pages chosen by BM25 retrieval. Fewer pages = smaller, cheaper prompt.")
        use_images = st.toggle("Send chart pages as images (multimodal)", value=info.vision, disabled=not info.vision)
        max_images = st.slider("Max page images", 0, 6, min(3, info.max_images or 0), disabled=not info.vision)
        context_chars = st.slider("Context budget (characters)", 6_000, 150_000, info.context_chars, 2_000,
                                  help="Maximum text from retrieved pages. Groq's free tier needs ≤16k.")

    with sb.expander("🛡️ Verification & summary"):
        verify = st.toggle("AI verification (auditor prompt)", value=True)
        cross = st.toggle("Use a second model as verifier", value=False, disabled=not verify,
                          help="Cross-model verification: a different model checks the extractor's work.")
        verifier_provider = verifier_model = None
        verifier_key = ""
        if verify and cross:
            verifier_provider = st.selectbox("Verifier provider", providers, key="v_provider",
                                             index=1 if provider == "gemini" and len(providers) > 1 else 0,
                                             format_func=lambda k: PROVIDERS[k].label)
            verifier_model = model_select(verifier_provider, "v_model", "Verifier model")
            verifier_key = api_key_input(verifier_provider, "v_api_key")
        summarize = st.toggle("Generate insight summary", value=True)

    with sb.expander("📋 Metrics to extract"):
        preset = st.segmented_control("Preset", ["Core 17 (ConsciESG)", "All 21"], default="Core 17 (ConsciESG)",
                                      key="metric_preset")
        default_ids = [m.metric_id for m in METRICS] if preset == "All 21" else CORE_METRIC_IDS
        metric_ids = st.multiselect("Metrics", [m.metric_id for m in METRICS], default=default_ids,
                                    format_func=lambda i: f"{METRICS_BY_ID[i].pillar} · {METRICS_BY_ID[i].name}",
                                    key=f"metrics_{preset}")

    with sb.expander("⚙️ Reliability"):
        max_attempts = st.slider("Max attempts per API call", 1, 6, 4,
                                 help="Retries with exponential backoff for rate limits, timeouts and 5xx errors.")
        timeout_s = st.slider("Timeout per call (s)", 30, 600, 180, 10)

    return PipelineConfig(
        provider=provider, model=model, api_key=api_key, strategy=strategy,
        metric_ids=metric_ids or CORE_METRIC_IDS, params=params, max_pages_to_model=max_pages,
        use_images=use_images, max_images=max_images, context_chars=context_chars, verify=verify,
        verifier_provider=verifier_provider, verifier_model=verifier_model, verifier_api_key=verifier_key,
        summarize=summarize, max_attempts=max_attempts, timeout_s=float(timeout_s))


# ======================================================================
# Helpers
# ======================================================================
def fmt_value(value, unit="") -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{value:,.0f}" if float(value).is_integer() or abs(value) >= 1000 else f"{value:,.4g}"
    return f"{text} {unit}".strip()


def require_result() -> dict | None:
    if state.result is None:
        st.info("No analysis loaded yet. Upload a report in **🚀 Analyze** or open one from **🗂️ History**.")
        return None
    return state.result


def missing_key(cfg: PipelineConfig) -> str | None:
    if cfg.provider != "mock" and not cfg.api_key:
        provider = PROVIDERS[cfg.provider]
        return (f"🔑 No API key for {provider.label}. Create a free key at {provider.key_url}, then put "
                f"`{provider.env_var}=...` in the `.env` file (or paste it in the sidebar).")
    if cfg.verify and cfg.verifier_provider and cfg.verifier_provider != "mock" and not cfg.verifier_api_key:
        return f"🔑 No API key for the verifier provider ({PROVIDERS[cfg.verifier_provider].label})."
    return None


def run_with_ui(cfg: PipelineConfig, pdf_bytes: bytes, filename: str):
    """Run the pipeline with a live progress bar and log; translate every error into a clear message."""
    progress = st.progress(0.0, text="Starting …")
    with st.status("Analysing report …", expanded=True) as status:
        def on_progress(fraction: float, message: str):
            progress.progress(min(1.0, fraction), text=message)
            status.update(label=message)
            st.write(message)

        def on_log(message: str):
            st.write(f"↻ {message}")

        try:
            pipeline = ESGPipeline(cfg, progress=on_progress, log=on_log)
            result = pipeline.run(pdf_bytes, filename).to_dict()
        except PDFValidationError as error:
            status.update(label="Invalid PDF", state="error")
            st.error(f"❌ {error}")
            return None
        except PipelineError as error:
            status.update(label="Analysis stopped", state="error")
            st.error(error.user_message)
            if error.detail:
                with st.expander("Technical details"):
                    st.code(error.detail)
            return None
        except LLMError as error:
            status.update(label="AI service error", state="error")
            st.error(error.user_message)
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")
            return None
        except Exception as error:  # last line of defence: never crash the app
            status.update(label="Unexpected error", state="error")
            st.error("Something unexpected went wrong. The app is still running – adjust the settings and retry.")
            with st.expander("Technical details"):
                st.exception(error)
            return None
        label = "Analysis complete" if result["status"] == "complete" else "Analysis complete (with warnings)"
        status.update(label=f"✅ {label} in {result['duration_s']} s", state="complete", expanded=False)
    storage.save_pdf(result["file_hash"], pdf_bytes)
    storage.save_analysis(result)
    return result


def kpi_row(result: dict):
    rows = result["rows"]
    count = lambda s: sum(1 for r in rows if r["status"] == s)  # noqa: E731
    tokens = sum(c["input_tokens"] + c["output_tokens"] for c in result["calls"])
    cols = st.columns(7)
    cols[0].metric("Pages", result["page_count"], border=True)
    cols[1].metric("✅ Valid", count("VALID"), border=True)
    cols[2].metric("⚠️ Uncertain", count("UNCERTAIN"), border=True)
    cols[3].metric("❌ Invalid", count("INVALID"), border=True)
    cols[4].metric("⬜ Not found", count("NOT FOUND"), border=True)
    cols[5].metric("LLM calls", len(result["calls"]), border=True)
    cols[6].metric("Tokens", f"{tokens:,}", border=True)


# ======================================================================
# Tabs
# ======================================================================
def tab_analyze(cfg: PipelineConfig):
    st.markdown("""<div class="hero"><h1>🌿 ESG Report Intelligence</h1>
    <p>Upload a sustainability, integrated or annual report. The app reads its text, tables and charts,
    extracts the ConsciESG metrics with generative AI, checks every number against the PDF and writes an
    analyst briefing – each figure traceable to its page.</p></div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        upload = st.file_uploader("Sustainability report (PDF)", type=["pdf"],
                                  help="Text-based PDFs work best. Scanned pages are read as images.")
    with right:
        st.markdown("**No report at hand?**")
        st.caption("Try the bundled *fictional* sample report (10 pages, values in text, tables and charts).")
        if st.button("📘 Load sample report", disabled=not SAMPLE_PDF.exists()):
            state.pdf_bytes, state.pdf_name = SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name
    if upload is not None:
        data = upload.getvalue()
        if data != state.pdf_bytes:
            state.pdf_bytes, state.pdf_name = data, upload.name

    if state.pdf_bytes:
        size = len(state.pdf_bytes) / 1e6
        st.markdown(f"**Selected:** `{state.pdf_name}` · {size:.1f} MB")
        previous = storage.find_by_hash(file_sha256(state.pdf_bytes))
        if previous:
            last = previous[0]
            c1, c2 = st.columns([4, 1])
            c1.info(f"This report was analysed before ({last['created_at']}, {last['model']}, "
                    f"{prompts.STRATEGIES.get(last['strategy'], prompts.STRATEGIES[prompts.DEFAULT_STRATEGY]).label}). "
                    "Load it to save free-tier quota, or run a new analysis.")
            if c2.button("Load previous", key="load_prev"):
                state.result = storage.load_analysis(last["analysis_id"])
                state["_notice"] = f"Loaded the previous analysis of {state.pdf_name}"
                state["_goto_tab"] = METRICS_TAB
                st.rerun()

        problem = missing_key(cfg)
        if problem:
            st.warning(problem)
        est_calls = 2 + int(cfg.verify) + int(cfg.summarize)
        st.caption(f"Plan: {est_calls} LLM calls with **{cfg.model}** · strategy "
                   f"*{prompts.STRATEGIES[cfg.strategy].label}* · {len(cfg.metric_ids)} metrics · "
                   f"up to {cfg.max_pages_to_model} pages and {cfg.max_images if cfg.use_images else 0} page images.")
        if st.button("🚀 Analyze report", type="primary", disabled=bool(problem)):
            result = run_with_ui(cfg, state.pdf_bytes, state.pdf_name)
            if result:
                state.result = result
                st.toast("Analysis finished – see the ESG Metrics tab.", icon="✅")

    result = state.result
    if result:
        st.divider()
        st.subheader(f"{result['profile'].get('company_name')} · {result['profile'].get('reporting_year') or ''}")
        kpi_row(result)
        if result["warnings"]:
            with st.expander(f"⚠️ {len(result['warnings'])} processing note(s)"):
                for warning in result["warnings"]:
                    st.write("•", warning)
        st.caption("Open **📊 ESG Metrics**, **🔎 Evidence** and **📝 Summary** for the results.")


def tab_metrics():
    result = require_result()
    if not result:
        return
    rows = result["rows"]
    st.subheader(f"ESG metrics – {result['profile'].get('company_name')}")

    comp = result["completeness"]
    cols = st.columns(4)
    for col, key in zip(cols, ["E", "S", "G", "ALL"]):
        if key in comp:
            item = comp[key]
            col.metric(PILLAR_NAMES.get(key, "All pillars"), f"{item['share']:.0%}",
                       f"{item['disclosed']}/{item['requested']} disclosed", delta_color="off",
                       delta_arrow="off", border=True)

    frame = pd.DataFrame(rows)
    counts = (frame.groupby(["pillar_name", "status"]).size().reset_index(name="metrics"))
    chart = alt.Chart(counts).mark_bar().encode(
        y=alt.Y("pillar_name:N", title=None, sort=["Environmental", "Social", "Governance"]),
        x=alt.X("metrics:Q", title="number of metrics", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLOR), range=list(STATUS_COLOR.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        order=alt.Order("status:N"),
        # Step-based height: the chart is drawn while its tab may be hidden, so it must not depend
        # on the container's size.
        tooltip=["pillar_name", "status", "metrics"]).properties(height=alt.Step(34))
    st.altair_chart(chart, width="stretch")

    f1, f2 = st.columns(2)
    pillars = f1.pills("Pillar", ["Environmental", "Social", "Governance"], selection_mode="multi",
                       default=["Environmental", "Social", "Governance"], key="pill_filter")
    statuses = f2.pills("Status", list(STATUS_ICON), selection_mode="multi", default=list(STATUS_ICON),
                        key="status_filter")
    view = frame[frame["pillar_name"].isin(pillars or []) & frame["status"].isin(statuses or [])].copy()
    view["Status"] = view["status"].map(STATUS_ICON)
    view["Value"] = [fmt_value(v, u) for v, u in zip(view["value"], view["unit"])]
    view["Standardised"] = [fmt_value(v, u) for v, u in zip(view["normalized_value"], view["standard_unit"])]
    for column in ("year", "page"):
        view[column] = [("" if pd.isna(v) else str(int(v))) for v in pd.to_numeric(view[column], errors="coerce")]
    view["source_type"] = view["source_type"].replace("none", "")
    st.dataframe(
        view[["metric_id", "metric", "pillar_name", "Status", "Value", "Standardised", "year", "page",
              "source_type", "verifier_verdict", "final_confidence"]],
        hide_index=True, width="stretch",
        column_config={
            "metric_id": st.column_config.NumberColumn("ID", width="small"),
            "metric": st.column_config.TextColumn("Metric", width="large"),
            "pillar_name": "Pillar", "year": st.column_config.TextColumn("Year", width="small"),
            "page": st.column_config.TextColumn("Page", width="small"),
            "source_type": "Source", "verifier_verdict": "AI verifier",
            "final_confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                                format="%.2f"),
        })

    history = export.history_frame(result)
    if not history.empty and history.groupby("metric")["year"].nunique().max() > 1:
        st.markdown("**📈 Time series reported in the document**")
        multi = [m for m, n in history.groupby("metric")["year"].nunique().items() if n > 1]
        metric = st.selectbox("Metric", multi, key="trend_metric")
        sub = history[history["metric"] == metric]
        line = alt.Chart(sub).mark_line(point=True, color="#1F6F50").encode(
            x=alt.X("year:O", title="Year"), y=alt.Y("value:Q", title=sub["unit"].iloc[0] or "value"),
            tooltip=["year", "value", "unit"]).properties(height=260, width=alt.Step(90))
        st.altair_chart(line, width="stretch")

    st.markdown("**⬇️ Export**")
    base = (result["profile"].get("company_name") or "esg").replace(" ", "_")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Excel workbook", export.to_excel(result), f"{base}_esg.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    c2.download_button("CSV", export.to_csv(result), f"{base}_esg.csv", "text/csv", width="stretch")
    c3.download_button("JSON (full result)", export.to_json(result), f"{base}_esg.json", "application/json",
                       width="stretch")
    c4.download_button("Markdown briefing", export.to_markdown(result).encode("utf-8"), f"{base}_briefing.md",
                       "text/markdown", width="stretch")


def tab_evidence():
    result = require_result()
    if not result:
        return
    st.subheader("🔎 Where did this number come from?")
    rows = sorted(result["rows"], key=lambda r: r["status"] == "NOT FOUND")
    labels = {r["metric_id"]: f"{STATUS_ICON[r['status']]} · {r['metric']}" for r in rows}
    metric_id = st.selectbox("Metric", list(labels), format_func=lambda i: labels[i], key="evidence_metric")
    row = next(r for r in rows if r["metric_id"] == metric_id)

    if row["status"] == "NOT FOUND":
        st.info(f"**{row['metric']}** was not found in the retrieved pages. The model was instructed to return "
                f"*not_found* rather than guess. Reasoning: {row['reasoning'] or '—'}")
        return

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"""<div class="valuecard"><div class="label">{row['metric']}</div>
        <div class="big">{fmt_value(row['value'], row['unit'])}</div>
        <div>{row['year'] or 'year n/a'} · page {row['page']} · source: {row['source_type']}
        {(' · ' + row['scope2_method']) if row['scope2_method'] else ''}</div>
        <div style="margin-top:.4rem">Standardised: <b>{fmt_value(row['normalized_value'], row['standard_unit'])}</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"**Status:** {STATUS_ICON[row['status']]} · final confidence **{row['final_confidence']:.2f}**")
        st.markdown("**Evidence quoted by the extractor**")
        st.markdown(f"<div class='quote'>“{row['evidence'] or '—'}”</div>", unsafe_allow_html=True)
        st.markdown(f"**Extractor reasoning:** {row['reasoning'] or '—'} "
                    f"(self-reported confidence {row['extractor_confidence']:.2f})")

        st.markdown("**Rule-based checks against the PDF**")
        symbol = {True: "✅", False: "❌", None: "–"}
        checks = pd.DataFrame([{"check": k.replace("_", " "), "result": symbol[v]}
                               for k, v in row["rule_checks"].items()])
        st.dataframe(checks, hide_index=True, width="stretch")

        st.markdown(f"**AI verifier:** {row['verifier_verdict']}")
        if row["verifier_reason"]:
            st.caption(row["verifier_reason"])
        if row["verifier_checks"]:
            st.caption(" · ".join(f"{symbol[v]} {k.replace('_', ' ')}" for k, v in row["verifier_checks"].items()))
        for note in row["notes"]:
            st.write("•", note)
        if row["history"]:
            st.markdown("**Earlier years in the same report:** " +
                        ", ".join(f"{h['year']}: {fmt_value(h['value'], row['unit'])}" for h in row["history"]))

    with right:
        pdf = storage.load_pdf(result["file_hash"])
        if pdf is None and state.pdf_bytes and file_sha256(state.pdf_bytes) == result["file_hash"]:
            pdf = state.pdf_bytes
        if pdf and row["page"] and 1 <= row["page"] <= result["page_count"]:
            terms = number_variants(row["value"]) if row["value"] is not None else []
            if row["evidence"] and row["source_type"] in ("text", "table"):
                terms.append(row["evidence"][:60])
            try:
                png, hits = render_page_with_highlights(pdf, row["page"], terms)
                st.image(png, caption=f"Page {row['page']} – {hits} highlight(s) of the value/evidence",
                         width="stretch")
            except Exception as error:
                st.warning(f"Could not render page {row['page']}: {error}")
        else:
            st.info("The source PDF is not available to display the page.")


def tab_summary():
    result = require_result()
    if not result:
        return
    summary = result.get("summary")
    profile = result["profile"]
    st.subheader(f"📝 ESG briefing – {profile.get('company_name')} {profile.get('reporting_year') or ''}")
    if not summary:
        st.info("No summary was generated (switched off, or the summary step failed – see processing notes).")
        return
    st.markdown(summary["executive_summary"])
    cols = st.columns(3)
    for col, key, title in zip(cols, ["environmental", "social", "governance"],
                               ["🌍 Environmental", "🤝 Social", "🏛️ Governance"]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(summary.get(key) or "—")
    cols = st.columns(4)
    for col, key, title in zip(cols, ["strengths", "disclosure_gaps", "red_flags", "recommended_follow_up"],
                               ["💪 Strengths", "🕳️ Disclosure gaps", "🚩 Red flags", "➡️ Follow-up"]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            for item in summary.get(key) or ["none"]:
                st.write("•", item)
    st.caption("Generated by the analyst prompt from verified data only (temperature "
               f"{result['config']['params']['summary']['temperature']}).")


def tab_pipeline():
    result = require_result()
    if not result:
        return
    profile = result["profile"]
    st.subheader("📄 Report profile")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Company**  \n{profile.get('company_name')}")
        c2.markdown(f"**Reporting year**  \n{profile.get('reporting_year') or 'Unknown'}")
        c3.markdown(f"**Report type**  \n{profile.get('report_type')}")
        c4.markdown(f"**Sector / country**  \n{profile.get('sector')} · {profile.get('country')}")
        st.markdown("**Frameworks:** " + (" ".join(f"`{f}`" for f in profile.get("frameworks") or []) or "none stated")
                    + f" · **Assurance:** {profile.get('assurance')}")
        if profile.get("one_line_description"):
            st.caption(profile["one_line_description"])

    st.subheader("🧩 Multimodal document processing")
    stats = result["page_stats"]
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Pages", "pages"), ("Text pages", "text_pages"),
                                        ("Scanned pages", "scanned_pages"), ("Pages sent", "selected_pages"),
                                        ("Tables extracted", "tables_found"), ("Page images sent", "chart_pages_sent")]):
        col.metric(label, stats.get(key, 0), border=True)
    st.markdown("**Retrieved pages (BM25 ranking) and why they were chosen**")
    st.dataframe(pd.DataFrame(result["retrieval"]), hide_index=True, width="stretch",
                 column_config={"score": st.column_config.NumberColumn("BM25 score", format="%.1f"),
                                "metrics": st.column_config.ListColumn("Matched metrics", width="large"),
                                "image_sent": st.column_config.CheckboxColumn("Image sent")})

    st.subheader("🔗 LLM calls (parameters, tokens, latency)")
    calls = pd.DataFrame(result["calls"])
    if not calls.empty:
        calls["stage"] = calls["stage"].map(lambda s: STAGE_LABELS.get(s, s))
        st.dataframe(calls[["stage", "model", "temperature", "top_p", "max_output_tokens", "reasoning", "n_images",
                            "input_tokens", "output_tokens", "latency_s", "attempts", "status", "error"]],
                     hide_index=True, width="stretch")
    if result["warnings"]:
        st.subheader("⚠️ Processing notes")
        for warning in result["warnings"]:
            st.write("•", warning)
    with st.expander("Settings used for this analysis"):
        st.json(result["config"])


def tab_lab(cfg: PipelineConfig):
    st.subheader("🧪 Prompt Lab – compare prompt techniques and parameters")
    st.markdown("The report is parsed and retrieved **once**; only the extraction prompt changes between runs. "
                "Each run is scored automatically: *grounded* = the number really appears on the cited page, "
                "*evidence* = the quote is on the page, *stability* = identical values across repeats, "
                "*agreement* = same values as the reference configuration (the last one).")
    if not state.pdf_bytes:
        st.info("Select a report in **🚀 Analyze** first (or load the sample).")
    c1, c2, c3 = st.columns([3, 2, 1])
    strategies = c1.multiselect("Prompt strategies", list(prompts.STRATEGIES), default=list(prompts.STRATEGIES),
                                format_func=lambda k: prompts.STRATEGIES[k].label)
    temperatures = c2.multiselect("Temperatures", [0.0, 0.2, 0.5, 0.7, 1.0, 1.3], default=[0.0, 1.0])
    repeats = c3.number_input("Repeats", 1, 5, 1, help="≥2 repeats measure stability (run-to-run variation).")
    n_runs = len(strategies) * len(temperatures) * int(repeats)
    st.caption(f"{n_runs} extraction calls + 1 profiling call with **{cfg.model}**. Free tiers allow a limited "
               "number of requests per minute/day – the lab pauses 3 s between runs.")

    with st.expander("Optional: ground truth for accuracy (e.g. the validated ESG database)"):
        gt_file = st.file_uploader("CSV or Excel with metric id, year and value columns", type=["csv", "xlsx"],
                                   key="gt_file")
        ground_truth = None
        if gt_file is not None:
            try:
                frame = load_ground_truth(gt_file.getvalue(), gt_file.name)
                companies = sorted(frame["company"].astype(str).unique()) if "company" in frame.columns else []
                company = st.selectbox("Company", companies) if companies else None
                year = st.number_input("Year", 1990, 2100, int((state.result or {}).get("profile", {})
                                                               .get("reporting_year") or 2024))
                ground_truth = ground_truth_for(frame, int(year), company, cfg.metric_ids)
                st.caption(f"{len(ground_truth)} reference value(s) for this company and year.")
            except Exception as error:
                st.error(f"Could not read the ground-truth file: {error}")

    problem = missing_key(cfg)
    if problem:
        st.warning(problem)
    if st.button("▶ Run experiments", type="primary",
                 disabled=not (state.pdf_bytes and strategies and temperatures) or bool(problem)):
        bar = st.progress(0.0, text="Preparing document …")
        try:
            pipeline = ESGPipeline(cfg, progress=lambda f, m: bar.progress(min(1.0, f * 0.2), text=m))
            prepared = pipeline.prepare(state.pdf_bytes, state.pdf_name)
            state.last_context = prepared.context
            runs = run_prompt_lab(pipeline, prepared, strategies, temperatures, int(repeats),
                                  ground_truth=ground_truth,
                                  progress=lambda f, m: bar.progress(0.2 + 0.8 * f, text=m))
            record = new_experiment_record(state.pdf_name, strategies, temperatures, int(repeats), cfg.model, runs)
            storage.save_experiment(record["experiment_id"], record["created_at"], state.pdf_name,
                                    record["config"], record["results"])
            state.lab = {"runs": [asdict(r) for r in runs], "file": state.pdf_name, "model": cfg.model}
        except PDFValidationError as error:
            st.error(f"❌ {error}")
        except LLMError as error:
            st.error(error.user_message)
            st.caption(f"{type(error).__name__}: {error}")
        except Exception as error:
            st.error("The experiment stopped unexpectedly.")
            st.exception(error)

    if state.lab:
        runs =[LabRun(**r) for r in state.lab["runs"]]
        summary = summarise_runs(runs)
        st.markdown(f"**Results** – {state.lab['file']} · {state.lab['model']}")
        show = summary.drop(columns=["Strategy"]).copy()
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="percent")
            for c in ["Grounded %", "Evidence %", "JSON first try %", "Accuracy %", "Stability %", "Agreement %"]})
        long = summary.melt(id_vars=["Configuration"], value_vars=["Grounded %", "Evidence %", "Agreement %"]
                            + (["Accuracy %"] if summary["Accuracy %"].notna().any() else []),
                            var_name="Metric", value_name="Score").dropna()
        if not long.empty:
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                x=alt.X("Score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Configuration:N", title=None, axis=alt.Axis(labelLimit=320)),
                color="Metric:N", yOffset="Metric:N",
                tooltip=["Configuration", "Metric", alt.Tooltip("Score:Q", format=".0%")]
            ).properties(height=max(160, 46 * len(summary))), width="stretch")
        failed = [r for r in runs if not r.ok]
        for run in failed:
            st.warning(f"{run.config_label} (repeat {run.repeat}) failed: {run.error}")
        latex = to_latex_table(summary, f"Prompt Lab results on {state.lab['file'].replace('_', ' ')} "
                                        f"({state.lab['model']}).")
        st.markdown("**LaTeX table for the report** (save as `results/prompt_lab_results.tex` in Overleaf)")
        st.code(latex, language="latex")
        st.download_button("⬇️ prompt_lab_results.tex", latex.encode("utf-8"), "prompt_lab_results.tex", "text/plain")
        st.download_button("⬇️ Raw runs (CSV)", pd.DataFrame([asdict(r) for r in runs]).to_csv(index=False),
                           "prompt_lab_runs.csv", "text/csv")

    with st.expander("🔍 Prompt inspector – the exact prompts sent to the model"):
        strategy_key = st.selectbox("Strategy", list(prompts.STRATEGIES), key="inspect_strategy",
                                    index=list(prompts.STRATEGIES).index(prompts.DEFAULT_STRATEGY),
                                    format_func=lambda k: prompts.STRATEGIES[k].label)
        context = state.last_context or "<the retrieved report pages are inserted here>"
        profile = ReportProfile(**(state.result["profile"])) if state.result else None
        system, user = prompts.build_extraction_prompt(prompts.STRATEGIES[strategy_key], cfg.metric_ids,
                                                       context[:3000] + ("\n[...]" if len(context) > 3000 else ""),
                                                       profile, 2)
        st.markdown("**System instruction**")
        st.code(system, language="text", wrap_lines=True)
        st.markdown("**User prompt** (excerpts shortened)")
        st.code(user, language="text", wrap_lines=True)
        st.markdown("**Verification prompt (auditor)**")
        st.code(prompts.AUDITOR_ROLE + "\n\n" + prompts.build_verification_prompt("<items with source text>", 0)[1],
                language="text", wrap_lines=True)


def tab_history():
    st.subheader("🗂️ Analysis history (SQLite)")
    items = storage.list_analyses()
    if not items:
        st.info("No analyses stored yet.")
        return
    frame = pd.DataFrame(items)
    st.dataframe(frame, hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([3, 1, 1])
    choice = c1.selectbox("Analysis", [i["analysis_id"] for i in items],
                          format_func=lambda a: next(f"{i['created_at']} · {i['company']} · {i['model']}"
                                                     for i in items if i["analysis_id"] == a))
    if c2.button("Open", width="stretch"):
        loaded = storage.load_analysis(choice)
        if loaded is None:
            st.error("This analysis could not be loaded from the database.")
            return
        state.result = loaded
        pdf = storage.load_pdf(loaded["file_hash"])
        if pdf:
            state.pdf_bytes, state.pdf_name = pdf, loaded["filename"]
        state["_notice"] = (f"Opened {loaded['profile'].get('company_name')} "
                            f"({loaded['profile'].get('reporting_year') or 'year n/a'}) – {loaded['filename']}")
        state["_goto_tab"] = METRICS_TAB
        st.rerun()
    if c3.button("Delete", width="stretch"):
        storage.delete_analysis(choice)
        if state.result and state.result["analysis_id"] == choice:
            state.result = None
        st.rerun()
    usage = storage.usage_by_model()
    if usage:
        st.markdown("**Token usage per model and stage**")
        st.dataframe(pd.DataFrame(usage), hide_index=True, width="stretch")


def tab_about():
    st.subheader("ℹ️ How it works")
    st.markdown("""
| Step | What happens | Technique |
|---|---|---|
| 1. Parse | PyMuPDF reads text, detects tables (→ Markdown) and pages with charts or scans | multimodal input |
| 2. Profile | Company, year, frameworks, assurance | zero-shot + role, T = 0 |
| 3. Retrieve | BM25 ranks pages per metric; chart pages are rendered as images | lexical RAG |
| 4. Extract | Values, units, years, pages, evidence quotes as JSON | role + few-shot + chain-of-thought, T ≈ 0.1 |
| 4b. Repair | Invalid JSON is sent back to the model with the parser error | iterative self-repair |
| 5. Rules | Number on page? Quote on page? Unit, range, target words | deterministic guardrails |
| 6. Verify | Auditor persona checks each value against the PDF text (optionally a 2nd model) | self-verification / cross-model |
| 7. Summarise | Analyst briefing from verified data only | role prompting, T ≈ 0.6 |

**Error handling:** invalid/encrypted/oversized PDFs, unreadable pages, missing API key, rate limits (exponential
backoff with retry-after), daily quota, timeouts, 5xx, oversized prompts (automatic context shrinking), unsupported
parameters (automatic fallback), malformed JSON (local repair → model self-repair), degenerate repetition at low
temperature (retry at T = 1.0), and graceful degradation when profiling, verification or the summary fail.

**Free API keys:** [Google AI Studio (Gemini)](https://aistudio.google.com/apikey) ·
[Groq console](https://console.groq.com/keys). Put them in `.env` as `GEMINI_API_KEY=...` / `GROQ_API_KEY=...`.
""")
    st.caption("The metric catalogue uses the ConsciESG metric IDs of the original ESG extraction project.")


# ======================================================================
# Main
# ======================================================================
config = sidebar()
# A button can ask to switch tabs (e.g. "Open" in History). The request is stored first and applied
# here, before the tabs widget is created, because a widget's state cannot change after it is drawn.
if "_goto_tab" in state:
    state["main_tab"] = state.pop("_goto_tab")
if "_notice" in state:
    st.toast(state.pop("_notice"), icon="✅")
tabs = st.tabs(TAB_NAMES, key="main_tab", on_change="rerun")
renderers = [lambda: tab_analyze(config), tab_metrics, tab_evidence, tab_summary, tab_pipeline,
             lambda: tab_lab(config), tab_history, tab_about]
for tab, render in zip(tabs, renderers):
    with tab:
        try:
            render()
        except Exception as error:  # keep the other tabs usable if one view fails
            st.error("This view could not be displayed. The rest of the app still works.")
            with st.expander("Technical details"):
                st.exception(error)