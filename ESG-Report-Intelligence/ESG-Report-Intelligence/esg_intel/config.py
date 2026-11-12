"""
Central configuration: AI providers, models, default generation parameters
and the ESG metric catalogue.

The metric catalogue re-uses the ConsciESG metric IDs (8, 9, 10, 33, ...)
from the original ESG extraction project, so results from this app line up
with the validated ESG database built there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "ESG Report Intelligence"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "esg_intelligence.db"
SAMPLES_DIR = BASE_DIR / "samples"

# Hard limits for uploaded documents (error handling for oversized input).
MAX_UPLOAD_MB = 80
MAX_PAGES_TO_SCAN = 600


# ============================================================
# AI PROVIDERS AND MODELS
# ============================================================

@dataclass(frozen=True)
class ModelInfo:
    """Static facts about one model, used to adapt requests to it."""

    model_id: str
    label: str
    vision: bool                 # accepts images in the prompt
    max_images: int = 0          # images per request the API accepts
    context_chars: int = 60_000  # safe text budget for retrieved pages
    max_output_tokens: int = 8192
    note: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    label: str
    env_var: str
    key_url: str
    models: tuple[ModelInfo, ...]
    # Parameters the provider exposes for "thinking" / reasoning depth.
    reasoning_levels: tuple[str, ...] = ()
    default_reasoning: str = ""


GEMINI = ProviderInfo(
    key="gemini",
    label="Google Gemini (free tier)",
    env_var="GEMINI_API_KEY",
    key_url="https://aistudio.google.com/apikey",
    models=(
        ModelInfo("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite (fast, highest free quota)",
                  vision=True, max_images=8, context_chars=90_000, max_output_tokens=16_384),
        ModelInfo("gemini-3.6-flash", "Gemini 3.6 Flash (better quality, lower free quota)",
                  vision=True, max_images=8, context_chars=90_000, max_output_tokens=16_384),
        ModelInfo("gemini-3.8-flash", "Gemini 3.8 Flash (newest Flash)",
                  vision=True, max_images=8, context_chars=90_000, max_output_tokens=16_384),
        ModelInfo("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite",
                  vision=True, max_images=8, context_chars=90_000, max_output_tokens=16_384),
    ),
    reasoning_levels=("minimal", "low", "medium", "high"),
    default_reasoning="low",
)

GROQ = ProviderInfo(
    key="groq",
    label="Groq (free tier, very fast)",
    env_var="GROQ_API_KEY",
    key_url="https://console.groq.com/keys",
    models=(
        # Groq's free plan allows ~8,000 tokens per minute, so a single
        # request must stay small: short context, at most one image.
        ModelInfo("qwen/qwen3.8-27b", "Qwen 3.8 27B (vision)",
                  vision=True, max_images=1, context_chars=10_000, max_output_tokens=2_500,
                  note="Preview model on Groq; free tier ≈ 8k tokens/minute."),
        ModelInfo("openai/gpt-oss-120b", "GPT-OSS 120B (text only)",
                  vision=False, context_chars=12_000, max_output_tokens=2_500),
        ModelInfo("openai/gpt-oss-20b", "GPT-OSS 20B (text only, fastest)",
                  vision=False, context_chars=12_000, max_output_tokens=2_500),
        ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B (text only)",
                  vision=False, context_chars=12_000, max_output_tokens=2_500),
    ),
    reasoning_levels=("none", "low", "medium", "high"),
    default_reasoning="low",
)

# Offline test double. Hidden from the UI unless ESG_ENABLE_MOCK=1 is set,
# it lets the full pipeline and the tests run without any API key.
MOCK = ProviderInfo(
    key="mock",
    label="Offline test provider (no API)",
    env_var="",
    key_url="",
    models=(ModelInfo("mock-extractor", "Rule-based test double", vision=True, max_images=4),),
)

PROVIDERS: dict[str, ProviderInfo] = {p.key: p for p in (GEMINI, GROQ, MOCK)}


def find_model(provider_key: str, model_id: str) -> ModelInfo:
    """Return the ModelInfo for a model; unknown (custom) IDs get safe defaults."""
    provider = PROVIDERS[provider_key]
    for model in provider.models:
        if model.model_id == model_id:
            return model
    # A custom model ID typed by the user: assume the provider's first
    # model's limits, which are the most conservative of that provider.
    base = provider.models[0]
    return ModelInfo(model_id, model_id, vision=base.vision, max_images=base.max_images,
                     context_chars=base.context_chars, max_output_tokens=base.max_output_tokens)


# ============================================================
# GENERATION PARAMETERS PER PIPELINE STAGE
# ============================================================

@dataclass
class StageParams:
    """Sampling parameters for one LLM call."""

    temperature: float
    top_p: float
    max_output_tokens: int
    reasoning: str = "low"  # Gemini thinking_level / Groq reasoning_effort


# Defaults chosen per task (see docs, section "Parameter tuning"):
# - profile / extraction / verification are factual tasks: low randomness
#   so the same report gives the same numbers;
# - the summary is prose for humans: moderate temperature for fluent,
#   less repetitive text.
DEFAULT_STAGE_PARAMS: dict[str, StageParams] = {
    "profile": StageParams(temperature=0.0, top_p=0.9, max_output_tokens=1024, reasoning="minimal"),
    "extraction": StageParams(temperature=0.1, top_p=0.9, max_output_tokens=8192, reasoning="low"),
    "verification": StageParams(temperature=0.0, top_p=0.9, max_output_tokens=4096, reasoning="low"),
    "summary": StageParams(temperature=0.6, top_p=0.95, max_output_tokens=2048, reasoning="low"),
}

STAGE_LABELS = {
    "profile": "1 · Report profiling",
    "extraction": "2 · Metric extraction",
    "repair": "2b · JSON self-repair",
    "verification": "3 · AI verification",
    "summary": "4 · Insight summary",
}


# ============================================================
# ESG METRIC CATALOGUE  (ConsciESG metric IDs)
# ============================================================

@dataclass(frozen=True)
class Metric:
    metric_id: int
    name: str
    pillar: str            # "E", "S" or "G"
    category: str
    standard_unit: str     # unit used in the validated ESG database
    unit_family: str       # emissions | volume | mass | percent | ratio | count | energy
    definition: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    core: bool = True      # part of the 17 ConsciESG template metrics


METRICS: tuple[Metric, ...] = (
    # ---------------- Environmental ----------------
    Metric(8, "Scope 1 GHG emissions total", "E", "Climate", "t CO2e", "emissions",
           "Total direct greenhouse-gas emissions from owned or controlled sources.",
           ("scope 1", "direct emissions", "direct ghg", "co2e", "tco2e", "greenhouse gas")),
    Metric(9, "Scope 2 GHG emissions total", "E", "Climate", "t CO2e", "emissions",
           "Total indirect emissions from purchased energy. Record whether market- or location-based.",
           ("scope 2", "indirect emissions", "market-based", "location-based", "purchased electricity", "co2e")),
    Metric(10, "Scope 3 GHG emissions total", "E", "Climate", "t CO2e", "emissions",
           "Total value-chain emissions (sum of all Scope 3 categories), never a single category.",
           ("scope 3", "value chain", "upstream", "downstream", "purchased goods", "co2e")),
    Metric(33, "Waste diverted from disposal: total reused/recycled", "E", "Waste", "t", "mass",
           "Waste diverted from disposal (reused, recycled, recovered), in tonnes or as % of total waste.",
           ("waste", "recycled", "recycling", "diverted", "reuse", "recovered", "landfill")),
    Metric(48, "Water discharge: percentage returned that is purified", "E", "Water", "%", "percent",
           "Share of discharged water that was treated / purified before being returned.",
           ("water discharge", "wastewater", "treated", "effluent", "purified", "discharged")),
    Metric(50, "Water consumption: total", "E", "Water", "m³", "volume",
           "Total water consumption (or withdrawal if only that is reported, stated in notes).",
           ("water consumption", "water withdrawal", "m3", "m³", "megalitres", "water use", "cubic")),
    # ---------------- Social ----------------
    Metric(69, "Maternity leave fully covered", "S", "Parental leave", "%", "percent",
           "Whether / share of maternity leave that is fully paid or covered by the employer.",
           ("maternity", "parental leave", "maternity leave", "fully paid", "returned to work")),
    Metric(70, "Paternity leave fully covered", "S", "Parental leave", "%", "percent",
           "Whether / share of paternity leave that is fully paid or covered by the employer.",
           ("paternity", "parental leave", "paternity leave", "fathers", "fully paid")),
    Metric(73, "Women in leadership", "S", "Diversity", "%", "percent",
           "Share of women in senior leadership / executive positions.",
           ("women in leadership", "senior management", "executive", "female leaders", "leadership positions")),
    Metric(75, "Women in management", "S", "Diversity", "%", "percent",
           "Share of women in all management positions.",
           ("women in management", "female managers", "management positions", "managers", "gender")),
    Metric(80, "Women-to-men average salary ratio: leadership", "S", "Pay equity", "ratio", "ratio",
           "Average female-to-male remuneration ratio for leadership.",
           ("pay ratio", "gender pay", "remuneration ratio", "salary ratio", "leadership")),
    Metric(83, "Women-to-men average salary ratio: management", "S", "Pay equity", "ratio", "ratio",
           "Average female-to-male remuneration ratio for management.",
           ("pay ratio", "gender pay gap", "remuneration", "salary ratio", "management")),
    Metric(86, "Women-to-men average salary ratio: workforce", "S", "Pay equity", "ratio", "ratio",
           "Average female-to-male remuneration ratio for the whole workforce.",
           ("gender pay gap", "pay ratio", "equal pay", "basic salary", "remuneration of women")),
    Metric(89, "Suppliers with forced or compulsory labor violations", "S", "Supply chain", "count", "count",
           "Number (or share) of suppliers identified with forced or compulsory labour violations.",
           ("forced labour", "forced labor", "compulsory labour", "modern slavery", "suppliers", "supplier audits")),
    Metric(90, "Suppliers with child labor violations", "S", "Supply chain", "count", "count",
           "Number (or share) of suppliers identified with child labour violations.",
           ("child labour", "child labor", "suppliers", "supplier audits", "human rights")),
    # ---------------- Governance ----------------
    Metric(71, "Women on board", "G", "Board", "%", "percent",
           "Share of women on the board of directors / supervisory board.",
           ("board of directors", "supervisory board", "women on the board", "board diversity", "female directors")),
    Metric(77, "Women-to-men average salary ratio: board", "G", "Board", "ratio", "ratio",
           "Average female-to-male remuneration ratio for board members.",
           ("board remuneration", "pay ratio", "board", "remuneration", "directors' fees")),
    # ---------------- Extended (optional) ----------------
    Metric(101, "Total energy consumption", "E", "Energy", "MWh", "energy",
           "Total energy consumed within the organisation.",
           ("energy consumption", "mwh", "gwh", "gj", "total energy", "fuel consumption"), core=False),
    Metric(102, "Renewable electricity share", "E", "Energy", "%", "percent",
           "Share of electricity from renewable sources.",
           ("renewable electricity", "renewable energy", "green electricity", "renewables", "share of renewable"), core=False),
    Metric(103, "Total employees", "S", "Workforce", "count", "count",
           "Total headcount at year end.",
           ("employees", "headcount", "workforce", "fte", "number of employees"), core=False),
    Metric(104, "Lost-time injury frequency rate", "S", "Health & safety", "rate", "count",
           "LTIFR / LTIR as reported (state the hours basis in notes).",
           ("ltifr", "lost time injury", "lost-time", "injury rate", "accident frequency"), core=False),
)

METRICS_BY_ID: dict[int, Metric] = {m.metric_id: m for m in METRICS}
CORE_METRIC_IDS: list[int] = [m.metric_id for m in METRICS if m.core]
PILLAR_NAMES = {"E": "Environmental", "S": "Social", "G": "Governance"}

# Words that mark a value as a target or forecast rather than a reported
# result. Used by the rule-based validator to flag suspicious evidence.
TARGET_WORDS = ("target", "by 2030", "by 2035", "by 2040", "by 2050", "goal", "ambition",
                "aim to", "commit to", "will reduce", "plan to", "forecast", "net zero by")
