"""
esg_intel – multimodal ESG report intelligence.

Modules
-------
config          providers, models, default parameters, ESG metric catalogue
pdf_processor   PDF validation, text/table/visual extraction, page rendering
retrieval       BM25 page ranking and prompt-context assembly (RAG)
prompts         prompt templates and prompt-engineering strategies
llm_client      Gemini / Groq clients, error classification, retries
schemas         Pydantic models for every structured LLM output
json_utils      robust JSON parsing of LLM responses
validation      rule-based checks, unit normalisation, result fusion
pipeline        the end-to-end analysis pipeline
experiments     Prompt Lab (strategy × temperature experiments)
storage         SQLite history and stored PDFs
export          Excel / CSV / JSON / Markdown exports
mock_provider   offline test double used by the tests
"""

__version__ = "1.0.0"
