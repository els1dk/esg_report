"""
Provider-independent LLM client with retries and error classification.

Two free-tier APIs are supported behind one interface:
  * Google Gemini (google-genai SDK) – multimodal (text + page images)
  * Groq (groq SDK, OpenAI-compatible) – very fast open models

Every SDK exception is translated into one of the error classes below, so
the pipeline and the UI can react in a uniform way:

  error                    retried?   typical cause
  ---------------------------------------------------------------------
  RateLimitError           yes        429, too many requests per minute
  QuotaExhaustedError      no         free daily quota used up
  ServiceTimeoutError      yes        network / server took too long
  ServiceUnavailableError  yes        5xx, connection reset
  EmptyResponseError       yes        model returned no text
  AuthenticationError      no         missing / invalid API key
  ContextTooLargeError     no*        prompt too long for model or TPM limit
  ModelNotFoundError       no         wrong / retired model ID
  SafetyBlockedError       no         provider safety filter
  BadRequestError          no*        unsupported parameter, malformed request

  * handled one level up: the pipeline shrinks the context, or the client
    retries once without the optional parameters.
"""

from __future__ import annotations

import base64
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ------------------------------------------------------------------
# Error classes
# ------------------------------------------------------------------
class LLMError(RuntimeError):
    retryable = False
    user_message = "The AI service returned an error."

    def __init__(self, message: str, *, retry_after: float | None = None, raw: str = ""):
        super().__init__(message)
        self.retry_after = retry_after
        self.raw = raw


class RateLimitError(LLMError):
    retryable = True
    user_message = "⏳ Rate limit reached on the free tier. The app waited and retried, but the limit persisted."


class QuotaExhaustedError(LLMError):
    user_message = ("📉 The free daily quota for this model is used up. Switch to another model "
                    "(Flash-Lite models have higher free limits) or another provider, or try tomorrow.")


class ServiceTimeoutError(LLMError):
    retryable = True
    user_message = "⌛ The AI service did not respond in time. Please retry."


class ServiceUnavailableError(LLMError):
    retryable = True
    user_message = "⚠️ The AI service is temporarily unavailable. Please retry in a minute."


class EmptyResponseError(LLMError):
    retryable = True
    user_message = "The model returned an empty answer."


class AuthenticationError(LLMError):
    user_message = ("🔑 The API key was rejected or is missing. Add it to the .env file or paste it in "
                    "the sidebar.")


class ContextTooLargeError(LLMError):
    user_message = "📚 The prompt is too large for this model. Reduce 'pages sent to the model'."


class ModelNotFoundError(LLMError):
    user_message = "❓ The model ID was not found. Pick another model in the sidebar."


class SafetyBlockedError(LLMError):
    user_message = "🛡️ The provider's safety filter blocked this request."


class BadRequestError(LLMError):
    user_message = "The AI service rejected the request."


# ------------------------------------------------------------------
# Request / response containers
# ------------------------------------------------------------------
@dataclass
class ImageInput:
    data: bytes
    mime_type: str = "image/jpeg"
    label: str = ""


@dataclass
class LLMRequest:
    system: str
    prompt: str
    images: list[ImageInput] = field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = 4096
    reasoning: str = "low"
    json_mode: bool = True
    timeout_s: float = 180.0


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    attempts: int = 1
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


class BaseProvider:
    """Interface every provider implements."""

    name = "base"

    def __init__(self, model: str):
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------
# Helpers for error translation
# ------------------------------------------------------------------
_RETRY_DELAY_RE = re.compile(r"(?:retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?|retry in\s+|try again in\s+)"
                             r"(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _retry_after_from_text(text: str) -> float | None:
    match = _RETRY_DELAY_RE.search(text or "")
    return float(match.group(1)) if match else None


def classify_http_error(status: int | None, message: str) -> LLMError:
    """Map an HTTP status + message to one of our error classes."""
    text = (message or "").lower()
    retry_after = _retry_after_from_text(message)
    if status in (401, 403) or "api key not valid" in text or "invalid api key" in text \
            or "api_key_invalid" in text or "permission denied" in text:
        return AuthenticationError(message)
    if status == 404 or "not found" in text and "model" in text:
        return ModelNotFoundError(message)
    if status == 413 or "too large" in text or "context length" in text or "token count" in text \
            or "exceeds the maximum" in text or "context_length_exceeded" in text:
        return ContextTooLargeError(message)
    if status == 429 or "resource_exhausted" in text or "rate limit" in text:
        if "per day" in text or "perday" in text or "daily" in text or "tokens per day" in text:
            return QuotaExhaustedError(message, retry_after=retry_after)
        return RateLimitError(message, retry_after=retry_after)
    if status in (408, 504) or "deadline" in text or "timed out" in text or "timeout" in text:
        return ServiceTimeoutError(message)
    if status is not None and status >= 500 or "unavailable" in text or "overloaded" in text:
        return ServiceUnavailableError(message)
    if "safety" in text or "blocked" in text:
        return SafetyBlockedError(message)
    return BadRequestError(message)


def _is_parameter_problem(error: LLMError) -> bool:
    text = str(error).lower()
    return isinstance(error, BadRequestError) and any(
        word in text for word in ("reasoning", "thinking", "top_p", "unsupported", "not supported",
                                  "unknown parameter", "unrecognized", "invalid argument", "unexpected keyword"))


# ------------------------------------------------------------------
# Google Gemini
# ------------------------------------------------------------------
class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        if not api_key:
            raise AuthenticationError("No Gemini API key provided (GEMINI_API_KEY).")
        from google import genai  # imported lazily: only needed when Gemini is used
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._use_optional_params = True

    def _config(self, request: LLMRequest):
        types = self._genai.types
        kwargs = dict(
            system_instruction=request.system,
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_output_tokens,
            http_options=types.HttpOptions(timeout=int(request.timeout_s * 1000)),
        )
        if request.json_mode:
            kwargs["response_mime_type"] = "application/json"
        if self._use_optional_params and request.reasoning:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=request.reasoning)
        return types.GenerateContentConfig(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        types = self._genai.types
        contents = []
        for image in request.images:
            if image.label:
                contents.append(image.label)
            contents.append(types.Part.from_bytes(data=image.data, mime_type=image.mime_type))
        contents.append(request.prompt)

        started = time.perf_counter()
        warnings: list[str] = []
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=self._config(request))
        except Exception as error:  # translate SDK errors
            translated = self._translate(error)
            if self._use_optional_params and _is_parameter_problem(translated):
                # Older/other models may not support thinking_level: retry without it.
                self._use_optional_params = False
                warnings.append(f"Model rejected an optional parameter; retried without it ({translated}).")
                response = self._retry_plain(contents, request)
            else:
                raise translated from error
        latency = time.perf_counter() - started
        return self._to_response(response, latency, warnings)

    def _retry_plain(self, contents, request):
        try:
            return self._client.models.generate_content(
                model=self.model, contents=contents, config=self._config(request))
        except Exception as error:
            raise self._translate(error) from error

    def _to_response(self, response, latency: float, warnings: list[str]) -> LLMResponse:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise SafetyBlockedError(f"Prompt blocked: {feedback.block_reason}")

        finish = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")
        if "SAFETY" in finish.upper() or "PROHIBITED" in finish.upper():
            raise SafetyBlockedError(f"Response blocked by safety filter ({finish}).")

        try:
            text = response.text or ""
        except Exception:  # .text raises if the candidate has no text parts
            text = ""
        if not text.strip():
            raise EmptyResponseError(f"Empty response (finish reason: {finish or 'unknown'}).")

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
            + int(getattr(usage, "thoughts_token_count", 0) or 0),
            latency_s=latency,
            finish_reason=finish,
            truncated="MAX_TOKENS" in finish.upper(),
            warnings=warnings,
        )

    @staticmethod
    def _translate(error: Exception) -> LLMError:
        if isinstance(error, LLMError):
            return error
        name = type(error).__name__.lower()
        if "timeout" in name:
            return ServiceTimeoutError(str(error))
        if "connect" in name or "network" in name:
            return ServiceUnavailableError(str(error))
        status = getattr(error, "code", None) or getattr(error, "status_code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        return classify_http_error(status, str(error))


# ------------------------------------------------------------------
# Groq (OpenAI-compatible chat completions)
# ------------------------------------------------------------------
class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, model: str, api_key: str, max_images: int = 1, vision: bool = True):
        super().__init__(model)
        if not api_key:
            raise AuthenticationError("No Groq API key provided (GROQ_API_KEY).")
        import groq  # lazy import
        self._groq = groq
        # max_retries=0: retries are handled by call_with_retries, so they
        # don't multiply with the SDK's own retry loop.
        self._client = groq.Groq(api_key=api_key, max_retries=0)
        self.max_images = max_images
        self.vision = vision
        self._use_optional_params = True

    def _reasoning_params(self, level: str) -> dict:
        """Each Groq model family names its reasoning controls differently."""
        if not self._use_optional_params:
            return {}
        model = self.model.lower()
        if "gpt-oss" in model:
            effort = level if level in ("low", "medium", "high") else "low"
            return {"reasoning_effort": effort, "include_reasoning": False}
        if "qwen" in model:
            effort = level if level in ("none", "low", "medium", "high") else "none"
            return {"reasoning_effort": effort, "reasoning_format": "hidden"}
        return {}

    def generate(self, request: LLMRequest) -> LLMResponse:
        warnings: list[str] = []
        # Groq recommends putting instructions in the user turn for reasoning
        # models, so the system text is prepended to the prompt.
        text = f"{request.system}\n\n{request.prompt}"
        images = request.images
        if images and not self.vision:
            warnings.append(f"{self.model} is text-only: {len(images)} page image(s) were not sent.")
            images = []
        if len(images) > self.max_images:
            warnings.append(f"Only {self.max_images} of {len(images)} images sent (model/free-tier limit).")
            images = images[: self.max_images]

        if images:
            content: list[dict] | str = [{"type": "text", "text": text}]
            for image in images:
                encoded = base64.b64encode(image.data).decode("ascii")
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"}})
        else:
            content = text

        def _call():
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=request.temperature,
                top_p=request.top_p,
                max_completion_tokens=request.max_output_tokens,
                timeout=request.timeout_s,
                **self._reasoning_params(request.reasoning),
            )
            if request.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return self._client.chat.completions.create(**kwargs)

        started = time.perf_counter()
        try:
            completion = _call()
        except Exception as error:
            translated = self._translate(error)
            if self._use_optional_params and _is_parameter_problem(translated):
                self._use_optional_params = False
                warnings.append(f"Model rejected an optional parameter; retried without it ({translated}).")
                try:
                    completion = _call()
                except Exception as error2:
                    raise self._translate(error2) from error2
            else:
                raise translated from error
        latency = time.perf_counter() - started

        choice = completion.choices[0]
        message_text = choice.message.content or ""
        if not message_text.strip():
            raise EmptyResponseError(f"Empty response (finish reason: {choice.finish_reason}).")
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=message_text,
            model=self.model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=latency,
            finish_reason=str(choice.finish_reason or ""),
            truncated=choice.finish_reason == "length",
            warnings=warnings,
        )

    def _translate(self, error: Exception) -> LLMError:
        groq = self._groq
        if isinstance(error, LLMError):
            return error
        if isinstance(error, groq.APITimeoutError):
            return ServiceTimeoutError(str(error))
        if isinstance(error, groq.APIConnectionError):
            return ServiceUnavailableError(str(error))
        status = getattr(error, "status_code", None)
        message = str(error)
        body = getattr(error, "body", None)
        # JSON mode failure: Groq returns 400 with the model's broken output in
        # "failed_generation". Surface it so the pipeline can run self-repair.
        if isinstance(body, dict):
            details = body.get("error", body)
            failed = details.get("failed_generation") if isinstance(details, dict) else None
            if failed:
                return BadRequestError("json_validate_failed", raw=failed)
        retry_after = None
        response = getattr(error, "response", None)
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after"))
            except (TypeError, ValueError):
                retry_after = None
        translated = classify_http_error(status, message)
        if retry_after and translated.retry_after is None:
            translated.retry_after = retry_after
        return translated


# ------------------------------------------------------------------
# Retry loop
# ------------------------------------------------------------------
def backoff_delay(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter: ~2s, 4s, 8s, 16s ... capped at 60s."""
    return min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, 1)


def call_with_retries(provider: BaseProvider, request: LLMRequest, max_attempts: int = 4,
                      on_event: Optional[Callable[[str], None]] = None,
                      sleep: Callable[[float], None] = time.sleep,
                      max_wait_s: float = 90.0) -> LLMResponse:
    """
    Call ``provider.generate`` and retry transient failures with exponential
    backoff. Honours the server's retry-after hint for rate limits.
    """
    last_error: LLMError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = provider.generate(request)
            response.attempts = attempt
            return response
        except LLMError as error:
            last_error = error
            if not error.retryable or attempt == max_attempts:
                raise
            wait = error.retry_after if error.retry_after else backoff_delay(attempt)
            wait = min(wait, max_wait_s)
            if on_event:
                on_event(f"{type(error).__name__}: {str(error)[:140]} → retrying in {wait:.0f}s "
                         f"(attempt {attempt + 1}/{max_attempts})")
            sleep(wait)
    raise last_error  # pragma: no cover - loop always returns or raises


def make_provider(provider_key: str, model: str, api_key: str = "", **kwargs) -> BaseProvider:
    """Factory used by the pipeline and the UI."""
    if provider_key == "gemini":
        return GeminiProvider(model, api_key)
    if provider_key == "groq":
        return GroqProvider(model, api_key, max_images=kwargs.get("max_images", 1),
                            vision=kwargs.get("vision", True))
    if provider_key == "mock":
        from .mock_provider import MockProvider
        return MockProvider(model)
    raise ValueError(f"Unknown provider: {provider_key}")
