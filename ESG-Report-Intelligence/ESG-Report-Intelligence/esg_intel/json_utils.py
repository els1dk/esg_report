"""
Robust JSON parsing for LLM output.

Even in JSON mode, models sometimes wrap JSON in Markdown fences, prepend
reasoning (<think>...</think>), leave trailing commas or get cut off by the
max_output_tokens limit. This module cleans the common cases locally;
whatever it cannot fix is sent back to the model with a self-repair prompt
(see pipeline.py).
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    """Raised when text cannot be turned into JSON, even after local repairs."""

    def __init__(self, message: str, raw_text: str, truncated: bool = False):
        super().__init__(message)
        self.raw_text = raw_text
        self.truncated = truncated


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balanced_slice(text: str) -> tuple[str, bool]:
    """
    Return the first complete top-level JSON object/array in ``text``.
    The bool is True when the JSON starts but never closes (truncated output).
    """
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return "", False
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                break
            if not stack:
                return text[start:index + 1], False
    return text[start:], True


def _light_repairs(text: str) -> str:
    """Fix the most frequent small syntax slips without changing content."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    # Python-style literals that some models emit.
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    return text


def parse_llm_json(raw_text: str) -> Any:
    """
    Parse JSON from a model response.

    Steps: strip reasoning tags -> unwrap Markdown fences -> take the first
    balanced JSON value -> json.loads -> (on failure) light repairs -> loads.
    Raises JSONParseError with a ``truncated`` flag when nothing works.
    """
    if raw_text is None or not raw_text.strip():
        raise JSONParseError("The model returned an empty response.", raw_text or "")

    text = _THINK_RE.sub("", raw_text).strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    candidate, truncated = _balanced_slice(text)
    if not candidate:
        raise JSONParseError("No JSON object was found in the response.", raw_text)

    for attempt in (candidate, _light_repairs(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as error:
            last_error = error

    message = f"Invalid JSON ({last_error.msg} at char {last_error.pos})."
    if truncated:
        message = "The JSON output was cut off before it ended (output token limit reached?)."
    raise JSONParseError(message, raw_text, truncated=truncated)


def as_list_payload(data: Any, key: str) -> dict:
    """
    Normalise a payload to {key: [...]}.
    Models sometimes return a bare list instead of the requested object.
    """
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, dict) and key not in data:
        # e.g. {"results": [...]} or {"data": [...]}: use the first list value.
        for value in data.values():
            if isinstance(value, list):
                return {key: value}
    return data
