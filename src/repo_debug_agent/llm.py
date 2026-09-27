"""Chat-completion client plus helpers for pulling structure out of model replies.

One client covers every provider we support, because Groq, llama.cpp,
Ollama, vLLM and LM Studio all speak the OpenAI chat-completions protocol.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from repo_debug_agent.config import LLMSettings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """A request failed; the caller may retry or skip the current item."""


class LLMTruncatedError(LLMError):
    """The reply hit max_tokens before finishing."""


class LLMUnavailableError(LLMError):
    """The endpoint cannot be reached at all; there is no point continuing the run."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatModel(Protocol):
    usage: Usage

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings):
        import openai  # imported lazily so `--help` stays fast

        self._openai = openai
        self.settings = settings
        self.usage = Usage()
        self._lock = threading.Lock()
        self._json_mode_supported = True
        self._client = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-needed",
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list()]
        except self._openai.APIConnectionError as e:
            raise LLMUnavailableError(f"cannot reach {self.settings.base_url}: {e}") from e
        except self._openai.APIError as e:
            raise LLMError(f"listing models at {self.settings.base_url} failed: {e}") from e

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        if self.settings.extra_body:
            kwargs["extra_body"] = self.settings.extra_body
        use_json = json_mode and self._json_mode_supported
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._openai.BadRequestError as e:
            if use_json and ("response_format" in str(e) or "json" in str(e).lower()):
                log.warning("endpoint rejected JSON mode; falling back to prompt-only JSON (%s)", e)
                self._json_mode_supported = False
                return self.complete(
                    messages, model=model, max_tokens=max_tokens, temperature=temperature
                )
            raise LLMError(f"request rejected: {e}") from e
        except self._openai.APITimeoutError as e:
            # Subclass of APIConnectionError, but the server is up, just slow (or busy with
            # queued requests): fail this request only, not the whole run.
            raise LLMError(f"request timed out after {self.settings.timeout:.0f}s") from e
        except self._openai.APIConnectionError as e:
            raise LLMUnavailableError(f"cannot reach {self.settings.base_url}: {e}") from e
        except self._openai.APIError as e:
            raise LLMError(str(e)) from e

        with self._lock:
            self.usage.calls += 1
            if response.usage:
                self.usage.prompt_tokens += response.usage.prompt_tokens or 0
                self.usage.completion_tokens += response.usage.completion_tokens or 0

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMTruncatedError(f"reply was cut off at max_tokens={max_tokens}")
        text = strip_reasoning(choice.message.content or "")
        if not text:
            raise LLMError(
                f"model returned no content (finish_reason={choice.finish_reason}); "
                "if it is a reasoning model, its thinking may have used the whole token budget"
            )
        return text


class RecordingLLM:
    """Wraps a ChatModel and keeps every exchange, for the run's transcript file."""

    def __init__(self, inner: ChatModel):
        self.inner = inner
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def usage(self) -> Usage:
        return self.inner.usage

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        started = time.monotonic()
        reply: str | None = None
        error: str | None = None
        try:
            reply = self.inner.complete(messages, **kwargs)
            return reply
        except LLMError as e:
            error = str(e)
            raise
        finally:
            record = {
                "model": kwargs.get("model"),
                "json_mode": kwargs.get("json_mode", False),
                "seconds": round(time.monotonic() - started, 2),
                "messages": [dict(m) for m in messages],
                "reply": reply,
                "error": error,
            }
            with self._lock:
                self.records.append(record)


# --------------------------------------------------------------------------- reply parsing

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE)
_NON_CODE_FENCES = {
    "diff",
    "patch",
    "text",
    "txt",
    "console",
    "output",
    "bash",
    "sh",
    "shell",
    "json",
}


def strip_reasoning(text: str) -> str:
    """Remove inline <think>...</think> blocks some reasoning models emit."""
    text = _THINK_BLOCK.sub("", text)
    # An unterminated <think> means the model ran out of tokens mid-thought.
    if re.match(r"\s*<think>", text, re.IGNORECASE):
        return ""
    return text.strip()


def parse_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply.

    Tolerates markdown fences and prose around the JSON, which small models add
    even when told not to.
    """
    text = strip_reasoning(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("no JSON value found in model reply")


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return (language_tag, code) for every fenced block in a reply."""
    return [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(strip_reasoning(text))]


def pick_code_block(text: str) -> str | None:
    """The block most likely to be the answer: the longest one that isn't a diff/log/etc."""
    blocks = [code for tag, code in extract_code_blocks(text) if tag not in _NON_CODE_FENCES]
    blocks = [b for b in blocks if b.strip()]
    return max(blocks, key=len) if blocks else None
