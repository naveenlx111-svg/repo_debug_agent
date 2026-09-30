"""Run configuration.

Everything is resolved once, up front, from CLI flags and environment variables,
and then passed explicitly to the components that need it. Nothing in this
package reads the environment or talks to the network at import time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """Invalid or missing configuration; shown to the user without a traceback."""


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    api_key_env: str | None
    triage_model: str | None
    fix_model: str | None
    extra_body: dict = field(default_factory=dict)
    # Concurrent requests. Local servers usually process one at a time, and queued requests
    # just wait out their client-side timeout.
    workers: int = 1


PROVIDERS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        triage_model="llama-3.1-8b-instant",
        fix_model="llama-3.3-70b-versatile",
        workers=4,
    ),
    # Any OpenAI-compatible local server: llama.cpp's llama-server, LM Studio, vLLM, ...
    # The model is auto-detected from /v1/models when not given. Thinking is disabled
    # for hybrid reasoning models (Qwen3.x): it is slow and eats the token budget, and
    # the task is structured enough not to need it.
    "local": ProviderPreset(
        base_url="http://127.0.0.1:8080/v1",
        api_key_env="LLM_API_KEY",
        triage_model=None,
        fix_model=None,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "ollama": ProviderPreset(
        base_url="http://127.0.0.1:11434/v1",
        api_key_env=None,
        triage_model="qwen2.5-coder:7b",
        fix_model="qwen2.5-coder:7b",
    ),
}


@dataclass
class LLMSettings:
    provider: str
    base_url: str
    api_key: str | None
    triage_model: str | None  # None = auto-detect from the server
    fix_model: str | None
    temperature: float = 0.1
    timeout: float = 300.0
    max_retries: int = 3
    extra_body: dict = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        triage_model: str | None = None,
        fix_model: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> LLMSettings:
        """Merge CLI values > environment variables > provider preset."""
        env = os.environ if env is None else env
        provider = (
            provider or env.get("LLM_PROVIDER") or ("groq" if env.get("GROQ_API_KEY") else "local")
        )
        if provider not in PROVIDERS:
            raise ConfigError(f"unknown provider {provider!r}; choose from: {', '.join(PROVIDERS)}")
        preset = PROVIDERS[provider]

        api_key = env.get("LLM_API_KEY") or (
            env.get(preset.api_key_env) if preset.api_key_env else None
        )
        if provider == "groq" and not api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Put it in a .env file or your environment, "
                "or use --provider local to run against a local model."
            )

        if timeout is None:
            try:
                timeout = float(env.get("LLM_TIMEOUT", "").strip() or 300)  # empty = unset
            except ValueError as e:
                raise ConfigError(f"LLM_TIMEOUT must be a number of seconds: {e}") from e

        model = model or env.get("LLM_MODEL")
        extra_body = dict(preset.extra_body)
        if env.get("LLM_EXTRA_BODY"):
            try:
                extra_body.update(json.loads(env["LLM_EXTRA_BODY"]))
            except (json.JSONDecodeError, TypeError) as e:
                raise ConfigError(f"LLM_EXTRA_BODY is not a JSON object: {e}") from e

        return cls(
            provider=provider,
            base_url=base_url or env.get("LLM_BASE_URL") or preset.base_url,
            api_key=api_key,
            triage_model=triage_model
            or env.get("LLM_TRIAGE_MODEL")
            or model
            or preset.triage_model,
            fix_model=fix_model or env.get("LLM_FIX_MODEL") or model or preset.fix_model,
            timeout=timeout,
            extra_body=extra_body,
        )


def default_index_dir() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "repo-debug-agent" / "index"


@dataclass
class AgentSettings:
    repo: Path
    llm: LLMSettings
    out_dir: Path

    # Verification
    test_cmd: str | None = None
    test_timeout: float = 600.0

    # What to do with the results
    apply: bool = False
    analyze_only: bool = False
    keep_sandbox: bool = False

    # Analysis
    include_tests: bool = False  # review/fix test files too (off: tests are the spec)
    max_issues: int = 20
    min_confidence: float = 0.6
    review_max_lines: int = 400  # files longer than this are reviewed in windows
    max_file_bytes: int = 256_000
    workers: int | None = None  # None: the provider's default

    # Fixing
    repro: bool = True  # reproduce Python bugs with a model-written script before fixing
    python: str | None = None  # interpreter for reproduction scripts (None: auto-detect)
    max_attempts: int = 3
    context_chars: int = 6_000  # budget for retrieved cross-file context in fix prompts

    # Retrieval
    use_embeddings: bool = True
    embed_model: str = "default"
    index_dir: Path = field(default_factory=default_index_dir)
    reindex: bool = False
    review_cache: bool = True  # reuse review replies for unchanged review prompts

    def __post_init__(self) -> None:
        self.repo = Path(self.repo).expanduser().resolve()
        if not self.repo.exists():
            raise ConfigError(f"repo path does not exist: {self.repo}")
        if not self.repo.is_dir():
            raise ConfigError(f"repo path is not a directory: {self.repo}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("--min-confidence must be between 0 and 1")
        if self.max_attempts < 1:
            raise ConfigError("--max-attempts must be at least 1")
        if self.workers is None:
            self.workers = PROVIDERS[self.llm.provider].workers
        if self.workers < 1:
            raise ConfigError("--workers must be at least 1")
