from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from repo_debug_agent.config import AgentSettings, LLMSettings
from repo_debug_agent.llm import Message, Usage


class ScriptedLLM:
    """A fake chat model: `respond(messages, json_mode)` decides each reply."""

    def __init__(self, respond: Callable[[list[Message], bool], str]):
        self.respond = respond
        self.usage = Usage()
        self.calls: list[list[Message]] = []
        self.options: list[dict] = []

    def complete(
        self,
        messages,
        *,
        model,
        max_tokens=2048,
        temperature=None,
        json_mode=False,
        json_schema=None,
    ):
        self.calls.append([dict(m) for m in messages])
        self.options.append({"json_mode": json_mode, "json_schema": json_schema})
        self.usage.calls += 1
        return self.respond(messages, json_mode)


def review_reply(*issues: dict) -> str:
    return json.dumps({"issues": list(issues)})


def fix_reply(code: str, explanation: str = "Fixed it.", lang: str = "python") -> str:
    return f"```{lang}\n{code}\n```\nEXPLANATION: {explanation}"


def is_repro_request(messages: list[Message]) -> bool:
    return "demonstrates the bug" in messages[1]["content"]


def is_vet_request(messages: list[Message]) -> bool:
    return "checking another engineer's bug reproduction" in messages[0]["content"]


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def make_settings(tmp_path):
    def make(repo: Path, **overrides) -> AgentSettings:
        llm = LLMSettings(
            provider="local",
            base_url="http://unused",
            api_key=None,
            triage_model="fake",
            fix_model="fake",
        )
        defaults = dict(
            repo=repo,
            llm=llm,
            out_dir=tmp_path / "reports",
            use_embeddings=False,
            repro=False,  # tests that want reproduction turn it on explicitly
            review_cache=False,  # likewise for the review cache
            review_passes=1,  # and for follow-up review passes
            workers=1,
            index_dir=tmp_path / "index",
        )
        defaults.update(overrides)
        return AgentSettings(**defaults)

    return make


PYTEST_CMD = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'
