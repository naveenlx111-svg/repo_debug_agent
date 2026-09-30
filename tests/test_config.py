import pytest

from repo_debug_agent.cli import build_parser, settings_from_args
from repo_debug_agent.config import AgentSettings, ConfigError, LLMSettings


def test_provider_defaults_to_groq_when_key_present():
    s = LLMSettings.resolve(env={"GROQ_API_KEY": "k"})
    assert s.provider == "groq" and s.api_key == "k" and s.fix_model == "llama-3.3-70b-versatile"


def test_local_provider_auto_detects_model_and_disables_thinking():
    s = LLMSettings.resolve(env={})
    assert s.provider == "local" and s.fix_model is None
    assert s.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_precedence_cli_over_env_over_preset():
    env = {"LLM_PROVIDER": "local", "LLM_MODEL": "env-model", "LLM_BASE_URL": "http://env/v1"}
    s = LLMSettings.resolve(model="cli-model", env=env)
    assert (s.triage_model, s.fix_model, s.base_url) == ("cli-model", "cli-model", "http://env/v1")
    s = LLMSettings.resolve(fix_model="big", env=env)
    assert (s.triage_model, s.fix_model) == ("env-model", "big")


def test_groq_without_key_is_a_clear_error():
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        LLMSettings.resolve(provider="groq", env={})


def test_bad_extra_body_is_rejected():
    with pytest.raises(ConfigError, match="LLM_EXTRA_BODY"):
        LLMSettings.resolve(env={"LLM_EXTRA_BODY": "{not json"})


def test_agent_settings_validate(tmp_path):
    llm = LLMSettings.resolve(env={})
    with pytest.raises(ConfigError, match="does not exist"):
        AgentSettings(repo=tmp_path / "missing", llm=llm, out_dir=tmp_path)
    with pytest.raises(ConfigError, match="min-confidence"):
        AgentSettings(repo=tmp_path, llm=llm, out_dir=tmp_path, min_confidence=5)


def test_cli_accepts_positional_and_legacy_repo_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    parser = build_parser()
    a = settings_from_args(parser.parse_args([str(tmp_path), "--provider", "local"]))
    b = settings_from_args(parser.parse_args(["--repo", str(tmp_path), "--provider", "local"]))
    assert a.repo == b.repo == tmp_path.resolve()


def test_empty_timeout_means_default():
    # Found by the agent reviewing its own source: every other LLM_* variable treats an
    # empty value (e.g. `LLM_TIMEOUT=` in .env) as unset, but this one raised ConfigError.
    assert LLMSettings.resolve(env={"LLM_TIMEOUT": ""}).timeout == 300
    assert LLMSettings.resolve(env={"LLM_TIMEOUT": "  "}).timeout == 300
    assert LLMSettings.resolve(env={"LLM_TIMEOUT": "60"}).timeout == 60


def test_non_numeric_timeout_is_a_config_error():
    with pytest.raises(ConfigError, match="LLM_TIMEOUT"):
        LLMSettings.resolve(env={"LLM_TIMEOUT": "soon"})


def test_workers_default_depends_on_provider(tmp_path):
    groq = LLMSettings.resolve(env={"GROQ_API_KEY": "k"})
    local = LLMSettings.resolve(env={})
    assert AgentSettings(repo=tmp_path, llm=groq, out_dir=tmp_path).workers == 4
    assert AgentSettings(repo=tmp_path, llm=local, out_dir=tmp_path).workers == 1
    assert AgentSettings(repo=tmp_path, llm=local, out_dir=tmp_path, workers=3).workers == 3
