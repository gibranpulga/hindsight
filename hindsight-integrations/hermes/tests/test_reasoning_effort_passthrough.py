"""`llm_reasoning_effort` passthrough: config key -> daemon env.

Thinking-capable models served over OpenAI-wire providers (qwen via OpenRouter)
reject `tool_choice=required` while in thinking mode, which breaks reflect.
Pinning `llm_reasoning_effort: "none"` in the plugin config must reach the
daemon through the profile env file, and must survive re-materialization —
which requires it to be part of the managed key set, not a hand-appended line
(those are wiped on rewrite).
"""

import textwrap

from hindsight_hermes import HindsightMemoryProvider, embedded


def test_reasoning_effort_is_a_managed_env_key():
    config = {
        "llm_provider": "openrouter",
        "llm_model": "qwen/qwen3.8-flash",
        "llm_api_key": "sk-test",
        "llm_reasoning_effort": "none",
    }
    env = embedded._build_embedded_profile_env(config)
    assert env["HINDSIGHT_API_LLM_REASONING_EFFORT"] == "none"


def test_reasoning_effort_omitted_when_unset():
    """Configs that never set it must see no behavior change: key absent, not empty."""
    config = {"llm_provider": "ollama", "llm_model": "gemma3:12b"}
    env = embedded._build_embedded_profile_env(config)
    assert "HINDSIGHT_API_LLM_REASONING_EFFORT" not in env


def test_reasoning_effort_survives_rematerialization(tmp_path):
    """The full write path: materialize -> file on disk contains the key."""
    import os

    config = {
        "profile": "efforttest",
        "llm_provider": "openrouter",
        "llm_model": "qwen/qwen3.8-flash",
        "llm_api_key": "sk-test",
        "llm_reasoning_effort": "none",
        "home": str(tmp_path),
    }
    path = embedded._materialize_embedded_profile_env(config)
    saved = embedded._load_simple_env(path)
    assert saved["HINDSIGHT_API_LLM_REASONING_EFFORT"] == "none"
    # the materialized file stays owner-only
    assert stat_mode(path) == 0o600


def stat_mode(path):
    import stat

    return stat.S_IMODE(path.stat().st_mode)


def test_reasoning_effort_disk_drift_and_rematerialization(hermes_env):
    """Compare actual profile values and ensure stale managed values are removed."""
    config = {
        "profile": "efforttest",
        "llm_provider": "openrouter",
        "llm_model": "qwen/qwen3.8-flash",
        "llm_api_key": "sk-test",
        "llm_reasoning_effort": "none",
    }
    path = embedded._materialize_embedded_profile_env(config)
    assert embedded._profile_env_drifted(config) is False

    config["llm_reasoning_effort"] = "low"
    assert embedded._profile_env_drifted(config) is True

    # Values outside the plugin-managed set survive a synchronization.
    path.write_text(path.read_text() + "HINDSIGHT_API_PORT=9177\n", encoding="utf-8")
    embedded._materialize_embedded_profile_env(config)
    saved = embedded._load_simple_env(path)
    assert saved["HINDSIGHT_API_LLM_REASONING_EFFORT"] == "low"
    assert saved["HINDSIGHT_API_PORT"] == "9177"
    assert embedded._profile_env_drifted(config) is False

    config.pop("llm_reasoning_effort")
    assert embedded._profile_env_drifted(config) is True
    embedded._materialize_embedded_profile_env(config)
    saved = embedded._load_simple_env(path)
    assert "HINDSIGHT_API_LLM_REASONING_EFFORT" not in saved
    assert saved["HINDSIGHT_API_PORT"] == "9177"
    assert embedded._profile_env_drifted(config) is False


def test_schema_exposes_the_setting():
    """The setup wizard / config panel must offer the key next to llm_model."""
    provider_schema = HindsightMemoryProvider().get_config_schema()
    keys = [field["key"] for field in provider_schema]
    assert "llm_reasoning_effort" in keys
    field = next(f for f in provider_schema if f["key"] == "llm_reasoning_effort")
    assert field["when"] == {"mode": "local_embedded"}
    assert "tool_choice" in field["description"]


def test_schema_placed_beside_llm_model():
    """Discoverable: directly after llm_model in the schema list."""
    provider_schema = HindsightMemoryProvider().get_config_schema()
    keys = [field["key"] for field in provider_schema]
    assert keys.index("llm_reasoning_effort") == keys.index("llm_model") + 1
