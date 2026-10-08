"""The start worker's profile-env drift check must only compare plugin-managed keys.

Regression tests for #5222: the daemon (and operators) append keys the plugin does
not manage (``HINDSIGHT_API_PORT`` at minimum) to the profile env file. A
full-dict ``saved != expected`` compare then reports "config changed" on every
session start and restarts a healthy daemon each time. The check must instead
compare only the keys ``_build_embedded_profile_env`` manages, and still detect
genuine drift in any managed key.
"""

import sys
from types import SimpleNamespace

import pytest

from hindsight_hermes import HindsightMemoryProvider


def _provider_with_env(monkeypatch, tmp_path, saved_env, expected_env):
    """A provider whose profile env path, saved env and expected env are fully
    controlled, plus call recording for the rewrite/stop/client steps."""
    order = []

    provider = HindsightMemoryProvider()
    provider._config = {"profile": "drifttest", "llm_provider": "ollama"}

    monkeypatch.setattr("hindsight_hermes._embedded_profile_env_path", lambda cfg: tmp_path / "p.env")
    monkeypatch.setattr("hindsight_hermes._load_simple_env", lambda path: dict(saved_env))
    monkeypatch.setattr("hindsight_hermes._build_embedded_profile_env", lambda cfg: dict(expected_env))
    monkeypatch.setattr("hindsight_hermes._may_rewrite_profile_env", lambda cfg: True)
    monkeypatch.setattr("hindsight_hermes._materialize_embedded_profile_env", lambda cfg: order.append("rewrote env"))
    monkeypatch.setattr("hindsight_hermes._daemon_is_running", lambda profile: True)
    monkeypatch.setattr("hindsight_hermes._stop_daemon", lambda profile: order.append("stopped daemon"))
    monkeypatch.setattr(type(provider), "_get_client", lambda self: order.append("built client"))
    return provider, order


class _Console(SimpleNamespace):
    pass


@pytest.fixture(autouse=True)
def _fake_embed_module(monkeypatch):
    """_daemon_start_worker imports hindsight_embed before the drift check."""
    from rich.console import Console

    manager_module = SimpleNamespace(console=None)
    monkeypatch.setitem(
        sys.modules,
        "hindsight_embed",
        SimpleNamespace(get_embed_manager=lambda: SimpleNamespace(), daemon_embed_manager=manager_module),
    )
    monkeypatch.setitem(sys.modules, "hindsight_embed.daemon_embed_manager", manager_module)
    manager_module.console = Console(file=open("/dev/null", "w"), force_terminal=False)


def test_daemon_added_foreign_key_is_not_drift(monkeypatch, tmp_path):
    """The exact #5222 repro: file holds the managed keys PLUS the daemon's own
    HINDSIGHT_API_PORT. No rewrite, no daemon restart — the client is still built."""
    expected = {
        "HINDSIGHT_API_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_LLM_API_KEY": "sk-x",
        "HINDSIGHT_API_LLM_MODEL": "gemma3:12b",
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    saved = {**expected, "HINDSIGHT_API_PORT": "9177"}
    provider, order = _provider_with_env(monkeypatch, tmp_path, saved, expected)

    provider._daemon_start_worker()

    assert order == ["built client"]


def test_matching_reasoning_effort_does_not_restart_daemon(monkeypatch, tmp_path):
    """A matching reasoning-effort setting is stable across daemon starts."""
    expected = {
        "HINDSIGHT_API_LLM_PROVIDER": "openrouter",
        "HINDSIGHT_API_LLM_API_KEY": "sk-x",
        "HINDSIGHT_API_LLM_MODEL": "qwen",
        "HINDSIGHT_API_LOG_LEVEL": "info",
        "HINDSIGHT_API_LLM_REASONING_EFFORT": "low",
    }
    provider, order = _provider_with_env(monkeypatch, tmp_path, expected, expected)

    provider._daemon_start_worker()

    assert order == ["built client"]


def test_removed_managed_key_is_drift(monkeypatch, tmp_path):
    """A managed key missing from the file IS drift: rewrite + restart, in order."""
    expected = {
        "HINDSIGHT_API_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_LLM_API_KEY": "sk-x",
        "HINDSIGHT_API_LLM_MODEL": "gemma3:12b",
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    saved = {k: v for k, v in expected.items() if k != "HINDSIGHT_API_LLM_MODEL"}
    provider, order = _provider_with_env(monkeypatch, tmp_path, saved, expected)

    provider._daemon_start_worker()

    assert order == ["rewrote env", "stopped daemon", "built client"]


def test_changed_managed_value_is_drift(monkeypatch, tmp_path):
    """A managed key with a different value IS drift (model rotation, key rotation)."""
    expected = {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_API_KEY": "sk-x",
        "HINDSIGHT_API_LLM_MODEL": "gpt-4o-mini",
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    saved = {**expected, "HINDSIGHT_API_LLM_MODEL": "gpt-4o"}
    provider, order = _provider_with_env(monkeypatch, tmp_path, saved, expected)

    provider._daemon_start_worker()

    assert order == ["rewrote env", "stopped daemon", "built client"]


def test_empty_file_is_drift(monkeypatch, tmp_path):
    """A missing/empty profile env file is drift (first boot, wiped file)."""
    provider, order = _provider_with_env(monkeypatch, tmp_path, {}, {"HINDSIGHT_API_LLM_PROVIDER": "ollama"})

    provider._daemon_start_worker()

    assert order == ["rewrote env", "stopped daemon", "built client"]


def test_unset_scope_still_skips_the_rewrite(monkeypatch, tmp_path):
    """The fail-closed key guard is unchanged: no rewrite when this process cannot
    see the key the file holds, even though the managed keys drifted."""
    expected = {"HINDSIGHT_API_LLM_API_KEY": "", "HINDSIGHT_API_LLM_MODEL": "gemma3:12b"}
    saved = {"HINDSIGHT_API_LLM_API_KEY": "sk-on-file", "HINDSIGHT_API_LLM_MODEL": "gemma3:12b"}
    provider, order = _provider_with_env(monkeypatch, tmp_path, saved, expected)
    monkeypatch.setattr("hindsight_hermes._may_rewrite_profile_env", lambda cfg: False)

    provider._daemon_start_worker()

    assert order == ["built client"]
