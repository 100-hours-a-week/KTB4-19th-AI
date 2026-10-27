from types import SimpleNamespace

import pytest

from zipsai import settings as settings_module
from zipsai.errors import LlmUnavailableError
from zipsai.settings import get_settings


@pytest.mark.parametrize(
    ("provider", "key", "base_url", "model_vars", "expected_models"),
    [
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            {
                "OPENROUTER_LLM_MODEL": "openrouter-llm",
                "OPENROUTER_VLM_MODEL": "openrouter-vlm",
            },
            ("openrouter-llm", "openrouter-vlm"),
        ),
        (
            "serverless",
            "SERVERLESS_API_KEY",
            "SERVERLESS_BASE_URL",
            {"SERVERLESS_MODEL": "serverless-model"},
            ("serverless-model", "serverless-model"),
        ),
        (
            "pod",
            "POD_API_KEY",
            "POD_BASE_URL",
            {"POD_MODEL": "pod-model"},
            ("pod-model", "pod-model"),
        ),
    ],
)
def test_get_settings_uses_code_selected_provider_profile(
    monkeypatch, provider, key, base_url, model_vars, expected_models
):
    monkeypatch.setattr(settings_module, "ACTIVE_MODEL_PROVIDER", provider)
    monkeypatch.setenv(key, "provider-secret")
    monkeypatch.setenv(base_url, "https://provider.example/v1")
    for name, value in model_vars.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.provider == provider
    assert settings.llm_api_key == "provider-secret"
    assert settings.llm_base_url == "https://provider.example/v1"
    assert (settings.llm_model, settings.vlm_model) == expected_models
    get_settings.cache_clear()


def test_get_settings_requires_selected_profile_values(monkeypatch):
    monkeypatch.setattr(settings_module, "ACTIVE_MODEL_PROVIDER", "pod")
    for name in ("POD_API_KEY", "POD_BASE_URL", "POD_MODEL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    with pytest.raises(LlmUnavailableError, match="POD_API_KEY"):
        get_settings()

    get_settings.cache_clear()


def test_get_settings_can_select_openrouter_for_urgency_only(monkeypatch):
    monkeypatch.setattr(settings_module, "ACTIVE_MODEL_PROVIDER", "pod")
    for name, value in {
        "OPENROUTER_API_KEY": "openrouter-key",
        "OPENROUTER_BASE_URL": "https://openrouter.example/v1",
        "OPENROUTER_LLM_MODEL": "openrouter-default",
        "OPENROUTER_VLM_MODEL": "openrouter-vlm",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()

    settings = get_settings("openrouter")

    assert settings.provider == "openrouter"
    assert settings.llm_model == "openrouter-default"
    get_settings.cache_clear()


def test_get_settings_rejects_non_https_provider_url(monkeypatch):
    monkeypatch.setattr(settings_module, "ACTIVE_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://provider.example/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "llm")
    monkeypatch.setenv("OPENROUTER_VLM_MODEL", "vlm")
    get_settings.cache_clear()

    with pytest.raises(LlmUnavailableError, match="HTTPS"):
        get_settings()

    get_settings.cache_clear()


def test_pod_settings_follow_locator_address_changes(monkeypatch):
    monkeypatch.setattr(settings_module, "ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setenv("POD_API_KEY", "pod-key")
    monkeypatch.setenv("POD_MODEL", "pod-model")
    monkeypatch.delenv("POD_BASE_URL", raising=False)
    locator = SimpleNamespace(base_url="https://first-8000.proxy.runpod.net/v1")
    monkeypatch.setattr(settings_module, "get_locator", lambda: locator)

    first = get_settings()
    assert first.llm_base_url == locator.base_url
    assert get_settings() is first

    locator.base_url = "https://second-8000.proxy.runpod.net/v1"
    second = get_settings()
    assert second.llm_base_url == locator.base_url
    assert second is not first

    locator.base_url = None
    with pytest.raises(LlmUnavailableError, match="POD_BASE_URL"):
        get_settings()


def test_other_providers_do_not_access_locator(monkeypatch):
    monkeypatch.setattr(
        settings_module, "get_locator", lambda: pytest.fail("locator accessed")
    )

    assert get_settings("serverless").provider == "serverless"
    assert get_settings("openrouter").provider == "openrouter"
