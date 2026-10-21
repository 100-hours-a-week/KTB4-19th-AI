import pytest

from zipsai.settings import get_settings


@pytest.fixture(autouse=True)
def _fake_openrouter_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings()는 ACTIVE_MODEL_PROVIDER=openrouter 기준 네 값을 필수로 본다.
    로컬 .env가 있으면 가려지지만 CI나 새 체크아웃엔 없어 실패한다 — 가짜 값으로 고정해
    결과가 로컬/CI 어디서나 같게 한다. 실제 OpenRouter를 부르는 테스트는 없다."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "test-llm-model")
    monkeypatch.setenv("OPENROUTER_VLM_MODEL", "test-vlm-model")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
