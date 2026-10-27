import pytest

from zipsai.integrations.runpod import get_locator
from zipsai.settings import get_settings


@pytest.fixture(autouse=True)
def _fake_openrouter_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """기본 serverless provider 설정을 가짜 값으로 고정해 로컬/CI 결과를 맞춘다."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "test-llm-model")
    monkeypatch.setenv("OPENROUTER_VLM_MODEL", "test-vlm-model")
    monkeypatch.setenv("SERVERLESS_API_KEY", "test-serverless-key")
    monkeypatch.setenv("SERVERLESS_BASE_URL", "https://serverless.example/v1")
    monkeypatch.setenv("SERVERLESS_MODEL", "test-serverless-model")
    # settings가 load_dotenv()를 하므로 개발자 .env의 실제 RunPod 키가 새어 들어올 수
    # 있다. 지우지 않으면 테스트가 진짜 RunPod API를 때린다. 쓰는 테스트가 직접 넣는다.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_NAME_PREFIX", raising=False)
    get_settings.cache_clear()
    get_locator.cache_clear()
    yield
    get_settings.cache_clear()
    get_locator.cache_clear()
