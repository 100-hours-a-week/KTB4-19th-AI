import pytest

import zipsai.integrations.qdrant as qdrant_module
from zipsai.integrations.qdrant import create_client


def test_create_client_sends_the_api_key_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qdrant에 인증이 켜지면 키 없이 붙은 요청은 전부 401이 된다.
    monkeypatch.setattr(qdrant_module, "QDRANT_API_KEY", "admin-key")

    client = create_client("http://qdrant:6333")

    assert client._client._api_key == "admin-key"


def test_create_client_connects_without_a_key_when_auth_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 로컬과 CI의 Qdrant에는 인증이 없다. 키를 필수로 만들면 거기서 기동이 막힌다.
    monkeypatch.setattr(qdrant_module, "QDRANT_API_KEY", None)

    client = create_client("http://qdrant:6333")

    assert client._client._api_key is None


def test_create_client_ignores_the_key_for_in_memory_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ":memory:"는 프로세스 안의 저장소라 인증할 상대가 없다.
    monkeypatch.setattr(qdrant_module, "QDRANT_API_KEY", "admin-key")

    assert create_client(":memory:") is not None


def test_create_client_refuses_to_start_without_a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qdrant_module, "QDRANT_URL", None)

    with pytest.raises(RuntimeError):
        create_client()
