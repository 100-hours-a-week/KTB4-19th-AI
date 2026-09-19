import httpx
import pytest

from zipsai.errors import EmbeddingError
from zipsai.integrations.embedding_client import HttpEncoder


def encoder_returning(handler) -> HttpEncoder:
    return HttpEncoder(
        base_url="http://embedding:8000",
        transport=httpx.MockTransport(handler),
    )


def test_encode_posts_texts_and_unpacks_dense_and_sparse() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": request.read().decode()})
        return httpx.Response(
            200,
            json={"dense": [[0.1, 0.2]], "sparse": [{"7": 0.5}]},
        )

    dense, sparse = encoder_returning(handler).encode(["첫 청크"])

    assert dense == [[0.1, 0.2]]
    assert sparse == [{"7": 0.5}]
    assert seen[0]["url"] == "http://embedding:8000/embed"
    assert "첫 청크" in seen[0]["body"]


def test_encode_raises_typed_error_on_server_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model is still loading"})

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])


def test_encode_raises_typed_error_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])


@pytest.mark.parametrize("body", [{}, {"dense": [[0.1]]}, []])
def test_encode_raises_typed_error_on_malformed_response(body: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(EmbeddingError):
        encoder_returning(handler).encode(["text"])
