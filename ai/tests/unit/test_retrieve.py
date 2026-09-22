from zipsai.integrations.embedding_client import DEFAULT_TIMEOUT_SECONDS
from zipsai.knowledge.retrieve import (
    QUERY_TIMEOUT_SECONDS,
    encode_question,
    query_encoder,
)
from zipsai.settings import EMBEDDING_DIM


class FakeEncoder:
    def __init__(
        self,
        dense: list[list[float]] | None = None,
        sparse: list[dict[str, float]] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._dense = dense
        self._sparse = sparse

    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        self.calls.append(texts)
        dense = self._dense or [[0.0] * EMBEDDING_DIM for _ in texts]
        sparse = self._sparse or [{} for _ in texts]
        return dense, sparse


def test_encode_question_sends_one_text_and_unwraps_both_vectors() -> None:
    encoder = FakeEncoder(dense=[[0.5] * EMBEDDING_DIM], sparse=[{"7": 0.25}])

    dense, sparse = encode_question("세탁실은 몇 시까지 쓸 수 있나요?", encoder=encoder)

    assert encoder.calls == [["세탁실은 몇 시까지 쓸 수 있나요?"]]
    assert dense == [0.5] * EMBEDDING_DIM
    assert sparse == {"7": 0.25}


def test_query_encoder_waits_far_less_than_the_indexing_batch() -> None:
    # 배치용 기본값을 그대로 물려받으면 백엔드가 먼저 요청을 끊는다.
    assert QUERY_TIMEOUT_SECONDS < DEFAULT_TIMEOUT_SECONDS
    assert query_encoder()._client.timeout.read == QUERY_TIMEOUT_SECONDS
