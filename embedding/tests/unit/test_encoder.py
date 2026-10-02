import numpy as np
import pytest

from embedding.encoder import BgeM3Encoder, ModelBusyError, _normalize


class FakeModel:
    def encode(self, texts: list[str], **_: object) -> dict:
        return {
            "dense_vecs": np.array([[0.1] for _ in texts]),
            "lexical_weights": [{} for _ in texts],
        }


def test_encode_raises_busy_when_lock_held_past_max_wait() -> None:
    encoder = BgeM3Encoder()
    encoder._model = FakeModel()

    encoder._lock.acquire()  # 다른 배치가 락을 쥐고 있다고 가정한다
    try:
        with pytest.raises(ModelBusyError):
            encoder.encode(["텍스트"], max_wait=0.05)
    finally:
        encoder._lock.release()


def test_encode_waits_indefinitely_when_max_wait_is_none() -> None:
    encoder = BgeM3Encoder()
    encoder._model = FakeModel()

    dense, _ = encoder.encode(["텍스트"])

    assert dense == [[0.1]]


def test_normalize_converts_numpy_outputs_to_python_types() -> None:
    dense, sparse = _normalize(
        {
            "dense_vecs": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "lexical_weights": [
                {"one": np.float32(0.5)},
                {"two": np.float64(0.25)},
            ],
        }
    )

    assert dense == [[1.0, 2.0], [3.0, 4.0]]
    assert sparse == [{"one": 0.5}, {"two": 0.25}]
    assert all(isinstance(value, float) for row in dense for value in row)
    assert all(
        isinstance(value, float) for weights in sparse for value in weights.values()
    )
