import os

import pytest

from embedding import encoder as encoder_mod
from embedding.encoder import configure_torch_threads, thread_snapshot


def test_unset_torch_threads_does_not_call_set_num_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_TORCH_THREADS", raising=False)
    calls: list[int] = []

    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(value))

    configure_torch_threads()

    assert calls == []


def test_torch_threads_env_is_applied_before_the_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_TORCH_THREADS", "2")
    calls: list[int] = []

    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(value))

    configure_torch_threads()

    assert calls == [2]
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(encoder_mod.encoder, "_max_index_batch", 4)
    assert thread_snapshot() == {
        "torch_threads": 2,
        "cpu_count": os.cpu_count(),
        "max_index_batch": 4,
    }


@pytest.mark.parametrize("raw", ["0", "-3", "1.5", "많음"])
def test_torch_threads_rejects_non_positive_integers(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("EMBEDDING_TORCH_THREADS", raw)

    with pytest.raises(RuntimeError, match="EMBEDDING_TORCH_THREADS"):
        configure_torch_threads()
