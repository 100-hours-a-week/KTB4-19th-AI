import pytest

from zipsai.errors import EmbeddingError
from zipsai.indexing.chunk import Chunk
from zipsai.indexing.embed import EmbeddedChunk, embed_chunks
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


def test_embed_chunks_preserves_chunk_metadata_and_embedding_values() -> None:
    chunks = [
        Chunk(text="first", page=1, section="intro"),
        Chunk(text="second", page=2, section=None),
    ]
    encoder = FakeEncoder(
        dense=[[float(index)] * EMBEDDING_DIM for index in range(2)],
        sparse=[{"first": 1.0}, {"second": 0.5}],
    )

    embedded = embed_chunks(chunks, encoder)

    assert all(isinstance(chunk, EmbeddedChunk) for chunk in embedded)
    assert [(chunk.text, chunk.page, chunk.section) for chunk in embedded] == [
        ("first", 1, "intro"),
        ("second", 2, None),
    ]
    assert [chunk.dense[0] for chunk in embedded] == [0.0, 1.0]
    assert [chunk.sparse for chunk in embedded] == [{"first": 1.0}, {"second": 0.5}]
    assert all(len(chunk.dense) == EMBEDDING_DIM for chunk in embedded)


def test_embed_chunks_calls_encoder_once_with_all_texts() -> None:
    encoder = FakeEncoder()

    embed_chunks(
        [Chunk("first", 1, None), Chunk("second", 1, None)],
        encoder,
    )

    assert encoder.calls == [["first", "second"]]


def test_embed_chunks_returns_empty_without_calling_encoder_for_empty_input() -> None:
    encoder = FakeEncoder()

    assert embed_chunks([], encoder) == []
    assert encoder.calls == []


def test_embed_chunks_drops_whitespace_only_chunks() -> None:
    encoder = FakeEncoder()

    embedded = embed_chunks(
        [Chunk(" \n\t", 1, "discarded"), Chunk("kept", 2, None)],
        encoder,
    )

    assert [chunk.text for chunk in embedded] == ["kept"]
    assert encoder.calls == [["kept"]]


def test_embed_chunks_rejects_wrong_dense_dimension() -> None:
    encoder = FakeEncoder(dense=[[0.0] * (EMBEDDING_DIM - 1)], sparse=[{}])

    with pytest.raises(EmbeddingError):
        embed_chunks([Chunk("text", 1, None)], encoder)


def test_embed_chunks_rejects_fewer_results_than_texts() -> None:
    encoder = FakeEncoder(dense=[[0.0] * EMBEDDING_DIM], sparse=[{}])

    with pytest.raises(EmbeddingError):
        embed_chunks(
            [Chunk("first", 1, None), Chunk("second", 1, None)],
            encoder,
        )
