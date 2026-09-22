from dataclasses import dataclass
from typing import Protocol

from zipsai.errors import EmbeddingError
from zipsai.indexing.chunk import Chunk
from zipsai.settings import EMBEDDING_DIM

EncodeOutput = tuple[list[list[float]], list[dict[str, float]]]


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> EncodeOutput: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    text: str
    page: int
    section: str | None
    dense: list[float]
    sparse: dict[str, float]


def embed_chunks(chunks: list[Chunk], encoder: Encoder) -> list[EmbeddedChunk]:
    selected = [chunk for chunk in chunks if chunk.text.strip()]
    if not selected:
        return []

    dense, sparse = encoder.encode([chunk.text for chunk in selected])
    if len(dense) != len(selected) or len(sparse) != len(selected):
        raise EmbeddingError(
            f"Encoder returned {len(dense)} dense and {len(sparse)} sparse results "
            f"for {len(selected)} texts"
        )
    if any(len(vector) != EMBEDDING_DIM for vector in dense):
        raise EmbeddingError(f"Dense embeddings must have length {EMBEDDING_DIM}")

    return [
        EmbeddedChunk(
            text=chunk.text,
            page=chunk.page,
            section=chunk.section,
            dense=vector,
            sparse=weights,
        )
        for chunk, vector, weights in zip(selected, dense, sparse, strict=True)
    ]
