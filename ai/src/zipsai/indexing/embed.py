from dataclasses import dataclass
from typing import Protocol

from zipsai.errors import ChunkTooLongError, EmbeddingError
from zipsai.indexing.chunk import Chunk
from zipsai.integrations.embedding_client import MAX_TEXT_CHARS
from zipsai.settings import EMBEDDING_DIM

EncodeOutput = tuple[list[list[float]], list[dict[str, float]]]


class Encoder(Protocol):
    def encode(self, texts: list[str], trace_id: str) -> EncodeOutput: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    text: str
    page: int
    section: str | None
    dense: list[float]
    sparse: dict[str, float]


def count_embed_texts(chunks: list[Chunk]) -> tuple[int, int]:
    """공백만인 청크를 뺀 전체 개수와 서로 다른 글 개수. embed_chunks와 같은 기준이다."""
    texts = [chunk.text for chunk in chunks if chunk.text.strip()]
    return len(texts), len(set(texts))


def embed_chunks(
    chunks: list[Chunk], encoder: Encoder, trace_id: str
) -> list[EmbeddedChunk]:
    selected = [chunk for chunk in chunks if chunk.text.strip()]
    if not selected:
        return []

    # 빈 청크는 서버로 가지 않으므로 번호는 전송 목록 기준이다.
    for index, chunk in enumerate(selected):
        length = len(chunk.text)
        if length > MAX_TEXT_CHARS:
            raise ChunkTooLongError(index, length, MAX_TEXT_CHARS)

    dense, sparse = encoder.encode([chunk.text for chunk in selected], trace_id)
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
