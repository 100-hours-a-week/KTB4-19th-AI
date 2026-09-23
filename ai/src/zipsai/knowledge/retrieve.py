import logging
from functools import lru_cache

from qdrant_client import QdrantClient, models

from zipsai.indexing.embed import Encoder
from zipsai.integrations.embedding_client import HttpEncoder
from zipsai.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    building_condition,
    to_sparse_vector,
)
from zipsai.settings import QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# 질의는 입주민이 화면 앞에서 기다리는 경로다. 색인이 쓰는 기본 120초는 32개 묶음을
# 한 번에 보내는 배치 기준이라, 그대로 쓰면 백엔드 타임아웃이 먼저 터진다.
QUERY_TIMEOUT_SECONDS = 5.0

# v1 초기값. 측정 전이라 흔한 값을 그대로 쓴다. 정답셋 평가에서 이 세 줄을 조정한다.
SCORE_THRESHOLD = 0.5
PREFETCH_LIMIT = 20
TOP_K = 5

QueryVector = tuple[list[float], dict[str, float]]


@lru_cache(maxsize=1)
def query_encoder() -> HttpEncoder:
    # 첫 질의가 들어올 때 만든다. 임포트 시점에 embedding 컨테이너로 붙지 않는다.
    return HttpEncoder(timeout=QUERY_TIMEOUT_SECONDS)


def encode_question(question: str, *, encoder: Encoder) -> QueryVector:
    """질문 하나를 색인과 같은 bge-m3로 dense·sparse 벡터로 만든다."""
    dense, sparse = encoder.encode([question])
    return dense[0], sparse[0]


def search_chunks(
    question_vector: QueryVector,
    building_id: int,
    *,
    client: QdrantClient,
    collection: str = QDRANT_COLLECTION,
) -> list[models.ScoredPoint]:
    """건물 문서에서 질문과 가까운 청크를 회수한다. 근거가 없으면 빈 목록이다."""
    dense, sparse = question_vector
    building_filter = models.Filter(must=[building_condition(building_id)])

    # 게이트를 먼저 통과시킨다. 실패가 성공보다 빠르고 싸야 하므로,
    # dense 한 건도 임계값을 못 넘으면 하이브리드도 LLM도 돌리지 않는다.
    # RRF 점수는 코사인이 아니라서 융합 뒤에는 이 임계값을 걸 수 없다.
    gate = client.query_points(
        collection_name=collection,
        query=dense,
        using=DENSE_VECTOR,
        query_filter=building_filter,
        limit=1,
        score_threshold=SCORE_THRESHOLD,
        with_payload=False,
    )
    if not gate.points:
        logger.info(
            "knowledge_search_gated building_id=%s threshold=%.2f",
            building_id,
            SCORE_THRESHOLD,
        )
        return []

    hits = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=dense,
                using=DENSE_VECTOR,
                filter=building_filter,
                limit=PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=to_sparse_vector(sparse),
                using=SPARSE_VECTOR,
                filter=building_filter,
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    ).points

    for hit in hits:
        payload = hit.payload or {}
        logger.info(
            "knowledge_search_hit building_id=%s doc_id=%s score=%.4f text=%.40s",
            building_id,
            payload.get("doc_id"),
            hit.score,
            payload.get("text", ""),
        )
    return hits
