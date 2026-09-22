from functools import lru_cache

from zipsai.indexing.embed import Encoder
from zipsai.integrations.embedding_client import HttpEncoder

# 질의는 입주민이 화면 앞에서 기다리는 경로다. 색인이 쓰는 기본 120초는 32개 묶음을
# 한 번에 보내는 배치 기준이라, 그대로 쓰면 백엔드 타임아웃이 먼저 터진다.
QUERY_TIMEOUT_SECONDS = 5.0

QueryVector = tuple[list[float], dict[str, float]]


@lru_cache(maxsize=1)
def query_encoder() -> HttpEncoder:
    # 첫 질의가 들어올 때 만든다. 임포트 시점에 embedding 컨테이너로 붙지 않는다.
    return HttpEncoder(timeout=QUERY_TIMEOUT_SECONDS)


def encode_question(question: str, *, encoder: Encoder) -> QueryVector:
    """질문 하나를 색인과 같은 bge-m3로 dense·sparse 벡터로 만든다."""
    dense, sparse = encoder.encode([question])
    return dense[0], sparse[0]
