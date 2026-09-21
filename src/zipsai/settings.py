import os
from typing import Final

API_PREFIX: Final = "/api/v3/ai/indexing"
EMBEDDING_DIM: Final = 1024

# 기본값을 두지 않는다. 빠뜨린 채 떠서 조용히 다른 저장소를 쓰느니
# 기동하지 못하는 편이 낫다 — 검사는 main.py의 lifespan이 한다.
QDRANT_URL: Final = os.getenv("QDRANT_URL")
S3_BUCKET: Final = os.getenv("S3_BUCKET")

QDRANT_COLLECTION: Final = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_API_URL: Final = os.getenv("EMBEDDING_API_URL", "http://embedding:8000")
AWS_REGION: Final = os.getenv("AWS_REGION", "ap-northeast-2")

REQUIRED_SETTINGS: Final = ("QDRANT_URL", "S3_BUCKET")


def missing_required_settings() -> list[str]:
    values = {"QDRANT_URL": QDRANT_URL, "S3_BUCKET": S3_BUCKET}
    return [name for name in REQUIRED_SETTINGS if not values[name]]
