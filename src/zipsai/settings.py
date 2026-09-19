import os
from typing import Final

API_PREFIX: Final = "/api/v3/ai/indexing"
EMBEDDING_DIM: Final = 1024

# 컨테이너에서는 Compose가 주입한다. 기본값은 테스트가 쓰는 로컬 설정이다.
QDRANT_URL: Final = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION: Final = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_API_URL: Final = os.getenv("EMBEDDING_API_URL", "http://embedding:8000")
S3_BUCKET: Final = os.getenv("S3_BUCKET")
AWS_REGION: Final = os.getenv("AWS_REGION", "ap-northeast-2")
