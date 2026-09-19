from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from zipsai.errors import DocumentFetchError
from zipsai.settings import AWS_REGION, S3_BUCKET


@lru_cache(maxsize=1)
def _default_client() -> Any:
    # 자격증명은 EC2 인스턴스 프로파일에서 온다. boto3 기본 탐색 순서를 그대로 쓴다.
    return boto3.client("s3", region_name=AWS_REGION)


def download(
    file_key: str,
    target_dir: Path,
    *,
    bucket: str | None = S3_BUCKET,
    client: Any | None = None,
) -> Path:
    if not bucket:
        raise DocumentFetchError("S3_BUCKET is not configured")

    # file_key에 상위 경로가 섞여도 target_dir 밖으로 나가지 않게 파일명만 쓴다.
    target = target_dir / Path(file_key).name
    try:
        (client or _default_client()).download_file(bucket, file_key, str(target))
    except (BotoCoreError, ClientError, OSError) as exc:
        raise DocumentFetchError(f"Failed to fetch '{file_key}': {exc}") from exc
    return target
