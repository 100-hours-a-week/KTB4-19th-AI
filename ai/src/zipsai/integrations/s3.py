import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from zipsai.errors import DocumentFetchError
from zipsai.settings import AWS_REGION, S3_BUCKET

# 가상 호스팅 형식: bucket.s3.region.amazonaws.com / bucket.s3.amazonaws.com
_VIRTUAL_HOST = re.compile(r"^(?P<bucket>[^.]+)\.s3[.-][^.]*\.?amazonaws\.com$")
# 경로 형식: s3.region.amazonaws.com/bucket/key
_PATH_HOST = re.compile(r"^s3[.-][^.]*\.?amazonaws\.com$")


@lru_cache(maxsize=1)
def _default_client() -> Any:
    # 자격증명은 EC2 인스턴스 프로파일에서 온다. boto3 기본 탐색 순서를 그대로 쓴다.
    return boto3.client("s3", region_name=AWS_REGION)


def parse_s3_location(file_key: str) -> tuple[str | None, str]:
    """버킷과 키를 뽑는다. 백엔드가 어떤 URL 형식을 쓸지 정해지지 않아 셋 다 받는다.

    스킴이 없으면 예전처럼 객체 키로 보고 버킷은 설정값을 따른다.
    """
    parsed = urlparse(file_key)

    if parsed.scheme == "s3":
        return parsed.netloc or None, unquote(parsed.path).lstrip("/")

    if parsed.scheme in ("http", "https"):
        key = unquote(parsed.path).lstrip("/")
        virtual = _VIRTUAL_HOST.match(parsed.netloc)
        if virtual:
            return virtual.group("bucket"), key
        if _PATH_HOST.match(parsed.netloc):
            bucket, _, rest = key.partition("/")
            return bucket or None, rest
        # 사설 도메인이면 버킷을 알 수 없다. 경로만 키로 쓴다.
        return None, key

    return None, file_key


def download(
    file_key: str,
    target_dir: Path,
    *,
    bucket: str | None = None,
    client: Any | None = None,
) -> Path:
    parsed_bucket, key = parse_s3_location(file_key)
    target_bucket = parsed_bucket or bucket or S3_BUCKET

    if not target_bucket:
        raise DocumentFetchError("S3_BUCKET is not configured")
    # URL에 실린 버킷을 그대로 믿으면 남의 버킷도 읽게 된다.
    if S3_BUCKET and target_bucket != S3_BUCKET:
        raise DocumentFetchError(f"Refusing to read from bucket '{target_bucket}'")
    if not key:
        raise DocumentFetchError(f"No object key in '{file_key}'")

    # 키에 상위 경로가 섞여도 target_dir 밖으로 나가지 않게 파일명만 쓴다.
    target = target_dir / Path(key).name
    try:
        (client or _default_client()).download_file(target_bucket, key, str(target))
    except (BotoCoreError, ClientError, OSError) as exc:
        raise DocumentFetchError(f"Failed to fetch '{file_key}': {exc}") from exc
    return target
