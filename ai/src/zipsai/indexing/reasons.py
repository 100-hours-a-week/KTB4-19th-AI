import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 성공(None)은 호출부에서 false로 둔다. 표에 없는 코드도 false다.
RETRYABLE: dict[str, bool] = {
    "FILE_DOWNLOAD_FAILED": True,
    "SETUP_FAILED": True,
    "EMBEDDING_FAILED": True,
    "VECTOR_STORE_FAILED": True,
    "CLEANUP_FAILED": True,
    "PARSE_FAILED": False,
    "EMPTY_DOCUMENT": False,
    "MASK_FAILED": False,
    "INDEX_FAILED": False,
    "COMMENT_TOO_SHORT": False,
    "VALIDATION_FAILED": False,
    "STALE_UPDATE": False,
    "CLEAN_FAILED": False,
    "CHUNK_FAILED": False,
}

# 소스에 적힌 reason_code 리터럴. 환경 변수 이름이나 다른 오류 코드는 빠진다.
_LITERAL = re.compile(
    r'"(?:(?:[A-Z]+_)+(?:FAILED|UPDATE)|EMPTY_DOCUMENT|COMMENT_TOO_SHORT)"'
)


def retryable_for(reason_code: str | None) -> bool:
    if reason_code is None:
        return False
    try:
        return RETRYABLE[reason_code]
    except KeyError:
        logger.warning("unknown_reason_code", extra={"reason_code": reason_code})
        return False


def reason_literals(source_root: Path) -> set[str]:
    found: set[str] = set()
    for path in source_root.rglob("*.py"):
        for match in _LITERAL.finditer(path.read_text(encoding="utf-8")):
            found.add(match.group(0)[1:-1])
    return found
