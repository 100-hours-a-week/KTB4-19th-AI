import json
import logging
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from zipsai.contracts.notification import Notification
from zipsai.settings import AWS_REGION, NOTIFICATION_QUEUE_URL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _default_client() -> Any:
    return boto3.client("sqs", region_name=AWS_REGION)


def send_notification(notification: Notification, *, client: Any | None = None) -> None:
    """처리 결과를 BE 큐에 넣는다. 실패해도 예외를 올리지 않는다.

    이미 작업은 끝났고 호출자에게 되돌려줄 곳이 없다. 로그가 유일한 흔적이다.
    """
    if NOTIFICATION_QUEUE_URL is None:
        # 운영에서 값을 빠뜨리면 실패도 BE에 알려지지 않는다. 로컬은 정상이라 경고로 눈에만 띄게 한다.
        logger.warning(
            "notification_skipped",
            extra={"kind": notification.kind.value, "trace_id": notification.trace_id},
        )
        return
    try:
        (client or _default_client()).send_message(
            QueueUrl=NOTIFICATION_QUEUE_URL,
            MessageBody=json.dumps(notification.to_message(), ensure_ascii=False),
        )
    except (BotoCoreError, ClientError) as error:
        logger.exception(
            "notification_failed",
            extra={
                "kind": notification.kind.value,
                "trace_id": notification.trace_id,
                "error_type": type(error).__name__,
            },
        )
