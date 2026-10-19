"""Check whether the configured provider reuses the complaint prompt prefix.

Run from the repository root with PYTHONPATH=ai/src.
The probe prints usage metadata only, never prompts or model responses.
"""

import json
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from zipsai.complaint.node import _EXTRACTION_SCHEMA
from zipsai.complaint.prompts import COMPLAINT_PROMPT
from zipsai.integrations.llm import generate_text

_MESSAGES = (
    "화장실 천장에서 물이 떨어져요.",
    "주방 싱크대 아래에서 물이 새요.",
    "베란다 조명이 켜지지 않아요.",
)


def main() -> None:
    today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    for index, message in enumerate(_MESSAGES, start=1):
        system, user = COMPLAINT_PROMPT.format_messages(
            today=today,
            conversation_history="",
            complaint_draft=None,
            message_text=message,
        )
        usage: dict[str, object] = {}
        started_at = perf_counter()
        generate_text(
            str(system.content),
            str(user.content),
            response_format=_EXTRACTION_SCHEMA,
            usage_sink=usage,
        )
        print(
            json.dumps(
                {
                    "call": index,
                    "elapsed_ms": int((perf_counter() - started_at) * 1000),
                    **usage,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
