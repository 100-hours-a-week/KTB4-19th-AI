"""generate_text_stream 단독 확인용 스크립트.

실행:
    uv run python scripts/test_llm_stream.py "짧게 자기소개해줘"
"""

import asyncio
import sys
import time

from zipsai.integrations.llm import generate_text_stream


async def main(user_prompt: str) -> None:
    started_at = time.perf_counter()
    chunk_count = 0

    print("[stream start]")
    async for text in generate_text_stream(
        system_prompt="짧고 자연스럽게 답변해줘.",
        user_prompt=user_prompt,
    ):
        chunk_count += 1
        print(text, end="", flush=True)

    elapsed = time.perf_counter() - started_at
    print(f"\n\n[stream done: chunks={chunk_count}, seconds={elapsed:.2f}]")


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "SSE 스트리밍을 한 문장으로 설명해줘."
    asyncio.run(main(prompt))
