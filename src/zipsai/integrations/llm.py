from functools import lru_cache

from openai import APIError, OpenAI

from zipsai.errors import LlmUnavailableError
from zipsai.settings import get_settings


def generate_text(system_prompt: str, user_prompt: str) -> str:
    settings = get_settings()
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIError as error:
        raise LlmUnavailableError("LLM request failed") from error

    content = response.choices[0].message.content
    if not content:
        raise LlmUnavailableError("LLM returned an empty response")
    return content


@lru_cache
def _get_client(api_key: str, base_url: str | None, timeout_seconds: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
