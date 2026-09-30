from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel

from zipsai.contracts.converse import ImageAnalysis, ImageObservation
from zipsai.errors import (
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations.llm import _REQUIRE_STRUCTURED_OUTPUTS, _get_client
from zipsai.settings import get_settings


class _ModelObservation(BaseModel):
    summary: str | None
    ocr_text: str | None


class _ModelImages(BaseModel):
    images: list[_ModelObservation]


def analyze_images(image_urls: list[str], prompt: str) -> ImageAnalysis:
    settings = get_settings()
    image_content = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.parse(
            model=settings.vlm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": image_content},
            ],
            response_format=_ModelImages,
            extra_body=_REQUIRE_STRUCTURED_OUTPUTS,
        )
    except RateLimitError as error:
        raise LlmRateLimitedError("VLM rate limit exceeded") from error
    except APITimeoutError as error:
        raise LlmTimeoutError("VLM request timed out") from error
    except APIStatusError as error:
        if error.status_code >= 500:
            raise LlmUpstreamError("VLM provider returned an upstream error") from error
        raise LlmUnavailableError("VLM request failed") from error
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
        raise ImageAnalysisError("VLM response was truncated or filtered") from error
    except APIError as error:
        raise LlmUnavailableError("VLM request failed") from error

    if not response.choices:
        raise ImageAnalysisError("VLM returned an empty response")

    message = response.choices[0].message
    if message.refusal:
        raise ImageAnalysisError(f"VLM refused to analyze the image: {message.refusal}")
    parsed = message.parsed
    if parsed is None:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    if len(parsed.images) != len(image_urls):
        raise ImageAnalysisError(
            "VLM returned observations that do not match input images"
        )

    # 모델 출력에는 URL을 요구하지 않고, 입력 순서로 원본 URL을 되붙인다.
    return ImageAnalysis(
        images=[
            ImageObservation(url=url, summary=obs.summary, ocr_text=obs.ocr_text)
            for url, obs in zip(image_urls, parsed.images)
        ]
    )
