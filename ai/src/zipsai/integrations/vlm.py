from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel

from zipsai.contracts.converse import ImageAnalysis, ImageAttachment, ImageObservation, Route
from zipsai.errors import (
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations.llm import _REQUIRE_STRUCTURED_OUTPUTS, _get_client
from zipsai.settings import get_settings
from zipsai.tracing import get_tracing_client


class _ModelObservation(BaseModel):
    summary: str | None
    ocr_text: str | None


class _ModelImages(BaseModel):
    images: list[_ModelObservation]


class _ModelIntentAndImages(_ModelImages):
    route: Route


def classify_and_analyze(
    images: list[ImageAttachment], *, system_prompt: str, user_prompt: str
) -> tuple[Route, ImageAnalysis]:
    settings = get_settings()
    content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image.url}}
        for image in images
        if image.url
    )
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.parse(
            model=settings.vlm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            **({"name": "intent-and-image-analysis"} if get_tracing_client() else {}),
            response_format=_ModelIntentAndImages,
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

    if not response.choices or response.choices[0].message.parsed is None:
        raise ImageAnalysisError("VLM returned an invalid intent or image analysis")
    if response.choices[0].message.refusal:
        raise ImageAnalysisError("VLM refused intent or image analysis")
    parsed = response.choices[0].message.parsed
    if len(parsed.images) != len(images):
        raise ImageAnalysisError("VLM returned observations that do not match input images")
    return parsed.route, ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=image.attachment_id,
                summary=observation.summary,
                ocrText=observation.ocr_text,
            )
            for image, observation in zip(images, parsed.images)
        ]
    )
