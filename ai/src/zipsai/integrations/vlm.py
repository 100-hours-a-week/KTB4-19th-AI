from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel

from zipsai.contracts.converse import (
    ImageAnalysis,
    ImageAttachment,
    ImageObservation,
    Route,
)
from zipsai.errors import (
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations.llm import (
    _REQUIRE_STRUCTURED_OUTPUTS,
    _get_client,
    _reason,
    _retry_after,
    start_llm_call,
)
from zipsai.settings import get_settings
from zipsai.tracing import get_tracing_client


class _ModelObservation(BaseModel):
    summary: str | None
    ocr_text: str | None


class _ModelImages(BaseModel):
    images: list[_ModelObservation]


class _ModelIntentAndImages(_ModelImages):
    route: Route
    contextualized_message: str | None


def classify_and_analyze(
    images: list[ImageAttachment], *, system_prompt: str, user_prompt: str
) -> tuple[Route, ImageAnalysis, str | None]:
    settings = get_settings()
    model = settings.vlm_model if images else settings.llm_model
    content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image.url}}
        for image in images
        if image.url
    )
    call = start_llm_call(
        purpose="image_analysis" if images else "intent",
        model=model,
        provider=settings.provider,
        base_url=settings.llm_base_url,
        streaming=False,
        structured=True,
        image_count=len(images),
    )
    outcome = "fail"
    status_code: int | None = None
    err_code: str | None = None
    finish_reason: str | None = None
    structured_ok: bool | None = True
    try:
        response = _get_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
        ).chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            **({"name": "intent-and-image-analysis"} if get_tracing_client() else {}),
            response_format=_ModelIntentAndImages,
            extra_body=_REQUIRE_STRUCTURED_OUTPUTS,
        )
        call.usage = getattr(response, "usage", None)
        if not response.choices or response.choices[0].message.parsed is None:
            outcome = "fail"
            structured_ok = False
            err_code = "structured_output"
            raise ImageAnalysisError("VLM returned an invalid intent or image analysis")
        if response.choices[0].message.refusal:
            outcome = "fail"
            structured_ok = False
            err_code = "structured_output"
            raise ImageAnalysisError("VLM refused intent or image analysis")
        finish_reason = _reason(getattr(response.choices[0], "finish_reason", None))
        outcome = "ok"
        status_code = 200
    except RateLimitError as error:
        outcome, status_code, err_code, structured_ok = "fail", 429, "MODEL_RATE_LIMITED", None
        raise LlmRateLimitedError(
            "VLM rate limit exceeded", retry_after_seconds=_retry_after(error)
        ) from None
    except APITimeoutError:
        outcome, status_code, err_code, structured_ok = "fail", None, "MODEL_TIMEOUT", None
        raise LlmTimeoutError("VLM request timed out") from None
    except APIStatusError as error:
        outcome = "fail"
        status_code = error.status_code
        structured_ok = None
        err_code = "MODEL_UPSTREAM_ERROR" if error.status_code >= 500 else "DEPENDENCY_NOT_READY"
        if error.status_code >= 500:
            raise LlmUpstreamError("VLM provider returned an upstream error") from None
        raise LlmUnavailableError("VLM request failed") from None
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
        outcome = "fail"
        structured_ok = False
        finish_reason = (
            "length" if isinstance(error, LengthFinishReasonError) else "content_filter"
        )
        err_code = type(error).__name__
        raise ImageAnalysisError("VLM response was truncated or filtered") from None
    except APIError:
        outcome, status_code, err_code, structured_ok = "fail", None, "DEPENDENCY_NOT_READY", None
        raise LlmUnavailableError("VLM request failed") from None
    except Exception as error:
        outcome = "fail"
        if err_code is None:
            status_code = None
            err_code = type(error).__name__
            structured_ok = None
        raise
    finally:
        call.finish(
            outcome=outcome,
            status_code=status_code,
            error_code=err_code,
            finish_reason=finish_reason,
            structured_output_ok=structured_ok,
        )
    parsed = response.choices[0].message.parsed
    if not images:
        return parsed.route, ImageAnalysis(images=[]), parsed.contextualized_message
    if len(parsed.images) != len(images):
        raise ImageAnalysisError(
            "VLM returned observations that do not match input images"
        )
    return (
        parsed.route,
        ImageAnalysis(
            images=[
                ImageObservation(
                    attachmentId=image.attachment_id,
                    summary=observation.summary,
                    ocrText=observation.ocr_text,
                )
                for image, observation in zip(images, parsed.images)
            ]
        ),
        parsed.contextualized_message,
    )
