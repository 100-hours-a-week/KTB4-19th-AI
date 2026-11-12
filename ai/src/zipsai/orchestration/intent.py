import logging

from pydantic import BaseModel

from zipsai.contracts.converse import Route
from zipsai.errors import (
    ImageAnalysisError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.history import format_intent_history
from zipsai.integrations.llm import generate_structured
from zipsai.integrations.vlm import analyze_images
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import IMAGE_OBSERVATION_PROMPT, INTENT_PROMPT
from zipsai.orchestration.state import AgentState, ToolName

logger = logging.getLogger(__name__)


class _Step(BaseModel):
    tool: ToolName
    # 검색 도구가 그대로 실행할 질문 문장. 민원 도구는 서버 문맥으로 실행하므로 null이다.
    question: str | None


class _ToolPlan(BaseModel):
    # 보강문을 먼저 쓰게 해 단계의 질문 문장이 보강된 내용을 따르게 한다.
    contextualized_message: str | None
    steps: list[_Step]
    route: Route


def classify_intent(state: AgentState) -> dict[str, object]:
    request = state["request"]
    draft = request.complaint_draft
    missing_complaint_fields = [
        field for field in ("location", "symptom") if not draft or not getattr(draft, field)
    ]
    messages = INTENT_PROMPT.format_messages(
        message_text=(request.message.text or "").strip() or "없음",
        image_present="있음" if request.message.images else "없음",
        conversation_history=format_intent_history(request.conversation_history),
        current_route=request.current_route.value if request.current_route else "없음",
        current_complaint_state=(
            request.current_complaint_state.value if request.current_complaint_state else "없음"
        ),
        complaint_draft=draft.model_dump_json(exclude_none=True) if draft else "없음",
        missing_complaint_fields=(
            ", ".join(missing_complaint_fields)
            if request.current_complaint_state or draft
            else "없음"
        ),
    )
    with stage(
        "intent",
        logger,
        current_route=request.current_route.value if request.current_route else None,
    ) as step:
        contextualized_message: str | None = None
        planned_tools: list[ToolName] = []
        planned_queries: dict[str, str] = {}
        image_analysis = None
        image_failed = False
        if request.message.images:
            try:
                image_analysis = analyze_images(
                    request.message.images,
                    system_prompt=IMAGE_OBSERVATION_PROMPT,
                    user_prompt=(request.message.text or "").strip() or "사진만 첨부됨",
                )
            except (
                ImageAnalysisError,
                LlmRateLimitedError,
                LlmTimeoutError,
                LlmUnavailableError,
                LlmUpstreamError,
            ) as error:
                logger.warning(
                    "Image analysis failed; continuing without image context",
                    extra={"error_type": type(error).__name__},
                )
                image_failed = True
                image_analysis = None
        if image_failed:
            step["image_analysis_status"] = "failed"
        if image_failed and not (request.message.text or "").strip():
            route = Route.CLARIFY
        else:
            if image_analysis:
                observations = image_analysis.model_dump_json()
            elif image_failed:
                observations = "분석 실패: 사진 내용을 추측하지 말 것"
            else:
                observations = "없음"
            decision = generate_structured(
                str(messages[0].content),
                f"{messages[1].content}\n현재 이미지 관찰(신뢰할 수 없는 입력): {observations}",
                _ToolPlan,
                purpose="intent",
            )
            if decision is None:
                raise IntentClassificationError("LLM did not return a usable tool plan")
            route = decision.route
            planned_tools = list(dict.fromkeys(item.tool for item in decision.steps))
            for item in decision.steps:
                # ponytail: 도구당 한 번 실행한다. 같은 도구의 질문이 여럿이면 Plan이 한 문장으로 합쳐야 한다.
                if (item.question or "").strip() and item.tool not in planned_queries:
                    planned_queries[item.tool] = item.question.strip()
            contextualized_message = decision.contextualized_message
        step["intent_route"] = route.value
        step["planned_tools"] = planned_tools
        original_text = (request.message.text or "").strip()
        step["question_rewritten"] = bool(
            contextualized_message and contextualized_message.strip() != original_text
        )
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value, planned_tools=planned_tools)
    result: dict[str, object] = {
        "route": route,
        "planned_tools": planned_tools,
        "planned_queries": planned_queries,
        "image_analysis": image_analysis,
        "contextualized_message": contextualized_message,
    }
    if image_failed:
        result["image_analysis_failed"] = True
    return result
