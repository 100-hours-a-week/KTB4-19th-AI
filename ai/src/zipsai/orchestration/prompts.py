from langchain_core.prompts import ChatPromptTemplate

_INTENT_SYSTEM_PROMPT = (
    "Route the resident's current turn to complaint (facility problem or repair/registration), "
    "knowledge (building-information question), or clarify (unclear intent). Use current text, "
    "images, conversation history, and current_route. For a brief reply, follow the prior exchange "
    "when it clearly establishes the intent; current_route alone is not enough. Image content alone "
    "never establishes intent. "
    "For image-only turns, continue an active route only when the prior exchange clearly requested "
    "a photo; otherwise clarify. Never invent a question from an image. For text-plus-image turns, "
    "classify their combined meaning. Also return one factual Korean summary and readable OCR text "
    "per attached image, in input order. Use null when unavailable; do not guess unseen details. "
    "Return exactly one image observation per attached image."
)

_INTENT_USER_TEMPLATE = (
    "현재 발화: {message_text}\n"
    "이미지 첨부: {image_present}\n"
    "이전 대화:\n{conversation_history}\n"
    "현재 진행 중인 route: {current_route}"
)

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _INTENT_SYSTEM_PROMPT),
        ("user", _INTENT_USER_TEMPLATE),
    ]
)


_CLARIFY_SYSTEM_PROMPT = (
    "You are helping a residential-building assistant recover from an ambiguous conversation. "
    "The previous route clarification did not resolve the user's intent. Using the conversation "
    "history and current turn, ask exactly one short Korean follow-up question that helps the user "
    "clearly choose between reporting a facility complaint and asking for building information. "
    "Do not answer the request, do not mention internal route names, and do not add explanations."
)

_CLARIFY_USER_TEMPLATE = (
    "이전 대화:\n{conversation_history}\n\n현재 발화: {message_text}"
)

CLARIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CLARIFY_SYSTEM_PROMPT),
        ("user", _CLARIFY_USER_TEMPLATE),
    ]
)
