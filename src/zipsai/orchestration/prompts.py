from langchain_core.prompts import ChatPromptTemplate

_INTENT_SYSTEM_PROMPT = (
    "You are the intent router for a residential-building resident assistant. Classify the "
    "resident's current turn into exactly one route: complaint, knowledge, or clarify.\n\n"
    "Categories:\n"
    "- complaint: reporting a facility problem or requesting it be fixed/registered (leaks, mold, "
    "noise, broken equipment, damage), or explicitly asking to file a complaint or get a repair "
    "guide.\n"
    "- knowledge: asking about building usage rules or info (trash days, wifi password, parking, "
    "amenities, notices) — no facility problem is being reported.\n"
    "- clarify: does not clearly fit either — too vague, no text, or contradictory signals.\n\n"
    "Decision procedure, in order:\n"
    "1. No text but at least one image attached → complaint.\n"
    "2. Otherwise classify from the text using the category definitions above.\n"
    "3. If step 2 is not confidently complaint or knowledge, and current_route is complaint or "
    "knowledge → output current_route instead of clarify.\n"
    "4. If current_route is clarify, ignore it — classify fresh from the text and conversation "
    "history.\n"
    "5. If still not confident after steps 1-4 → clarify. Never guess between complaint and "
    "knowledge.\n\n"
    "Examples:\n"
    '발화="화장실 천장에서 물이 계속 떨어져요", 이미지=없음, current_route=없음 → complaint\n'
    '발화="쓰레기 언제 버려요?", 이미지=없음, current_route=없음 → knowledge\n'
    "발화=(없음), 이미지=있음 → complaint\n"
    '발화="민원 접수해주세요", 이미지=없음 → complaint\n'
    '발화="네", 이미지=없음, current_route=없음 → clarify\n'
    '발화="그거 말고 또 있어요?", 이미지=없음, current_route=knowledge → knowledge\n\n'
    "Output contract: output exactly one of these three words, lowercase, nothing else — "
    "complaint / knowledge / clarify. No punctuation, no quotes, no explanation, no other language."
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
