from langchain_core.prompts import ChatPromptTemplate

COMPLAINT_SYSTEM_PROMPT = (
    "You extract facility-complaint details from a resident's CURRENT turn in a residential-building "
    "assistant. Use the existing draft and conversation history only to interpret a correction in the "
    "current message. Extract ONLY values explicitly stated or corrected in the current message; do not "
    "repeat unchanged values. Those are merged separately.\n\n"
    "issue_type taxonomy (pick exactly one, or null if the current turn doesn't clearly indicate one):\n"
    "- water_supply: water supply is missing or insufficient (no water, low pressure, no hot water).\n"
    "- drain: drainage is blocked or malfunctioning (clogged drain, backflow, sewage smell).\n"
    "- leak: water is escaping where it shouldn't (leaking from ceiling, wall, pipe, faucet).\n"
    "- heating: heating/hot-water-for-heating is not working or not controllable.\n"
    "- electricity: power issue (outage, broken outlet/light, tripped breaker, exposed wiring).\n"
    "- mold: mold or mildew growth.\n"
    "- pest: insects or pests (cockroaches, ants, rodents, etc.).\n"
    "- facility: a non-water/electrical fixture is broken (elevator, door, window, lock, intercom).\n"
    "- noise: noise complaint (from another unit, construction, equipment).\n"
    "- other: a real facility complaint that doesn't fit any category above.\n"
    "Disambiguation: water not coming out → water_supply, not drain. Water backing up or not going "
    "down → drain, not water_supply. Water actively dripping/pooling somewhere → leak, even if the "
    "source is a pipe or the water system.\n\n"
    "Other fields:\n"
    "- location: where the problem is (e.g. 화장실, 주방). null if not stated in this turn.\n"
    "- symptom: what's wrong, in the resident's own words. null if not stated in this turn.\n"
    "- occurred_at: when the problem started or was first noticed, only if stated in this turn. "
    "Resolve relative expressions (어제, 오늘, 그저께, 3일 전, 지난주 등) against today's date, given "
    "below as Asia/Seoul. Output an ISO 8601 date (YYYY-MM-DD); if only a time is known, keep the "
    "date and omit finer precision. null if not stated in this turn.\n"
    '- missing: an array listing which of "location"/"symptom" are still unknown overall — look '
    "at the EXISTING draft below together with what you just extracted this turn, not just this "
    'turn\'s message. Empty array if both are known. Only "location" and/or "symptom" are valid '
    "entries; never include issue_type or occurred_at.\n"
    "- reply: a short, natural Korean follow-up question for the resident, asking ONLY about the "
    'field(s) listed in `missing`. Empty string "" if `missing` is empty. One short, friendly '
    "sentence, no lists. Never ask about issue_type or occurred_at.\n\n"
    "Output contract: output ONLY a JSON object with exactly these six keys, nothing else. No "
    "markdown, no explanation, no code fences.\n"
    '{{"issue_type": null, "location": null, "symptom": null, "occurred_at": null, "missing": [], '
    '"reply": ""}}'
)

COMPLAINT_USER_TEMPLATE = (
    "오늘 날짜(Asia/Seoul): {today}\n\n"
    "이전 대화:\n{conversation_history}\n\n현재 민원 초안: {complaint_draft}\n\n현재 발화: {message_text}"
)


COMPLAINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", COMPLAINT_SYSTEM_PROMPT),
        ("user", COMPLAINT_USER_TEMPLATE),
    ]
)

VLM_ANALYSIS_PROMPT = (
    "You are inspecting a resident-submitted photo of a residential facility issue "
    "(plumbing, electrical, appliance, structural).\n\n"
    "Output raw JSON only. Do NOT wrap the output in markdown, code fences, or backticks. "
    "Do NOT add any text before or after the JSON object.\n\n"
    "Schema (exact keys, no extras):\n"
    '{"images":[{"summary":"string or null","ocr_text":"string or null"}]}\n\n'
    "Rules:\n"
    "- One object per image, in the exact order the images were given.\n"
    "- summary: 한글 1~2문장. 기기 종류, 에러/이상 여부, 화면·표시등에 보이는 핵심 수치만 담는다. "
    "부가 설명이나 나열식 서술은 넣지 않는다.\n"
    "- ocr_text: 이미지에 보이는 텍스트를 그대로 옮긴다. 보이지 않으면 null.\n"
    "- Use JSON null when an observation is unavailable.\n"
    "- Do not infer or guess anything that is not visible in the image."
)
