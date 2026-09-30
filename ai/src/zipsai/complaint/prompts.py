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
    "- complaint_switch: whether this turn is still about the SAME complaint as the existing draft "
    'below. Output exactly one of "same", "ask", "accept".\n'
    '  "same": the turn continues, corrects, or adds to the existing draft. Also use "same" when the '
    "previous assistant turn offered to switch complaints and the resident refused (아니요, 아까 "
    "그거요, 그거 말고요).\n"
    '  "ask": the turn reports a problem clearly different from the draft\'s symptom — e.g. the draft '
    "is 보일러 누수 and the resident now says 세탁기가 안 돌아가요.\n"
    '  "accept": the previous assistant turn asked whether to switch to a specific new complaint, and '
    "the resident agreed (네, 맞아요, 그걸로 해주세요).\n"
    '  Default to "same" whenever you are not confident. Discarding a draft the resident already '
    "filled in costs far more than merging one extra turn into it.\n"
    '  If the draft is empty or has no symptom yet, always output "same".\n'
    "  If the current turn states no symptom, judge from issue_type instead: a clearly different "
    'issue_type is "ask"; anything else is "same". A bare answer that only supplies a missing field '
    '(화장실이요, 어제요, 네) is always "same".\n'
    "- location: where the problem is (e.g. 화장실, 주방). null if not stated in this turn. "
    "Exception: when the previous assistant turn asked for the location and the resident "
    "replies that they do not know or will not say (몰라, 모르겠어요, 안 알려줄래요), output "
    'the literal string "모름" instead of null — an unknown location is a usable value '
    "because the manager can call and confirm it.\n"
    "- symptom: what's wrong, in the resident's own words. null if not stated in this turn. "
    'NEVER output "모름" or any placeholder here; a complaint with no symptom cannot be '
    "acted on, so leave it null and it will be asked again. Two exceptions may take the symptom "
    "from an earlier assistant turn instead of the current message:\n"
    "  (1) an earlier assistant turn reported what a photo showed and the resident confirms it or "
    "points back to the photo (네, 맞아요, 사진에 있는 그거요, 사진 봐) — use that reported "
    "observation.\n"
    '  (2) complaint_switch is "accept" — use the new complaint named inside the previous assistant '
    "turn's switch question, quoted there between 작은따옴표. Copy it as written. Do not invent a "
    "symptom that does not appear in that question.\n"
    "  These two are the only cases where a value may come from an earlier turn.\n"
    "- occurred_at: when the problem started or was first noticed, only if stated in this turn. "
    "Resolve relative expressions (어제, 오늘, 그저께, 3일 전, 지난주 등) against today's date, given "
    "below as Asia/Seoul. Output an ISO 8601 date (YYYY-MM-DD); if only a time is known, keep the "
    "date and omit finer precision. null if not stated in this turn. Output null when "
    'complaint_switch is "accept" unless the current message itself states a time.\n'
    '- missing: an array listing which of "location"/"symptom" are still unknown overall — look '
    "at the EXISTING draft below together with what you just extracted this turn, not just this "
    'turn\'s message. Empty array if both are known. Only "location" and/or "symptom" are valid '
    "entries; never include issue_type or occurred_at. "
    'When complaint_switch is "accept", ignore the existing draft entirely and judge only against '
    "the new complaint — the old draft is being discarded.\n"
    "- reply: a short, natural Korean follow-up question asking about EXACTLY ONE field — the "
    'first still-unknown field in the order "location", then "symptom". Never ask about two '
    'fields in one turn. Empty string "" if `missing` is empty. One short, friendly sentence, '
    "no lists. Never ask about issue_type or occurred_at. "
    'Output "" when complaint_switch is "ask" — the service writes that question itself.\n\n'
    "Output contract: output ONLY a JSON object with exactly these seven keys, nothing else. No "
    "markdown, no explanation, no code fences.\n"
    '{{"complaint_switch": "same", "issue_type": null, "location": null, "symptom": null, '
    '"occurred_at": null, "missing": [], "reply": ""}}'
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
