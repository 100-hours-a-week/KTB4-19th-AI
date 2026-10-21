#!/usr/bin/env bash
# /converse 해피패스 e2e. 컨테이너가 떠 있는 상태에서 돌린다.
#
#   docker compose up -d
#   IMAGE_URL='<presigned URL>' ai/scripts/e2e_happy.sh
#
# IMAGE_URL은 S3 presigned URL이며 경로에 .jpg/.jpeg/.png/.webp가 있어야 한다
# (contracts/converse.py의 확장자 검증). 없으면 5단계를 건너뛴다.
#
# 실패 경로(근거 없음·건물 격리·clarify·오류코드)는 유닛 테스트가 덮는다.
# 여기는 "컨테이너를 실제로 띄웠을 때 정상 흐름이 끝까지 도는가"만 본다.
set -uo pipefail

API=http://127.0.0.1:${AI_PORT:-8001}
QDRANT=http://127.0.0.1:${QDRANT_PORT:-6333}
COLLECTION=${QDRANT_COLLECTION:-documents}
BUILDING_ID=101
DOC_ID=e2e-receipt
# 표가 넓은 PDF는 청크가 임베딩 상한(8000자)을 넘어 색인이 실패한다. 평문으로
# OCR되는 파일을 쓴다.
FILE_KEY=${FILE_KEY:-documents/rules/5c1af08a-c93d-449b-9db3-2ff15c65a16e-image0__3_.png}
INDEX_TIMEOUT=${INDEX_TIMEOUT:-300}

pass=0
fail=0

ok()   { echo "  PASS  $1"; pass=$((pass + 1)); }
ng()   { echo "  FAIL  $1"; fail=$((fail + 1)); }
step() { echo; echo "=== $1"; }

converse() {
  # 사진이 붙은 민원은 추출 LLM과 VLM을 직렬로 부른다. LLM_TIMEOUT_SECONDS가
  # 각각에 걸리므로 클라이언트 상한은 그 두 배보다 넉넉해야 한다.
  curl -s --max-time 300 -X POST "$API/api/v3/ai/converse" \
    -H 'Content-Type: application/json' --data-binary @-
}

# 응답 JSON을 인자로 받아 파이썬 식(expr)을 평가한다. d=data, r=result.
# 본문을 stdin으로 받으면 호출부가 파이프가 되고, 그러면 이 함수가 서브셸에서 돌아
# pass/fail 카운터 증가가 사라진다(= 실패해도 exit 0). 반드시 인자로 받는다.
check() { # $1=설명 $2=파이썬식 $3=응답본문
  local desc=$1 expr=$2 body=$3
  if python3 -c '
import json, sys
body = sys.stdin.read()
try:
    d = json.loads(body)["data"]
except Exception:
    print("  응답 파싱 실패:", body[:300]); raise SystemExit(1)
r = d["result"]
sys.exit(0 if ('"$expr"') else 1)
' <<<"$body"; then
    ok "$desc"
  else
    ng "$desc"
    echo "      응답: $(head -c 300 <<<"$body")"
  fi
}

step "0. health"
health=$(curl -s --max-time 10 "$API/health")
if grep -q '"ai_service":"ready"' <<<"$health" \
  && grep -q '"vector_store":"ready"' <<<"$health"; then
  ok "health 200 / 의존성 ready"
else
  ng "health"
  echo "      응답: $health"
  echo; echo "컨테이너가 안 떠 있다. docker compose up -d 먼저."
  exit 1
fi

step "1. 색인 — 실제 S3 문서"
# 지난 실행이 남긴 청크를 먼저 지운다. 안 지우면 아래 count 폴링이 옛 청크를 보고
# 즉시 통과한다 — 색인이 실패해도 PASS가 나오고, 아직 도는 중인데 다음 단계가
# 출발해 임베딩 서비스와 경합한다.
curl -s --max-time 30 -o /dev/null -X POST \
  "$QDRANT/collections/$COLLECTION/points/delete?wait=true" \
  -H 'Content-Type: application/json' \
  -d "{\"filter\":{\"must\":[{\"key\":\"doc_id\",\"match\":{\"value\":\"$DOC_ID\"}}]}}"

code=$(curl -s --max-time "$INDEX_TIMEOUT" -o /dev/null -w '%{http_code}' \
  -X POST "$API/api/v3/ai/indexing/jobs" -H 'Content-Type: application/json' \
  -d "{\"building_id\":$BUILDING_ID,\"doc_id\":\"$DOC_ID\",\"title\":\"e2e 테스트 문서\",\"file_key\":\"$FILE_KEY\"}")
if [ "$code" = "200" ]; then
  ok "색인 작업 완료 (200)"
else
  ng "색인 작업 완료 — HTTP $code"
fi

# 200은 작업이 끝났다는 뜻이다. 청크가 Qdrant에 들어갔는지 확인한다.
echo "  색인 완료 대기 (최대 ${INDEX_TIMEOUT}초, docling OCR은 느리다)..."
deadline=$((SECONDS + INDEX_TIMEOUT))
count=0
while [ $SECONDS -lt $deadline ]; do
  count=$(curl -s --max-time 10 -X POST "$QDRANT/collections/$COLLECTION/points/count" \
    -H 'Content-Type: application/json' \
    -d "{\"filter\":{\"must\":[{\"key\":\"doc_id\",\"match\":{\"value\":\"$DOC_ID\"}}]},\"exact\":true}" \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["result"]["count"])
except Exception: print(0)')
  [ "${count:-0}" -gt 0 ] && break
  sleep 5
done
if [ "${count:-0}" -gt 0 ]; then
  ok "Qdrant에 청크 ${count}건 적재"
else
  ng "색인 미완료 — 청크 0건 (docker compose logs zipsai-api 확인)"
fi

step "2. 지식 질의"
resp=$(converse <<JSON
{"building_id":$BUILDING_ID,"room_no":"101","resident_id":"r-1","conversation_id":"e2e-k",
 "trace_id":"e2e-k1","current_route":null,"current_complaint_state":null,
 "message":{"message_id":"m-k1","text":"ChatGPT 구독 결제 금액이 얼마인가요?","image_urls":[]},
 "conversation_history":[],"complaint_draft":null}
JSON
)
check "route=knowledge, 근거 있음" \
  'd["route"]=="knowledge" and len(r["citations"])>=1 and r["has_sufficient_evidence"] is True' \
  "$resp"

step "3. 민원 1턴 — 정보 부족"
resp=$(converse <<JSON
{"building_id":$BUILDING_ID,"room_no":"101","resident_id":"r-1","conversation_id":"e2e-c",
 "trace_id":"e2e-c1","current_route":null,"current_complaint_state":null,
 "message":{"message_id":"m-c1","text":"집에 문제가 생겼어요","image_urls":[]},
 "conversation_history":[],"complaint_draft":null}
JSON
)
check "route=complaint, collecting, 부족 필드 있음" \
  'd["route"]=="complaint" and d["next_complaint_state"]=="collecting" and len(r["missing_fields"])>0' \
  "$resp"

step "4. 민원 2턴 — 상태 되돌려주고 정보 보충"
resp=$(converse <<JSON
{"building_id":$BUILDING_ID,"room_no":"101","resident_id":"r-1","conversation_id":"e2e-c",
 "trace_id":"e2e-c2","current_route":"complaint","current_complaint_state":"collecting",
 "message":{"message_id":"m-c2","text":"주방 싱크대 아래에서 물이 새요","image_urls":[]},
 "conversation_history":[],"complaint_draft":null}
JSON
)
check "ready_to_confirm, draft에 위치·증상 채워짐" \
  'd["next_complaint_state"]=="ready_to_confirm" and r["missing_fields"]==[] and r["complaint_draft"]["location"] and r["complaint_draft"]["symptom"]' \
  "$resp"

step "5. 민원 + 사진 — VLM 실호출"
if [ -z "${IMAGE_URL:-}" ]; then
  echo "  SKIP  IMAGE_URL 미설정"
else
  payload=$(python3 -c '
import json, os
print(json.dumps({
    "building_id": '"$BUILDING_ID"', "room_no": "101", "resident_id": "r-1",
    "conversation_id": "e2e-i", "trace_id": "e2e-i1",
    "current_route": None, "current_complaint_state": None,
    "message": {"message_id": "m-i1", "text": "세탁기에서 물이 새요. 고쳐주세요",
                "image_urls": [os.environ["IMAGE_URL"]]},
    "conversation_history": [], "complaint_draft": None,
}, ensure_ascii=False))')
  resp=$(converse <<<"$payload")
  check "image_analysis 채워짐" \
    'r["image_analysis"] and r["image_analysis"]["images"][0]["summary"]' \
    "$resp"
fi

echo
echo "================================"
echo "PASS $pass / FAIL $fail"
[ "$fail" -eq 0 ] || exit 1
