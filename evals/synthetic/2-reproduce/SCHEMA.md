# 합성데이터 틀

건물 관리 문서 RAG의 색인·검색을 검증하기 위한 데이터셋 정의다.
문서만 만드는 것이 아니라 **문서 + 정답셋**을 한 묶음으로 만든다. 정답셋이 없으면
임베딩 모델 비교(4B vs 8B)에서 무엇이 좋아졌는지 측정할 수단이 없다.

## 디렉터리

```
zipsai-synthetic/
├── SCHEMA.md              이 파일
├── validate.py            제약 검증
├── CONVERT.md             md → PDF 변환 안내
├── zones/z1~z4.yaml       구역 공통 정보
├── buildings/b001~b010/
│   ├── building.yaml      프로필 + 문서 매니페스트
│   ├── documents/*.md     문서 본문
│   └── goldset.jsonl      정답셋
└── path-b/qa-cards.jsonl  관리자 답변이 달린 질의 카드
```

## 두 층 구조 — 이 데이터셋의 핵심

| 층 | 정보 | 같은 구역 건물 간 | 검색이 엉뚱한 건물을 가져왔을 때 |
|---|---|---|---|
| 구역 공통 | 쓰레기 배출 요일·장소, 종량제봉투, 재활용 기준 | 동일 | **답이 맞아버린다** |
| 건물 고유 | 주차, 관리비, 세탁실, 택배, 현관 | 다름 | 답이 틀린다 |

`building_id` 격리 검증은 **건물 고유 질문으로만** 성립한다.
구역 공통 질문만으로 테스트하면 격리가 깨져 있어도 통과한다.
그래서 정답셋의 모든 문항은 `scope` 라벨을 반드시 갖는다.

## zones/z{N}.yaml

```yaml
zone_id: Z1
zone_name: 대학로 구역
waste:
  general_days: [월, 수, 금]        # 일반 쓰레기 배출 요일
  recycle_days: [화, 목]            # 재활용 배출 요일
  food_days: [매일]                 # 음식물
  time: 저녁 7시 ~ 자정
  place: 건물 앞 지정 배출대
  bag_shops: [행복마트, 대학로편의점]   # 종량제봉투 판매처 (가상)
bulk_waste:
  method: 주민센터 신고 후 스티커 부착
  fee: 품목당 3,000원부터
notes: 배출 요일을 어기면 수거하지 않고 계고장이 붙는다
```

구역 4개의 `general_days`·`recycle_days`는 서로 달라야 한다.
같으면 구역을 나눈 의미가 없다.

## buildings/b{NNN}/building.yaml

```yaml
building_id: 1
building_code: b001
zone_id: Z1
building_name: 한빛빌라            # 가상. 실존 건물명 금지
building_type: 원룸형 빌라          # 원룸형 빌라 | 다가구주택 | 소규모 임대건물
built_year: 2015
room_count: 30
room_no_format: "1NN"             # Rooms.room_no VARCHAR(5) 이내
manager:
  resident: false                 # 상주 여부. false면 전화만
  contact_hours: 평일 09:00~18:00

# 건물 고유 정보 — 같은 구역 건물과 반드시 달라야 하는 값
specifics:
  parking: 선착순 12면 (세대수 대비 부족)
  monthly_fee: 정액 50,000원
  fee_includes: [공용전기, 수도, 인터넷]
  laundry: 지하 공용 세탁실 (세탁기 2, 건조기 1)
  parcel: 무인택배함 1층
  entrance: 공동현관 비밀번호, 분기별 변경
  elevator: true
  heating: 개별 도시가스 보일러

documents:
  - file: documents/rule-001-생활수칙.md
    document_title: 입주자 생활수칙      # 20자 이내 (Rule_Documents.document_title)
    doc_type: rule                    # rule | notice | facility
    version: 1
    is_valid: false                   # v2가 있으므로 false
    file_type: PDF                    # jpg | png | heic | PDF 만 허용 (Files.file_type)
    intent: [CHUNK_NO_SECTION]
    scope: building
  - file: documents/notice-004-소방시설점검.md
    document_title: 소방시설 점검
    doc_type: notice
    version: 1
    is_valid: true
    file_type: PDF
    intent: [PARSE_NO_TEXT_LAYER]
    scope: building
    note: 스캔본. make_inputs.py 가 렌더 후 이미지로 구워 텍스트 레이어를 없앤다
```

**모든 문서 항목은 md 본문을 가진다.** 스캔본·사진 케이스도 마찬가지다.
`make_inputs.py`가 md를 PDF로 렌더한 뒤 `intent`에 따라 이미지로 구워 텍스트 레이어를 없애거나
jpg로 바꾼다. 본문이 없으면 OCR 품질을 잴 수 없어 그 케이스가 무의미해진다.

## intent — 파이프라인 분기 코드

각 문서가 어느 분기를 발동시키려고 존재하는지 적는다. 데이터셋의 존재 이유를 문서마다 못박는 칸이다.

| 코드 | P1 단계 | 발동시키는 분기 |
|---|---|---|
| `PARSE_NO_TEXT_LAYER` | 2 파싱 | 텍스트 레이어 없는 스캔 PDF → 색인 보류 |
| `PARSE_IMAGE_ONLY` | 1 업로드 | 사진으로 찍은 공지 → 이미지 직접 색인(docling OCR) |
| `MASK_PII_DETECTED` | 3 마스킹 | 개인정보 탐지 → 관리자 확인 대기 |
| `MASK_FALSE_POSITIVE` | 3 마스킹 | 문서번호·금액을 전화번호로 오인 |
| `CLEAN_REPEATED_HEADER` | 4 정제 | 반복 머리말·꼬리말 제거 |
| `CLEAN_OVER_REMOVAL` | 4 정제 | 제거 비율 30% 초과 → 보류 |
| `CHUNK_NO_SECTION` | 5 청킹 | 조·항 구조가 없어 경계 우선 규칙이 발동하지 않음 |
| `CHUNK_TOO_SHORT` | 5 청킹 | 문서 전체가 한 청크에 못 미침 |
| `CHUNK_TABLE_WHOLE` | 5 청킹 | 표는 통째로 한 청크 |
| `CHUNK_OVERLAP` | 5 청킹 | 400자 초과 → 앞뒤 50자 겹침 분할 |
| `INDEX_VERSION_REPLACE` | 7 적재 | 개정판 → 기존 청크 삭제 후 적재 |
| `INDEX_ZONE_SHARED` | 7 적재 | 구역 공통 정보 → 격리 오탐을 유발 |

## goldset.jsonl — 한 줄에 한 문항

```json
{"id": "b001-q01", "building_code": "b001", "zone_id": "Z1", "question": "세탁실은 몇 시까지 쓸 수 있나요?", "answerable": true, "scope": "building", "expected_answer": "지하 공용 세탁실은 오전 6시부터 밤 11시까지 이용할 수 있습니다.", "expected_citation": {"file": "documents/facility-001-세탁실안내.md", "section": "운영 시간"}, "note": ""}
```

| 필드 | 규칙 |
|---|---|
| `id` | `b{NNN}-q{NN}` |
| `question` | 입주민이 실제로 쓸 법한 구어체. 존댓말 |
| `answerable` | 그 건물 문서로 답할 수 있으면 true |
| `scope` | `building` 또는 `zone`. **필수** |
| `expected_answer` | **200자 이내** (Messages.content VARCHAR(200)) |
| `expected_citation` | `answerable: true`면 필수. 실존하는 파일·섹션이어야 함 |
| `expected_citation.section` | 문서의 `## 제목` 과 정확히 일치 |

`answerable: false` 문항은 `expected_citation`을 `null`로 두고 `expected_answer`에
"문서에 없음 — 관리자 확인 필요"를 적는다. 이 문항들이 경로 B의 원천이다.

## path-b/qa-cards.jsonl

`answerable: false` 문항 중 **관리자가 답을 단 것만** 담는다. 답이 안 달린 나머지는
파일에 없다. 그것이 "관리자가 답하지 않으면 루프가 멈춘다"는 상태의 재현이다.

```json
{"qa_id": 1, "building_code": "b001", "goldset_id": "b001-q18", "question": "계약 연장은 언제까지 말씀드려야 하나요?", "answer": "만기 2개월 전까지 관리사무소로 연락 주시면 됩니다.", "answered_by": 1, "answered_at": "2026-09-12T14:20:00+09:00", "issue_type": "qa_card"}
```

`issue_type`은 항상 `qa_card` 고정이다. 민원 택소노미 10종은 민원 경로에서만 쓰이며
이 데이터셋과 무관하다.

## 문서 본문 작성 규칙

| 항목 | 규칙 |
|---|---|
| 형식 | Markdown. `## 섹션제목`이 곧 `expected_citation.section` |
| 어투 | 원룸·다가구 관리인이 실제로 쓰는 말투. 관리규약 조문체가 아님 |
| 조·항 | **쓰지 않는다.** 30호실 건물은 의무관리대상이 아니라 관리규약이 없다 |
| 구역 정보 | 구역 공통 값은 해당 `zones/z{N}.yaml`과 글자 그대로 일치 |
| 개인정보 | `MASK_PII_DETECTED` 문서에만 의도적으로 심는다. 전부 가상 값 |
| 실존 정보 | 실존 건물명·주소·상호·전화번호 금지 |

## DB에서 온 제약

| 제약 | 출처 |
|---|---|
| `document_title` 20자 이내 | `Rule_Documents.document_title VARCHAR(20)` |
| `room_no` 5자 이내 | `Rooms.room_no VARCHAR(5)` |
| `expected_answer` 200자 이내 | `Messages.content VARCHAR(200)` |
| `file_type`은 jpg·png·heic·PDF만 | `Files.file_type` |
| 모든 문서에 파일 필요 | `Rule_Documents.attachment_id NOT NULL` |
| 개정은 version 증가 + is_valid | `Rule_Documents.version`, `is_valid` |

`file_type` 허용 4종 중 **색인 대상은 PDF·JPG·PNG 3종**이다. 이미지도 docling OCR을 거쳐 PDF와 같은 경로에 합류한다.
heic만 예외로, 백엔드가 확장자를 변환해 S3에 올리므로 AI는 받지 않는다.
이 데이터셋은 md 원본을 PDF로 변환해 쓰고, 이미지 입력은 jpg 2건으로 재현한다.

## 미해결 — 팀 확정 필요

| 항목 | 내용 |
|---|---|
| `doc_type` 저장 위치 | `Rule_Documents`에 문서 종류 컬럼이 없다. 이 데이터셋은 매니페스트에 유지하되 DB 매핑은 미정 |
| 표 400자 초과 시 우선순위 | **코드는 표 우선으로 확정**(`ai/src/zipsai/indexing/chunk.py:171-173` — 표가 capacity를 넘으면 통째로 한 청크). 설계서에는 여전히 답이 없고, 8000자를 넘는 표를 어떻게 자를지도 미정이다 |
