# 평가 데이터 v1

이 데이터는 코드 계약과 프롬프트를 읽고 작성한 **합성 seed 평가셋**이다. 실제 입주민 데이터가 아니며 사람 검수 완료를 뜻하지 않는다. 정답 생성에 평가 대상 모델의 출력을 사용하지 않았다. 제품 배포 판단 전에 도메인 담당자가 라벨과 표현을 검토하고 별도의 실제 사례 holdout을 추가해야 한다. 현재 표본 수로 일반화 성능이나 통계적 유의성을 주장하지 않는다. 확대분도 합성 데이터이며 평가셋 규모 증가가 독립 인간 검수를 대체하지 않는다. 클래스별 최소 8건은 기능 커버리지 목적이고 실제 민원 빈도를 반영한 분포가 아니다.

우선순위는 초안 추출·유형 → 수집 대화 → 라우팅이다. `complaint_cases.jsonl`은 110턴: raw 유형 water_supply 10건, drain 9건, 나머지 8종 각 8건, null 27건이다. 기존 38턴에 독립적으로 라벨링한 72턴을 추가했다. 현재 턴 무정보·이전 값 재추출 금지, 물 관련 정정, 장소 정정, 날짜 5건, 위치 모름·거절, 기존 장소 보존, 증상 모름, 사진 관찰 확인과 관찰 없는 사진 언급을 포함한다. `complaint_conversations.jsonl`은 11대화 28턴, `routing_cases.jsonl`은 20건이다. 추가 대화는 위치 거절, 증상 미확인 후 재질문, 급수→배수 정정, 날짜 우선 제공, 장소 정정을 다룬다. 이미지 URL은 라우팅의 첨부 유무 테스트용이며 다운로드 대상이 아니다.

## 스키마

턴은 `id`, `tags`, `today`, `request`, `expected`, `notes`를 갖는다. `request`는 프로덕션 `ConverseRequest` 전체 입력이다. 식별자는 모두 가상 값이다. `today`는 Asia/Seoul 기준 `2026-09-28`로 고정한다. 실행기는 실제 프롬프트에 같은 날짜를 넣어야 한다.

`expected`:

- `delta`: issue_type, location, symptom, occurred_at 4개 키가 항상 존재한다. 이번 턴에 제공되지 않은 값은 null이다.
- `origin`: 슬롯별 current / history / absent. current는 현재 발화가 값을 제공·정정하거나 이전 사진 관찰을 명시적으로 확인했음을 뜻한다. history는 기존 초안에만 있는 값으로 delta는 null이어야 한다. absent는 어느 쪽에도 추출 근거가 없다. 사진 확인은 프롬프트의 명시적 예외이므로 current이다.
- `draft`: delta를 기존 초안에 병합한 최종 4개 슬롯과 image_urls. 두 필수 슬롯이 차면 null 유형을 other로 보정하는 현재 코드의 동작을 포함한다. raw issue_type 점수는 delta로 평가해야 한다.
- `missing_fields`, `asked_field`: 병합 후 부족한 필수 필드와 다음 질문 슬롯. 순서는 location, symptom이다. 완성되면 [] / null이다.
- `symptom_keywords`: 의미 보존 점검용 내용어 문자열. 비어 있으면 키워드 재현율을 계산하지 않는다. 일부 예시 어간은 활용형에 민감하므로 이 점수만으로 의미 정확성을 확정하지 않는다.

자유 서술 symptom은 유일한 정답 문장이 아니다. 장소는 욕실/화장실, 주방/부엌의 동의 표현을 허용하며 symptom은 명시 사실의 보존과 새로운 사실 추가 여부를 함께 검토한다. 날짜만 채점하고 시각은 무시한다. 기준일 없는 날짜 및 `지난주`처럼 단일 날짜가 유일하게 정해지지 않는 표현은 v1 exact-match 셋에서 제외했다.

`location`은 발화가 명시한 방·구역 또는 물리적으로 특정된 고정 위치(예: 싱크대 아래, 창틀)로 라벨링했다. 기기 종류만 언급해 어느 방인지 알 수 없으면 null이다. 설비 이름 자체가 위치를 충분히 특정하는지는 운영 기준이 아직 명시되지 않았으므로, 관련 사례는 사람 검수에서 다시 확인해야 한다.

대화 행은 `id`, `tags`, `today`, `turns`, `minimum_completion_turns`를 갖는다. 각 턴은 위 스키마이며 이전 **정답** 초안과 대화가 명시되어 있다. 기본 실행은 teacher-forced 방식이므로 최종 JGA도 정답 문맥하의 평가이다. 실제 이전 모델 응답을 다음 턴에 넣는 closed-loop 대화 성능과 구분해야 한다. minimum_completion_turns는 제공된 사용자 발화 시퀀스에서 처음 필수 정보가 모두 주어지는 턴 번호이며, 적응형 사용자 시뮬레이터의 이론 최소 턴 수가 아니다.

라우팅 행은 `id`, `tags`, `request`, `expected`를 갖는다. expected는 `route`, `entry_node`, `classifier_route`이다. collecting 민원 2건은 그래프가 분류기를 우회하므로 classifier_route가 null이다. 나머지는 실제 모델 분류 정답이다. knowledge/complaint의 모호한 후속 발화 유지, 명시적 새 의도 전환, clarify 재분류, 빈 발화 및 사진만 있는 발화를 포함한다.

## 검수 시 특히 볼 경계

- 온수가 안 나옴은 water_supply, 난방 바닥이 차가움은 heating이다.
- 물이 안 내려감은 drain, 수도에서 안 나옴은 water_supply, 잠근 수도꼭지에서 샘은 leak이다.
- 담배 냄새·쓰레기 적치는 other로 라벨링했다. 하수 냄새와 혼동하지 않는다.
- E4 표시 확인만으로 난방 고장을 추론하지 않는다. 이 사례의 delta 유형은 null이고 완성 초안은 코드 보정으로 other이다.
- 이전 사진 관찰 없는 `사진 봐 주세요`는 증상을 생성하지 않는다.
- 이전 질문이 위치일 때의 모름/거절만 위치 `모름`으로 추출한다. 증상 모름은 null이다.

## VLM 후보 이미지

생성 JPG 120장은 [IMAGE-CANDIDATES.md](IMAGE-CANDIDATES.md)와 `image_candidates.jsonl`에 기록했다. 아직 보이는 사실·실제 OCR·이상 여부의 독립 검수가 끝나지 않아 `photo_cases.jsonl`은 없다. 사진을 검수한 뒤 승인된 사례만 그 파일에 등록하고 VLM 실행·채점을 시작한다. 생성 사진 결과는 합성 조건 평가로 보고하고, 실사용 품질에는 별도 실사진 holdout을 사용한다.
