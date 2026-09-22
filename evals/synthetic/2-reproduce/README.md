# 재현 방법

`1-dataset/`의 완성물을 여기서 만든다.

## 필요한 것

| 항목 | 값 |
|---|---|
| Python | 3.12 |
| 패키지 | `pyyaml` · `reportlab` · `pypdf` |
| OS | **macOS 전용** — 아래 제약 참고 |

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyyaml reportlab pypdf
```

## 실행

```bash
python3 validate.py       # 데이터셋 제약 검사
python3 make_inputs.py    # ../1-dataset/build/ 생성 + 자체 검사
python3 make_report.py    # ../1-dataset/데이터셋-안내.md 갱신
```

세 명령이 모두 `PASS`로 끝나야 정상이다.
`make_inputs.py --check`는 새로 만들지 않고 이미 있는 결과만 검사한다.

## 무엇이 무엇을 읽고 쓰나

| 스크립트 | 읽는 것 | 쓰는 것 |
|---|---|---|
| `validate.py` | `buildings/` `zones/` | 없음 (검사만) |
| `make_inputs.py` | `buildings/` | `../1-dataset/build/` |
| `make_report.py` | `buildings/` `path-b/` | `../1-dataset/데이터셋-안내.md` |

## 원본 구성

| 경로 | 내용 |
|---|---|
| `buildings/b001~b010/building.yaml` | 건물 프로필 + 문서 매니페스트 |
| `buildings/b001~b010/documents/*.md` | 문서 원본 51건 |
| `buildings/b001~b010/goldset.jsonl` | 정답셋 235문항 |
| `zones/z1~z4.yaml` | 구역 공통 정보 (쓰레기 요일, 종량제봉투, 재활용 기준) |
| `path-b/qa-cards.jsonl` | 관리자 답변이 달린 질의 카드 50건 |

필드 의미와 `intent` 코드 12종은 [`SCHEMA.md`](SCHEMA.md)에,
변환 원리와 함정은 [`CONVERT.md`](CONVERT.md)에 있다.

## 문서를 고치거나 늘릴 때

1. `buildings/bNNN/documents/*.md`를 고친다
2. 문서를 새로 넣었다면 `buildings/bNNN/building.yaml`의 `documents:`에 항목을 추가한다
3. 질문을 넣거나 고쳤다면 `goldset.jsonl`도 함께 고친다
4. 세 명령을 다시 돌린다

`validate.py`가 길이 제약·인용 실존·격리 조건을 검사하므로, 빠뜨리면 여기서 걸린다.

## 제약 두 가지

**macOS 전용이다.** `make_inputs.py`가 한글 렌더링에
`/System/Library/Fonts/Supplemental/AppleGothic.ttf`를 직접 가리키고,
텍스트 레이어를 없앨 때 macOS 내장 `sips`를 쓴다.
Linux CI나 리눅스 환경이 필요해지면 폰트를 레포에 넣고 `sips` 대신 다른 수단을 써야 한다.

**한글 파일명은 NFD로 저장된다.** macOS가 그렇게 저장하므로
`"스캔본" in path.name`도 `glob("*스캔본.pdf")`도 그냥은 맞지 않는다.
`unicodedata.normalize("NFC", ...)`로 맞춘 뒤 비교해야 한다. 쉘의 `grep`·`ls`도 같은 함정이 있다.
