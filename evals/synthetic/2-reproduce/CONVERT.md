# 변환 안내

md 원본을 색인 대상 PDF로 바꾸고, 파싱 실패 케이스용 이상 파일을 만드는 절차다.
**설치가 필요한 도구를 쓰지 않는다.** macOS 기본 명령만 사용한다.

이 문서는 안내다. 실행은 사용자가 한다.

## 왜 PDF인가

`Files.file_type`이 허용하는 건 jpg·png·heic·PDF 넷인데, P1이 이미지 3종을 v1 색인 대상에서
제외한다. 남는 건 PDF 하나뿐이다.

## 쓰는 도구 — 전부 macOS 내장

| 도구 | 역할 | 확인 결과 |
|---|---|---|
| `cupsfilter` | 텍스트 → PDF (텍스트 레이어 있음) | 있음 |
| `sips` | PDF ↔ 이미지 변환 | 있음 |
| `textutil` | 문서 포맷 변환 | 있음 |
| `qlmanage` | 미리보기 썸네일 | 있음 |

`pandoc`·`wkhtmltopdf`·ImageMagick·Ghostscript는 이 기기에 없다. 없이도 아래가 다 된다.

## 1. 전부 한 번에 — make_inputs.py

세 종류를 한 명령으로 만든다. 매니페스트의 `intent`를 읽어 어느 문서를 어떤 형태로 만들지
스스로 정하므로, 건물이 늘어도 이 명령만 다시 돌리면 된다.

```bash
cd /Users/hwangsubin/zipsai-synthetic
python3 make_inputs.py
```

| 산출 | 개수 | 근거 |
|---|---|---|
| 본문 PDF (텍스트 레이어 있음) | 46 | `documents/*.md` |
| 스캔본 PDF (텍스트 레이어 없음) | 3 | `intent: PARSE_NO_TEXT_LAYER` |
| 사진 jpg | 2 | `intent: PARSE_IMAGE_ONLY` |

만들지 않고 이미 있는 `build/`만 검사하려면:

```bash
python3 make_inputs.py --check
```

## 2. 스크립트가 하는 일

| 단계 | 수단 | 비고 |
|---|---|---|
| md 파싱 | 자체 파서 | `#`·`##`·`\| 표 \|`·`- 목록`·`- N -` 쪽나눔만 다룬다 |
| 렌더링 | **reportlab** | 자동 줄바꿈, 괘선 표, 제목 계층 |
| 한글 폰트 | `AppleGothic.ttf` | AppleSDGothicNeo는 TTC 묶음 + PostScript 아웃라인이라 reportlab이 못 읽는다 |
| 반복 머리말 | `onPage` 콜백 | 본문에 글자로 박힌 머리말을 실제 페이지 머리말로 올리고 쪽번호를 찍는다 |
| 텍스트 레이어 제거 | `sips` PDF→PNG→PDF | **쪽마다 따로** 굽고 다시 합친다. `sips`는 PDF 첫 장만 변환한다 |
| 사진 공지 | `sips -s format jpeg` | 품질 70 |

`cupsfilter`는 쓰지 않는다. text/plain만 받고 **줄을 접지 않아 긴 줄의 오른쪽을 버린다.**
HTML·RTF 입력도 변환 필터가 없어 실패한다.

## 3. 알아둘 것 두 가지

**파일명 정규화** — macOS는 한글 파일명을 **NFD**로 저장한다. 파이썬 리터럴은 NFC라서
`"스캔본" in path.name`도 `glob("*스캔본.pdf")`도 그냥은 맞지 않는다.
`unicodedata.normalize("NFC", ...)`로 맞춘 뒤 비교해야 한다. 쉘의 `grep`·`ls`도 같은 함정이 있다.

**개인정보 줄바꿈** — `302호 김미정님`이 줄 끝에서 갈라지면 마스킹 정규식이 놓친다.
렌더링 단계에서 해당 표현에 `&nbsp;`를 넣어 붙여 둔다.

## 4. 용량 확인

`Files.file_size`가 10MB 제한이다.

```bash
find build -type f -size +10M
```

아무것도 안 나와야 한다.

## 기억할 것

원본은 `documents/*.md` 하나다. PDF·PNG·JPG는 전부 파생물이고 `build/`에만 산다.
문서를 고칠 때는 md만 고치고 위 명령을 다시 돌린다. PDF를 직접 고치지 않는다.
