"""이번 브랜치에서 바뀐 node.py 흐름을 눈으로 보는 데모.

실제 API를 호출하지 않는다 — 텍스트 LLM/VLM 응답을 시나리오별로 미리 정해두고
_gather_turn_evidence -> _advance_complaint -> _finalize_turn을 그대로 돌려서
각 단계의 중간 결과를 바로 출력한다. 비용 없이 몇 번이든 다시 돌려볼 수 있다.

실행: PYTHONPATH=.:ai/src ai/.venv/bin/python ai/temp_complaint_demo.py
"""

from unittest.mock import patch

from zipsai.complaint import node
from zipsai.contracts.converse import (
    ComplaintDraft,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)

_NO_FINALIZATION_CALL = "턴완성 LLM을 부르면 안 되는 시나리오인데 호출됐다"


def _request(text, image_urls=None, draft=None):
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "demo",
            "conversation_id": "demo-conv",
            "turn_id": "demo-turn",
            "trace_id": "demo-trace",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {
                "message_id": "demo-msg",
                "text": text,
                "image_urls": image_urls or [],
            },
            "conversation_history": [],
            "complaint_draft": draft.model_dump(mode="json") if draft else None,
        }
    )


def _run(
    label,
    request,
    *,
    interpret_result,
    image_result,
    finalization_result=_NO_FINALIZATION_CALL,
):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(f"입력 텍스트: {request.message.text!r}")
    print(f"입력 사진: {request.message.image_urls}")
    print(f"기존 초안: {request.complaint_draft}")

    def fail_if_called(**_kwargs):
        raise AssertionError(_NO_FINALIZATION_CALL)

    with (
        patch.object(node, "_interpret_turn", lambda *_a, **_k: interpret_result),
        patch.object(node, "analyze_images", lambda *_a, **_k: image_result),
        patch.object(
            node,
            "generate_structured",
            fail_if_called
            if finalization_result is _NO_FINALIZATION_CALL
            else (lambda **_kwargs: finalization_result),
        ),
    ):
        evidence = node._gather_turn_evidence(request)
        print(
            f"\n[1. 근거 수집] interpretation.switch = {evidence.interpretation.switch}"
        )
        print(
            f"[1. 근거 수집] interpretation.fields = {evidence.interpretation.fields}"
        )
        print(f"[1. 근거 수집] image_analysis = {evidence.image_analysis}")

        progress = node._advance_complaint(request, evidence)
        print(
            f"\n[2. 초안 갱신] 최종 switch(settle 후) = {progress.interpretation.switch}"
        )
        print(f"[2. 초안 갱신] previous_symptom = {progress.previous_symptom!r}")
        print(f"[2. 초안 갱신] current_symptom  = {progress.current_symptom!r}")
        print(f"[2. 초안 갱신] image_symptom    = {progress.image_symptom!r}")
        print(f"[2. 초안 갱신] missing_fields   = {progress.missing_fields}")

        turn = node._finalize_turn(progress, evidence)
        print(
            f"\n[3. 턴 완성] reply_source = {turn.reply_source}  (llm=모델이 문구 생성, fixed=고정문구)"
        )
        print(f"[3. 턴 완성] reply        = {turn.reply!r}")
        print(f"[3. 턴 완성] 최종 draft   = {turn.result.complaint_draft}")


def main():
    # 1) 신규 민원 + 사진: 기존 symptom이 없어 전환 위험이 없으므로 텍스트+VLM을
    #    병렬로 돌린다(코드만 보면 안 드러나지만 _gather_turn_evidence 내부에서
    #    ThreadPoolExecutor로 동시 실행된다). location이 비어 있지만 증상 병합도
    #    필요 없어서 턴완성 LLM은 생략되고 고정 질문으로 바로 끝난다.
    _run(
        "① 신규 민원 + 사진 (병렬 실행, 턴완성 생략)",
        _request("세탁기에서 LE 오류가 떠요", ["https://example.com/washer.jpg"]),
        interpret_result=node._TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 탈수할 때 LE 오류가 표시됨")
        ),
        # 텍스트가 이미 말한 증상과 같은 내용이라 병합할 게 없다 -> 턴완성 생략.
        image_result=ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/washer.jpg",
                    summary="세탁기가 탈수할 때 LE 오류가 표시됨",
                    ocr_text="LE",
                )
            ]
        ),
    )

    # 2) 기존 증상이 있고, 같은 민원에 세부 내용이 덧붙는 턴: 두 증상이 다르므로
    #    증상 병합이 필요해 턴완성 LLM이 실제로 호출된다(여기서는 모킹).
    _run(
        "② 기존 증상 + 세부 내용 추가 (증상 병합 필요 -> 턴완성 LLM 호출)",
        _request(
            "탈수할 때 LE 오류도 떠요",
            draft=ComplaintDraft(location="세탁실", symptom="세탁기가 작동하지 않음"),
        ),
        interpret_result=node._TurnInterpretation(
            ComplaintDraft(symptom="탈수할 때 LE 오류가 표시됨")
        ),
        image_result=None,
        finalization_result=node._TurnFinalization(
            symptom="세탁기가 작동하지 않고 탈수할 때 LE 오류가 표시됨", reply=""
        ),
    )

    # 3) 사진만 있는 턴: 텍스트가 없으니 텍스트 LLM 호출 자체를 생략한다. 이번
    #    브랜치에서 새로 추가된 부분 — VLM이 읽은 내용이 바로 symptom 후보
    #    (image_symptom)가 돼서 최종 draft.symptom에 반영된다.
    _run(
        "③ 사진만 있는 턴 (텍스트 LLM 생략, VLM 결과가 바로 symptom에 반영)",
        _request(None, ["https://example.com/boiler.jpg"]),
        interpret_result=node._TurnInterpretation(),
        image_result=ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/boiler.jpg",
                    summary="보일러 표시창에 E4 오류 코드가 보임",
                    ocr_text="E4",
                )
            ]
        ),
    )

    # 4) 완전히 다른 기기 얘기가 나온 턴: 기존 증상이 있어 전환 가능성이 생기고,
    #    모델이 ask로 판정하면 고정 전환 확인 문구로만 응답한다(턴완성 LLM 없음).
    _run(
        "④ 다른 민원으로 전환 제안 (고정 전환 문구, LLM 호출 없음)",
        _request(
            "에어컨도 F2 오류가 떠요",
            draft=ComplaintDraft(location="세탁실", symptom="세탁기가 작동하지 않음"),
        ),
        interpret_result=node._TurnInterpretation(
            ComplaintDraft(symptom="에어컨에서 F2 오류가 표시됨"), switch="ask"
        ),
        image_result=None,
    )


if __name__ == "__main__":
    main()
