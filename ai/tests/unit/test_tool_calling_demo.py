from runpy import run_path
from types import SimpleNamespace

import httpx


def test_demo_shows_no_results_and_complaint_result_together(capsys):
    run_path("scripts/demo_tool_calling.py")["main"]()

    output = capsys.readouterr().out
    assert "3. 검색 no_results + 민원 초안" in output
    assert "[LOG] agent_decision route=complaint outcome=tool_selected tool_call_count=0" in output
    assert "[LOG] tool_call_completed tool=search_building_docs outcome=success" in output
    assert '"qa_card_draft": {' in output
    assert '"question": "누수 처리 기한"' in output
    assert '"complaint_draft": {' in output
    assert '"issue_type": "plumbing"' in output
    assert "시나리오 검증: PASS" in output
    assert "4. Action 형식 오류: text → retry → 안전한 종료" in output
    assert "outcome=protocol_failure" in output
    assert "시나리오 검증: PASS (도구 실행 없이 실패 닫힘)" in output
    assert "요청한 작업을 처리하지 못했습니다. 다시 말씀해 주세요." in output
    assert "이 문자열을 로그에 남기지 마세요" not in output


def test_live_demo_posts_compound_request_to_converse_api(capsys, monkeypatch):
    module = run_path("scripts/demo_tool_calling.py")
    sent = {}
    response_body = {"data": {"meta": {"model": "Qwen/Qwen3-VL-8B-Instruct"}}}

    def post(url, *, json, timeout):
        sent.update(url=url, payload=json, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            is_success=True,
            json=lambda: response_body,
        )

    monkeypatch.setattr(httpx, "post", post)

    result = module["main"](live=True)

    assert result == 0
    assert sent["url"] == "http://127.0.0.1:8000/api/v3/ai/converse"
    assert sent["payload"]["message"]["text"] == (
        "누수 처리 기한도 알려주고, 욕실 누수 민원도 작성해 주세요."
    )
    assert sent["payload"]["trace_id"].startswith("tool-call-live-")
    assert sent["timeout"] >= 115
    assert "Qwen/Qwen3-VL-8B-Instruct" in capsys.readouterr().out
