from runpy import run_path


def test_demo_shows_no_results_and_complaint_result_together(capsys):
    run_path("scripts/demo_tool_calling.py")["main"]()

    output = capsys.readouterr().out
    assert "3. 검색 no_results + 민원 초안" in output
    assert "[LOG] agent_decision route=complaint outcome=tool_selected tool_call_count=0" in output
    assert "[LOG] tool_call_completed tool=search_building_docs outcome=success" in output
    assert '"qa_card_draft": {' in output
    assert '"question": "누수 처리 기한"' in output
    assert '"complaint_draft": {' in output
    assert '"issue_type": "leak"' in output
    assert "시나리오 검증: PASS" in output
    assert "4. Action 형식 오류: text → retry → 안전한 자연어 답변" in output
    assert "outcome=protocol_failure" in output
    assert "시나리오 검증: PASS (실행 0회, 재시도 답변)" in output
    assert "이 문자열을 로그에 남기지 마세요" not in output
