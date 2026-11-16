import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError
from PIL import Image

from zipsai.indexing import image_read, parse
from zipsai.indexing.image_read import LONG_EDGE, RUNAWAY_CHARS, read_scan_pages
from zipsai.indexing.parse import parse_document
from zipsai.indexing.vlm_table import pipe_runs_to_markdown
from zipsai.settings import Settings

VLM_TEXT = "소방시설 점검 안내 본문"
OCR_TEXT = "이지오씨알 대체 본문"


def _scan_pdf(path: Path, pages: int) -> Path:
    images = [Image.new("RGB", (595, 842), "white") for _ in range(pages)]
    images[0].save(path, format="PDF", save_all=True, append_images=images[1:])
    return path


def _settings(provider: str = "pod") -> Settings:
    return Settings(
        "pod-key",
        "https://pod.example/v1",
        "pod-model",
        30,
        vlm_model="pod-vlm",
        provider=provider,
    )


def _completion(text: str, finish: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)
        ]
    )


def _use_pod(monkeypatch: pytest.MonkeyPatch, complete) -> None:
    def settings(override=None):
        if override == "openrouter":
            raise AssertionError("index must not request openrouter")
        return _settings("pod")

    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setattr(image_read, "get_settings", settings)
    monkeypatch.setattr(image_read, "_vlm_complete", complete)


def _ocr_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def ocr(_path: Path, page_no: int) -> tuple[str, float]:
        seen.append(page_no)
        return f"{OCR_TEXT} {page_no}", 0.5

    monkeypatch.setattr(parse, "ocr_scan_page", ocr)
    return seen


def test_scan_pdf_goes_to_vlm_first_and_keeps_page_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sizes: list[tuple[int, int]] = []

    def complete(_settings, jpeg, _budget):
        from io import BytesIO

        with Image.open(BytesIO(jpeg)) as image:
            sizes.append(image.size)
        return _completion("항목 | 날짜\n소화기 | 10월 7일")

    _use_pod(monkeypatch, complete)
    seen = _ocr_spy(monkeypatch)

    pages = parse_document(_scan_pdf(tmp_path / "scan.pdf", 2))

    assert [page["page"] for page in pages] == [1, 2]
    assert pages.parser == "qwen-vl"
    assert pages.parser_route == "index_vlm"
    assert pages.route_reason == "text_layer_none"
    assert pages[0]["text"] == pipe_runs_to_markdown("항목 | 날짜\n소화기 | 10월 7일")
    assert seen == []
    assert all(max(size) == LONG_EDGE for size in sizes)


@pytest.mark.parametrize(
    "outcome",
    [
        "timeout",
        _completion("잘린 출력", finish="length"),
        _completion("가" * (RUNAWAY_CHARS + 1)),
        _completion("\n".join(["같은줄반복"] * 5)),
        _completion("   "),
    ],
    ids=["timeout", "length", "too_long", "repeat", "empty"],
)
def test_each_vlm_failure_reads_that_page_with_easyocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome
) -> None:
    def complete(*_args):
        if outcome == "timeout":
            raise APITimeoutError(
                request=httpx.Request("POST", "https://pod.example/v1/chat/completions")
            )
        return outcome

    _use_pod(monkeypatch, complete)
    seen = _ocr_spy(monkeypatch)

    pages = parse_document(_scan_pdf(tmp_path / "scan.pdf", 1))

    assert seen == [1]
    assert pages.parser == "docling"
    assert pages.parser_route == "docling_ocr"
    assert pages[0] == {"page": 1, "text": f"{OCR_TEXT} 1"}


def test_only_failed_page_falls_back_in_multi_page_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def complete(*_args):
        calls["n"] += 1
        if calls["n"] == 2:
            return _completion("잘린 출력", finish="length")
        return _completion(VLM_TEXT)

    _use_pod(monkeypatch, complete)
    seen = _ocr_spy(monkeypatch)

    pages = parse_document(_scan_pdf(tmp_path / "scan.pdf", 3))

    assert seen == [2]
    assert [page["page"] for page in pages] == [1, 2, 3]
    assert pages[0]["text"] == VLM_TEXT
    assert pages[1]["text"] == f"{OCR_TEXT} 2"
    assert pages[2]["text"] == VLM_TEXT
    assert pages.parser == "qwen-vl+docling"
    assert pages.parser_route == "index_vlm_mixed"


def test_scan_never_calls_openrouter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"complete": 0, "openrouter": 0}

    def settings(override=None):
        if override == "openrouter":
            calls["openrouter"] += 1
        return _settings("openrouter")

    def complete(*_args):
        calls["complete"] += 1
        raise AssertionError("index must not call the model")

    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setattr(image_read, "get_settings", settings)
    monkeypatch.setattr(image_read, "_vlm_complete", complete)
    seen = _ocr_spy(monkeypatch)
    path = _scan_pdf(tmp_path / "scan.pdf", 2)

    pages = parse_document(path)
    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "openrouter")
    parse_document(path)

    assert pages.parser == "docling"
    assert seen == [1, 2, 1, 2]
    assert calls == {"complete": 0, "openrouter": 0}


def test_scan_pages_share_the_single_index_vlm_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = {"now": 0, "max": 0, "calls": 0}
    lock = threading.Lock()

    def complete(*_args):
        with lock:
            state["now"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return _completion(VLM_TEXT)

    _use_pod(monkeypatch, complete)
    monkeypatch.setattr(image_read, "INDEX_VLM_CONCURRENCY", 1)
    monkeypatch.setattr(image_read, "_vlm_gate", None)
    paths = [_scan_pdf(tmp_path / f"scan{i}.pdf", 2) for i in range(3)]
    threads = [
        threading.Thread(
            target=read_scan_pages, args=(p,), kwargs={"reason": None, "layer": None}
        )
        for p in paths
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["calls"] == 6
    assert state["max"] == 1


def test_scan_logs_carry_no_read_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"n": 0}

    def complete(*_args):
        calls["n"] += 1
        if calls["n"] == 1:
            return _completion("   ")
        return _completion(VLM_TEXT)

    _use_pod(monkeypatch, complete)
    _ocr_spy(monkeypatch)
    caplog.set_level(logging.INFO, logger="zipsai.indexing.image_read")

    parse_document(_scan_pdf(tmp_path / "scan.pdf", 2))

    records = [r for r in caplog.records if r.getMessage() == "index_scan_page"]
    assert [(r.page, r.scan_route, r.scan_reason) for r in records] == [
        (1, "ocr", "vlm_empty"),
        (2, "vlm", None),
    ]
    for record in caplog.records:
        dumped = repr(record.__dict__)
        assert VLM_TEXT not in dumped
        assert OCR_TEXT not in dumped
