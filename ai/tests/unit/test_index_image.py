import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError
from PIL import Image

from zipsai.errors import PdfParseError
from zipsai.indexing import image_read
from zipsai.indexing.image_read import (
    LONG_EDGE,
    MIN_OUTPUT_TOKENS,
    RUNAWAY_CHARS,
    call_index_vlm,
    is_runaway,
    output_budget,
    prepare_image,
    read_image_pages,
)
from zipsai.indexing.reasons import retryable_for
from zipsai.indexing.vlm_table import pipe_runs_to_markdown
from zipsai.settings import Settings


def _save(tmp_path: Path, size: tuple[int, int], orientation: int | None = None) -> Path:
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", size, "white")
    image.putpixel((0, 0), (255, 0, 0))
    if orientation is None:
        image.save(path, format="JPEG", quality=90)
        return path
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, format="JPEG", quality=90, exif=exif)
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
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish,
            )
        ]
    )


def _ok_text() -> str:
    return "관리사무소 안내문 본문입니다"


def test_prepare_rotates_before_shrink_and_does_not_upscale(tmp_path: Path) -> None:
    wide = prepare_image(_save(tmp_path, (4000, 3000), orientation=6))
    assert wide.orig_size == (4000, 3000)
    assert wide.sent_size == (1536, 2048)
    assert max(wide.sent_size) == LONG_EDGE

    small_path = tmp_path / "small.jpg"
    Image.new("RGB", (80, 40), "white").save(small_path, format="JPEG", quality=90)
    small = prepare_image(small_path)
    assert small.orig_size == (80, 40)
    assert small.sent_size == (80, 40)

    exact = tmp_path / "exact.jpg"
    Image.new("RGB", (LONG_EDGE, 100), "white").save(exact, format="JPEG")
    assert prepare_image(exact).sent_size == (LONG_EDGE, 100)


def test_runaway_catches_only_the_long_output() -> None:
    assert is_runaway("가" * (RUNAWAY_CHARS + 1))
    assert not is_runaway("가" * 2423)
    repeated = "\n".join(["같은줄반복"] * 5)
    assert is_runaway(repeated)
    assert not is_runaway("\n".join(["같은줄반복"] * 4))


def test_each_vlm_failure_falls_through_to_paddle(monkeypatch, tmp_path: Path) -> None:
    path = _save(tmp_path, (64, 64))
    seen = {"paddle": 0, "vlm": 0}

    def paddle(_image):
        seen["paddle"] += 1
        return "대체 경로로 읽은 안내문입니다", 0.2

    monkeypatch.setattr(image_read, "call_index_paddle", paddle)
    monkeypatch.setattr(image_read, "prepare_image", lambda _path: image_read.PreparedImage(
        (64, 64), Image.new("RGB", (64, 64), "white"), (64, 64), b"jpeg"
    ))

    cases = [
        Vlm("vlm_error"),
        Vlm("vlm_truncated"),
        Vlm("vlm_runaway"),
        Vlm("vlm_empty"),
        Vlm("vlm_skipped"),
    ]
    for case in cases:
        monkeypatch.setattr(image_read, "call_index_vlm", lambda *_a, _c=case, **_k: _c)
        pages = read_image_pages(path)
        assert pages.parser == "paddleocr"
        assert " | " not in pages[0]["text"] or "대체" in pages[0]["text"]
    assert seen["paddle"] == 5

    monkeypatch.setattr(
        image_read,
        "call_index_vlm",
        lambda *_a, **_k: image_read.VlmResult(_ok_text(), None, 0.1, 10),
    )
    pages = read_image_pages(path)
    assert pages.parser == "qwen-vl"
    assert seen["paddle"] == 5


class Vlm(image_read.VlmResult):
    def __init__(self, reason: str) -> None:
        super().__init__("짧은", reason, 0.1, 10)


def test_openrouter_and_pod_failure_do_not_call_openrouter(monkeypatch) -> None:
    calls = {"complete": 0, "fallback": 0}

    def complete(*_args, **_kwargs):
        calls["complete"] += 1
        raise APIConnectionError(
            message="pod down",
            request=httpx.Request("POST", "https://pod.example/v1/chat/completions"),
        )

    def settings(override=None):
        if override == "openrouter":
            calls["fallback"] += 1
            raise AssertionError("index must not request openrouter")
        return _settings("pod")

    monkeypatch.setattr(image_read, "_vlm_complete", complete)
    monkeypatch.setattr(image_read, "get_settings", settings)
    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")
    failed = call_index_vlm(b"jpeg", 64, 64)
    assert failed.reason == "vlm_error"
    assert calls["complete"] == 1
    assert calls["fallback"] == 0

    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "openrouter")
    skipped = call_index_vlm(b"jpeg", 64, 64)
    assert skipped.reason == "vlm_skipped"
    assert calls["complete"] == 1

    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")
    assert output_budget(4096, 4096) < MIN_OUTPUT_TOKENS
    tight = call_index_vlm(b"jpeg", 4096, 4096)
    assert tight.reason == "vlm_skipped"
    assert calls["complete"] == 1


def test_undiscovered_pod_skips_openrouter_and_uses_paddle(monkeypatch, tmp_path: Path) -> None:
    calls = {"complete": 0, "openrouter": 0}

    def settings(override=None):
        if override == "openrouter":
            calls["openrouter"] += 1
            raise AssertionError("index must not request openrouter")
        return _settings("openrouter")

    def complete(*_args, **_kwargs):
        calls["complete"] += 1
        raise AssertionError("index must not call the model")

    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")
    monkeypatch.setattr(image_read, "get_settings", settings)
    monkeypatch.setattr(image_read, "_vlm_complete", complete)
    skipped = call_index_vlm(b"jpeg", 64, 64)
    assert skipped.reason == "vlm_skipped"
    assert calls["complete"] == 0
    assert calls["openrouter"] == 0

    monkeypatch.setattr(
        image_read,
        "call_index_paddle",
        lambda _image: ("대체 경로로 읽은 안내문입니다", 0.2),
    )
    pages = read_image_pages(_save(tmp_path, (32, 32)))
    assert pages.parser == "paddleocr"
    assert pages.parser_route == "index_paddle"
    assert calls["complete"] == 0
    assert calls["openrouter"] == 0


def test_both_failures_use_existing_parse_reason(monkeypatch, tmp_path: Path) -> None:
    from zipsai.indexing.pipeline import _STAGE_REASON as stage_reason

    path = _save(tmp_path, (32, 32))
    monkeypatch.setattr(
        image_read,
        "call_index_vlm",
        lambda *_a, **_k: image_read.VlmResult(None, "vlm_error", 0.0, 1),
    )
    monkeypatch.setattr(image_read, "call_index_paddle", lambda _image: ("짧다", 0.1))

    with pytest.raises(PdfParseError):
        read_image_pages(path)

    assert stage_reason["parse"] == "PARSE_FAILED"
    assert retryable_for("PARSE_FAILED") is False


def test_table_conversion_applies_to_vlm_only(monkeypatch, tmp_path: Path) -> None:
    path = _save(tmp_path, (32, 32))
    pipes = "관리비 | 2500원\n청소 | 주1회"
    monkeypatch.setattr(
        image_read,
        "call_index_vlm",
        lambda *_a, **_k: image_read.VlmResult(pipes, None, 0.1, 8),
    )
    pages = read_image_pages(path)
    assert pages[0]["text"] == pipe_runs_to_markdown(pipes)
    assert pages[0]["text"] == "관리비: 2500원\n청소: 주1회"

    monkeypatch.setattr(
        image_read,
        "call_index_vlm",
        lambda *_a, **_k: image_read.VlmResult(None, "vlm_error", 0.0, 8),
    )
    monkeypatch.setattr(image_read, "call_index_paddle", lambda _image: (pipes, 0.2))
    pages = read_image_pages(path)
    assert pages[0]["text"] == pipes


def test_vlm_calls_stay_serialized(monkeypatch) -> None:
    state = {"active": 0, "peak": 0}
    gate = threading.Lock()
    monkeypatch.setattr(image_read, "get_settings", lambda override=None: _settings())
    monkeypatch.setenv("ACTIVE_MODEL_PROVIDER", "pod")

    def complete(*_args, **_kwargs):
        with gate:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        threading.Event().wait(0.05)
        with gate:
            state["active"] -= 1
        return _completion(_ok_text())

    monkeypatch.setattr(image_read, "_vlm_complete", complete)
    threads = [
        threading.Thread(target=call_index_vlm, args=(b"jpeg", 64, 64))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert state["peak"] == 1


def test_log_omits_extracted_text(monkeypatch, tmp_path: Path, caplog) -> None:
    secret = "고유문장토큰XYZ안내문본문"
    path = _save(tmp_path, (32, 32))
    monkeypatch.setattr(
        image_read,
        "call_index_vlm",
        lambda *_a, **_k: image_read.VlmResult(secret, None, 0.1, 4),
    )
    with caplog.at_level("INFO", logger="zipsai.indexing.image_read"):
        pages = read_image_pages(path)
    assert secret in pages[0]["text"]
    blob = " ".join(
        " ".join(f"{key}={value}" for key, value in record.__dict__.items())
        for record in caplog.records
    )
    assert secret not in blob
    assert "image_route=vlm" in blob
