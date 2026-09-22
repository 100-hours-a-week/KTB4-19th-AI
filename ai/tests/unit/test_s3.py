from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from zipsai.errors import DocumentFetchError
from zipsai.integrations import s3
from zipsai.integrations.s3 import download, parse_s3_location


class FakeS3:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._error = error

    def download_file(self, bucket: str, key: str, target: str) -> None:
        if self._error is not None:
            raise self._error
        self.calls.append((bucket, key, target))
        Path(target).write_bytes(b"%PDF-1.4")


def test_download_writes_into_target_dir_using_basename(tmp_path: Path) -> None:
    client = FakeS3()

    result = download(
        "buildings/101/notice-001.pdf",
        tmp_path,
        bucket="zipsai-files",
        client=client,
    )

    assert result == tmp_path / "notice-001.pdf"
    assert result.read_bytes() == b"%PDF-1.4"
    assert client.calls == [
        ("zipsai-files", "buildings/101/notice-001.pdf", str(result))
    ]


def test_download_without_bucket_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentFetchError):
        download("notice.pdf", tmp_path, bucket=None, client=FakeS3())


def test_download_wraps_client_error_in_typed_error(tmp_path: Path) -> None:
    error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")

    with pytest.raises(DocumentFetchError):
        download("notice.pdf", tmp_path, bucket="zipsai-files", client=FakeS3(error))


def test_download_keeps_traversal_key_inside_target_dir(tmp_path: Path) -> None:
    result = download(
        "../../etc/passwd", tmp_path, bucket="zipsai-files", client=FakeS3()
    )

    assert result.parent == tmp_path


@pytest.mark.parametrize(
    ("file_key", "expected"),
    [
        # 백엔드가 어떤 URL 형식으로 줄지 정해지지 않아 셋 다 받는다.
        ("s3://zipsai-files/a/b.pdf", ("zipsai-files", "a/b.pdf")),
        (
            "https://zipsai-files.s3.ap-northeast-2.amazonaws.com/a/b.pdf",
            ("zipsai-files", "a/b.pdf"),
        ),
        (
            "https://s3.ap-northeast-2.amazonaws.com/zipsai-files/a/b.pdf",
            ("zipsai-files", "a/b.pdf"),
        ),
        # 스킴이 없으면 예전처럼 객체 키다.
        ("a/b.pdf", (None, "a/b.pdf")),
        # 퍼센트 인코딩된 한글 파일명이 그대로 키로 가면 404가 난다.
        ("s3://zipsai-files/a/%EA%B3%B5%EC%A7%80.pdf", ("zipsai-files", "a/공지.pdf")),
    ],
)
def test_parse_s3_location_accepts_every_url_form(
    file_key: str, expected: tuple[str | None, str]
) -> None:
    assert parse_s3_location(file_key) == expected


def test_download_uses_the_bucket_carried_in_the_url(tmp_path: Path) -> None:
    client = FakeS3()

    download(
        "s3://zipsai-files/buildings/101/notice-001.pdf",
        tmp_path,
        bucket=None,
        client=client,
    )

    assert client.calls[0][:2] == ("zipsai-files", "buildings/101/notice-001.pdf")


def test_download_refuses_a_bucket_other_than_the_configured_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # URL에 실린 버킷을 그대로 믿으면 남의 버킷도 읽게 된다.
    monkeypatch.setattr(s3, "S3_BUCKET", "zipsai-files")

    with pytest.raises(DocumentFetchError):
        download("s3://someone-elses-bucket/a.pdf", tmp_path, client=FakeS3())


def test_download_without_an_object_key_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentFetchError):
        download("s3://zipsai-files/", tmp_path, client=FakeS3())
