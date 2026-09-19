from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from zipsai.errors import DocumentFetchError
from zipsai.integrations.s3 import download


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
