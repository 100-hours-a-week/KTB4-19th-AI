from types import SimpleNamespace

from zipsai.indexing.parse import conversion_stats


def test_table_counts_come_from_the_document_and_failures_stay_unknown() -> None:
    result = SimpleNamespace(
        document=SimpleNamespace(tables=[object(), object()], pictures=[object()]),
        status=SimpleNamespace(name="SUCCESS"),
        errors=[
            SimpleNamespace(
                page_no=3,
                error_message="화장실 천장에서 물이 새요",
            )
        ],
    )
    pages = conversion_stats(
        result,
        [{"page": 1, "text": "본문"}, {"page": 3, "text": "   "}],
    )

    assert pages.tables_found == 2
    assert pages.tables_parsed == 2
    assert pages.tables_found > 0
    assert pages.tables_parsed > 0
    assert pages.tables_failed is None
    assert pages.table_fail_reasons is None
    assert pages.empty_pages == 1
    assert pages.images_found == 1
    assert pages.parser_status == "SUCCESS"
    assert pages.ocr_used is None
    assert pages.failed_pages == [3]
    assert "화장실" not in repr(pages.failed_pages)
