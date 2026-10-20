import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from zipsai.api.indexing import _dependencies
from zipsai.contracts.comment import (
    CommentBatch,
    CommentDeleteItem,
    CommentDeleteRequest,
    CommentItem,
    CommentRequest,
)
from zipsai.contracts.notification import ItemResult, ResultStatus
from zipsai.contracts.sync_response import CommentResult, SyncBody
from zipsai.indexing.admission import Rejected, acquire, in_flight, release
from zipsai.indexing.comment import delete_comments, upsert_comments
from zipsai.indexing.reasons import retryable_for
from zipsai.metrics import record_indexing_result
from zipsai.observability import bind, elapsed_ms, stage
from zipsai.settings import LOCATION_PLACEHOLDER

logger = logging.getLogger(__name__)
router = APIRouter()

Kind = Literal["upsert", "delete"]


@dataclass
class _Slot:
    raw: Any
    parsed: CommentItem | CommentDeleteItem | None
    validation_fields: str | None
    validation_ms: int


@router.post("/comment")
def upsert(payload: CommentBatch) -> JSONResponse:
    return _handle("upsert", payload)


@router.post("/comment/delete")
def delete(payload: CommentBatch) -> JSONResponse:
    return _handle("delete", payload)


def _handle(kind: Kind, payload: CommentBatch) -> JSONResponse:
    started = perf_counter()
    job_id = str(uuid4())
    model = CommentItem if kind == "upsert" else CommentDeleteItem
    slots = [_slot(raw, model) for raw in payload.items]
    try:
        acquire(None)
    except Rejected as rejected:
        logger.info(
            "indexing_rejected",
            extra={
                "trace_id": payload.trace_id,
                "job_id": job_id,
                "route": "comment" if kind == "upsert" else "comment_delete",
                "error_code": rejected.error_code,
                "doc_id": None,
                "building_id": _single_building(payload.items),
                "items": len(payload.items),
                "indexing_in_flight": in_flight(),
                "retry_after_s": rejected.retry_after,
            },
        )
        return JSONResponse(
            status_code=rejected.status_code,
            content={"trace_id": payload.trace_id, "error_code": rejected.error_code},
            headers={"Retry-After": str(rejected.retry_after)},
        )
    try:
        # 거절된 요청은 indexing_rejected만 남긴다. 접수 로그는 들어간 요청만 comment_done과 짝이 된다.
        logger.info(
            "comment_accepted",
            extra={
                "job_id": job_id,
                "trace_id": payload.trace_id,
                "kind": kind,
                "items": len(payload.items),
                "building_ids": _joined_ids(payload.items, "building_id", unique=True),
                "complaint_detail_ids": _joined_ids(
                    payload.items, "complaint_detail_id", unique=False
                ),
            },
        )
        with bind(job_id=job_id, trace_id=payload.trace_id):
            results, stats = _execute(kind, payload.trace_id, slots)
            durations = _durations(slots, stats)
            comment_results = [_comment_result(item) for item in results]
            for slot, result, duration in zip(
                slots, comment_results, durations, strict=True
            ):
                _log_item(kind, payload.trace_id, job_id, slot, result, duration)
            metric_kind = "comment_upsert" if kind == "upsert" else "comment_delete"
            for item in comment_results:
                record_indexing_result(
                    metric_kind, item.status.value, item.reason_code
                )
            logger.info(
                "comment_done",
                extra={
                    "kind": kind,
                    "items": len(comment_results),
                    "succeeded": sum(
                        item.status is ResultStatus.SUCCEEDED
                        for item in comment_results
                    ),
                    "failed": sum(
                        item.status is ResultStatus.FAILED for item in comment_results
                    ),
                    "skipped": sum(
                        item.status is ResultStatus.SKIPPED
                        for item in comment_results
                    ),
                    "total_ms": elapsed_ms(started),
                    "http_status": 200,
                    "indexing_in_flight": in_flight(),
                    "embed_batches": stats["embed_batches"],
                    "embed_ms": stats["embed_ms"],
                },
            )
        body = SyncBody(trace_id=payload.trace_id, results=comment_results).to_body()
        return JSONResponse(status_code=200, content=body)
    finally:
        release(None)


def _execute(
    kind: Kind, trace_id: str, slots: list[_Slot]
) -> tuple[list[ItemResult], dict[str, object]]:
    stats: dict[str, object] = {"embed_batches": 0, "embed_ms": 0, "item_ms": []}
    results: list[ItemResult | None] = [None] * len(slots)
    for index, slot in enumerate(slots):
        if slot.parsed is None:
            detail_id, updated_at, _building = _identity(slot)
            results[index] = ItemResult(
                status=ResultStatus.SKIPPED,
                reason_code="VALIDATION_FAILED",
                complaint_detail_id=detail_id,
                updated_at=updated_at,
            )
    if not any(slot.parsed is not None for slot in slots):
        return _filled(results), stats
    try:
        with stage("setup", logger):
            client, encoder = _dependencies()
    except Exception:  # noqa: BLE001 — 준비 실패는 아직 비어 있는 칸만 실패로 채운다
        for index, slot in enumerate(slots):
            if results[index] is not None or slot.parsed is None:
                continue
            results[index] = ItemResult(
                status=ResultStatus.FAILED,
                reason_code="SETUP_FAILED",
                complaint_detail_id=slot.parsed.complaint_detail_id,
                updated_at=slot.parsed.updated_at_raw,
            )
        return _filled(results), stats

    if kind == "upsert":
        items = [slot.parsed for slot in slots if isinstance(slot.parsed, CommentItem)]
        produced = upsert_comments(
            CommentRequest(trace_id=trace_id, items=items),
            encoder=encoder,
            client=client,
            stats=stats,
        )
    else:
        items = [
            slot.parsed
            for slot in slots
            if isinstance(slot.parsed, CommentDeleteItem)
        ]
        produced = delete_comments(
            CommentDeleteRequest(trace_id=trace_id, items=items),
            client=client,
            stats=stats,
        )
    cursor = 0
    for index, slot in enumerate(slots):
        if slot.parsed is None:
            continue
        results[index] = produced[cursor]
        cursor += 1
    return _filled(results), stats


def _filled(results: list[ItemResult | None]) -> list[ItemResult]:
    finished: list[ItemResult] = []
    for item in results:
        if item is None:
            raise RuntimeError("comment result missing")
        finished.append(item)
    return finished


def _slot(raw: Any, model: type[CommentItem] | type[CommentDeleteItem]) -> _Slot:
    started = perf_counter()
    try:
        parsed = model.model_validate(raw)
    except ValidationError as error:
        return _Slot(raw, None, _field_names(error), elapsed_ms(started))
    if not isinstance(parsed, (CommentItem, CommentDeleteItem)):
        return _Slot(raw, None, "", elapsed_ms(started))
    return _Slot(raw, parsed, None, elapsed_ms(started))


def _field_names(error: ValidationError) -> str:
    names: list[str] = []
    for item in error.errors():
        for part in item["loc"]:
            if isinstance(part, str) and part not in names:
                names.append(part)
                break
    return ",".join(names)


def _comment_result(item: ItemResult) -> CommentResult:
    return CommentResult(
        status=item.status,
        reason_code=item.reason_code,
        retryable=retryable_for(item.reason_code),
        complaint_detail_id=item.complaint_detail_id,
        updated_at=item.updated_at,
    )


def _durations(slots: list[_Slot], stats: dict[str, object]) -> list[int]:
    raw = stats.get("item_ms")
    item_ms = raw if isinstance(raw, list) else []
    durations: list[int] = []
    cursor = 0
    for slot in slots:
        if slot.parsed is None:
            durations.append(slot.validation_ms)
            continue
        if cursor < len(item_ms):
            durations.append(int(item_ms[cursor]))
        else:
            durations.append(slot.validation_ms)
        cursor += 1
    return durations


def _log_item(
    kind: Kind,
    trace_id: str,
    job_id: str,
    slot: _Slot,
    result: CommentResult,
    duration_ms: int,
) -> None:
    detail_id, updated_at, building_id = _identity(slot)
    logger.info(
        "comment_item_done",
        extra={
            "trace_id": trace_id,
            "job_id": job_id,
            "kind": kind,
            "complaint_detail_id": detail_id,
            "building_id": building_id,
            "updated_at": updated_at,
            "status": result.status.value,
            "reason_code": result.reason_code,
            "retryable": result.retryable,
            "duration_ms": duration_ms,
            "validation_fields": slot.validation_fields,
            **_detail_fields(kind, slot),
        },
    )


def _detail_fields(kind: Kind, slot: _Slot) -> dict[str, object]:
    empty: dict[str, object] = {
        "complaint_type": None,
        "issue_type": None,
        "has_location": None,
        "has_occurred_time": None,
        "symptom_chars": None,
        "comment_chars": None,
    }
    if kind != "upsert" or not isinstance(slot.parsed, CommentItem):
        return empty
    item = slot.parsed
    return {
        "complaint_type": item.complaint_type,
        "issue_type": item.issue_type,
        "has_location": _has_location(item.location),
        "has_occurred_time": _present(item.occurred_time),
        "symptom_chars": len(item.symptom),
        "comment_chars": len(item.comment),
    }


def _present(value: str | None) -> bool:
    return bool(value and value.strip())


def _has_location(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip() not in ("", LOCATION_PLACEHOLDER)


def _identity(slot: _Slot) -> tuple[int | None, str | None, int | None]:
    parsed = slot.parsed
    if parsed is not None:
        return parsed.complaint_detail_id, parsed.updated_at_raw, parsed.building_id
    raw = slot.raw if isinstance(slot.raw, dict) else {}
    return (
        _optional_int(raw.get("complaint_detail_id")),
        _optional_str(raw.get("updated_at")),
        _optional_int(raw.get("building_id")),
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _joined_ids(items: list[Any], key: str, *, unique: bool) -> str:
    values: list[str] = []
    seen: set[int] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        value = _optional_int(raw.get(key))
        if value is None or (unique and value in seen):
            continue
        seen.add(value)
        values.append(str(value))
    return ",".join(values)


def _single_building(items: list[Any]) -> int | None:
    raw = _joined_ids(items, "building_id", unique=True)
    if raw == "" or "," in raw:
        return None
    return int(raw)
