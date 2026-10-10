"""합성문서 51건을 로컬 컬렉션에 넣고 235문항을 검색만 채점한다.

LLM은 부르지 않는다. 컬렉션 documents_eval 을 만들며, 끝나면 지우지 않는다.
운영 컬렉션 이름 documents 와 로컬이 아닌 Qdrant 주소는 거절한다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_t1.py --check
    uv run python ../evals/query/run_t1.py --embedding http://127.0.0.1:8001

compose는 embedding 포트를 호스트에 열지 않는다. localhost:8000 은 ai-api 다.
호스트에서 embedding에 붙는 주소를 --embedding 으로 준다.
질의 인코더는 knowledge/retrieve.py 의 query_encoder 와 같이 5초, 재시도 1회,
락 대기 3.5초다. 색인 인코더는 기본값(120초, 재시도 2회)이다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# 스크립트로 실행하면 이 디렉터리가 import 경로에 오른다.
from score import (
    BUILD,
    EXPECTED_FILES,
    EXPECTED_QUESTIONS,
    REPRODUCE,
    GoldItem,
    ManifestDoc,
    Scored,
    aggregate,
    build_files,
    check_inputs,
    format_report,
    load_gold,
    load_manifest,
)

COLLECTION = "documents_eval"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def qdrant_target(url: str, allow_remote: bool) -> str:
    host = urlparse(url).hostname
    if host not in LOCAL_HOSTS and not allow_remote:
        _die("로컬 Qdrant만 사용합니다. 운영 주소로 붙지 않습니다.")
    return url


def embedding_target(url: str | None) -> str:
    chosen = url or os.environ.get("EMBEDDING_URL")
    if not chosen:
        _die(
            "embedding 주소가 없습니다. compose는 embedding 포트를 열지 않고, "
            "localhost:8000 은 ai-api 입니다. "
            "--embedding http://127.0.0.1:8001 처럼 호스트에서 열리는 주소를 주세요."
        )
    host = urlparse(chosen).hostname
    if host == "embedding":
        _die("호스트 이름 embedding 은 컨테이너 안에서만 열립니다. 호스트 주소를 주세요.")
    return chosen.rstrip("/")


def collection_name(name: str) -> str:
    if name == "documents":
        _die("컬렉션 documents 는 본색인과 겹칩니다. documents_eval 을 씁니다.")
    return name


def index_one(client, encoder, doc: ManifestDoc, path: Path, collection: str) -> dict[str, object]:
    from zipsai.contracts.indexing import IndexingJobRequest
    from zipsai.errors import EmptyDocumentError
    from zipsai.indexing.chunk import chunk_pages
    from zipsai.indexing.clean import apply_cleaning
    from zipsai.indexing.embed import embed_chunks
    from zipsai.indexing.mask import apply_masking
    from zipsai.indexing.parse import parse_document
    from zipsai.indexing.upsert import upsert_document

    request = IndexingJobRequest(
        building_id=doc.building_id,
        trace_id=f"t1-{doc.doc_id}",
        doc_id=doc.doc_id,
        title=doc.title,
        file_key=str(path),
    )
    try:
        # run_indexing_job 과 같은 순서다. S3 다운로드만 로컬 경로로 바꾼다.
        # 글자가 없거나 임베딩이 거절해도 그 파일만 실패로 적고 다음 파일로 간다.
        pages = parse_document(path)
        masking = apply_masking(pages)
        cleaning = apply_cleaning(masking.pages)
        chunks = chunk_pages(cleaning.pages)
        embedded = embed_chunks(chunks, encoder, request.trace_id)
        if not embedded:
            raise EmptyDocumentError(f"No chunks to store for document '{doc.doc_id}'")
        stored = upsert_document(
            client, request, embedded, masked=bool(masking.detections), collection=collection
        )
    except Exception as exc:  # noqa: BLE001  한 파일의 실패로 나머지 색인을 멈추지 않는다.
        return {
            "doc_id": doc.doc_id,
            "kind": doc.kind,
            "outcome": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "doc_id": doc.doc_id,
        "kind": doc.kind,
        "outcome": "ok",
        "points": stored,
        "masked": bool(masking.detections),
    }


def index_all(client, encoder, docs: list[ManifestDoc], build: Path, collection: str) -> list[dict]:
    from zipsai.integrations.qdrant import ensure_collection

    ensure_collection(client, collection)
    names = build_files(build)
    results = []
    for doc in docs:
        path = names.get(doc.build_name)
        if path is None:
            row = {"doc_id": doc.doc_id, "kind": doc.kind, "outcome": "missing"}
        else:
            row = index_one(client, encoder, doc, path, collection)
        results.append(row)
        print(f"색인 {row['outcome']} {doc.doc_id}", flush=True)
    return results


def _gate(client, collection: str, dense: list[float], building_id: int) -> tuple[bool, float | None]:
    from qdrant_client import models

    from zipsai.integrations.qdrant import DENSE_VECTOR, building_condition
    from zipsai.knowledge.retrieve import SCORE_THRESHOLD

    gate = client.query_points(
        collection_name=collection,
        query=dense,
        using=DENSE_VECTOR,
        query_filter=models.Filter(must=[building_condition(building_id)]),
        limit=1,
        with_payload=False,
    )
    top = gate.points[0].score if gate.points else None
    return top is not None and top >= SCORE_THRESHOLD, top


def score_all(client, encoder, items: list[GoldItem], collection: str) -> list[Scored]:
    from zipsai.knowledge.retrieve import encode_question, search_chunks

    rows: list[Scored] = []
    for index, item in enumerate(items, start=1):
        vector = encode_question(item.question, encoder=encoder, trace_id=item.id)
        passed, top = _gate(client, collection, vector[0], item.building_id)
        points = (
            search_chunks(vector, item.building_id, client=client, collection=collection)
            if passed
            else []
        )
        hit_docs = []
        hit_buildings = []
        for point in points:
            payload = point.payload or {}
            hit_docs.append(str(payload.get("doc_id") or ""))
            hit_buildings.append(str(payload.get("building_id") or ""))
        rows.append(
            Scored(
                item_id=item.id,
                answerable=item.answerable,
                scope=item.scope,
                kind=item.kind,
                passed=passed,
                top_score=top,
                expected_doc_id=item.expected_doc_id,
                requested_building_id=str(item.building_id),
                hit_doc_ids=tuple(hit_docs),
                hit_building_ids=tuple(hit_buildings),
            )
        )
        if index % 20 == 0 or index == len(items):
            print(f"채점 {index}/{len(items)}", flush=True)
    return rows


def _index_lines(results: list[dict]) -> list[str]:
    ok = sum(1 for row in results if row["outcome"] == "ok")
    failed = [row for row in results if row["outcome"] != "ok"]
    lines = [f"색인 성공 {ok}  실패 {len(failed)}"]
    for row in failed:
        detail = row.get("error_type") or row["outcome"]
        lines.append(f"실패 {row['doc_id']}  {detail}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="합성문서 T1 검색 채점")
    parser.add_argument("--check", action="store_true", help="파일과 문항 수만 검사한다")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--out", type=Path, default=None, help="집계 JSON을 이 경로에 쓴다")
    args = parser.parse_args()

    errors = check_inputs(REPRODUCE, BUILD)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    if args.check:
        print(
            f"확인  문서 {EXPECTED_FILES}건  문항 {EXPECTED_QUESTIONS}  "
            "컬렉션은 만들지 않았습니다"
        )
        return

    if args.index_only and args.score_only:
        _die("--index-only 와 --score-only 는 같이 쓰지 않습니다.")

    from qdrant_client import QdrantClient

    from zipsai.integrations.embedding_client import HttpEncoder
    from zipsai.knowledge.retrieve import QUERY_TIMEOUT_SECONDS

    # 환경의 QDRANT_URL·QDRANT_API_KEY 는 쓰지 않는다. 운영 키를 로컬에 붙이지 않기 위해서다.
    url = qdrant_target(args.qdrant, args.allow_remote)
    collection = collection_name(args.collection)
    client = QdrantClient(url=url)
    docs = load_manifest(REPRODUCE)
    items, _gold_errors = load_gold(REPRODUCE, docs)
    index_results: list[dict] | None = None

    if not args.score_only:
        encoder = HttpEncoder(base_url=embedding_target(args.embedding))
        print(f"색인 시작  {url}  컬렉션 {collection}  문서 {len(docs)}", flush=True)
        index_results = index_all(client, encoder, docs, BUILD, collection)

    rows = None
    if not args.index_only:
        if not client.collection_exists(collection_name=collection):
            _die(f"컬렉션 {collection} 이 없습니다. 색인부터 실행하세요.")
        query_encoder = HttpEncoder(
            base_url=embedding_target(args.embedding),
            timeout=QUERY_TIMEOUT_SECONDS,
            attempts=1,
            lock_wait_seconds=3.5,
        )
        print(f"채점 시작  문항 {len(items)}", flush=True)
        rows = score_all(client, query_encoder, items, collection)

    if rows is None:
        print("\n".join(_index_lines(index_results or [])), flush=True)
        return

    metrics = aggregate(rows)
    text = format_report(metrics, index_lines=_index_lines(index_results) if index_results else None)
    print(text, end="")
    if args.out:
        payload = {"metrics": metrics, "index": index_results}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"기록 {args.out}")


if __name__ == "__main__":
    main()
