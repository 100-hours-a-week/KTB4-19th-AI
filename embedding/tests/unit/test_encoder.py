import threading
from concurrent.futures import CancelledError, Future, wait

import numpy as np
import pytest

from embedding.encoder import (
    BgeM3Encoder,
    ModelBusyError,
    ModelNotLoadedError,
    _normalize,
)

HOLD = "hold"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeModel:
    """HOLD 텍스트에서 release가 올 때까지 멈춰 워커를 붙잡아 둔다."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[list[str]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_on: set[str] = set()
        self.drop_last_result = False
        self.clock = clock

    def encode(self, texts: list[str], **_: object) -> dict:
        self.calls.append(list(texts))
        if texts == [HOLD]:
            self.entered.set()
            assert self.release.wait(timeout=10)
        if self.clock:
            self.clock.now += 0.5
        if self.fail_on & set(texts):
            raise RuntimeError("model failed")
        returned = texts[:-1] if self.drop_last_result else texts
        return {
            "dense_vecs": np.array([[float(sum(map(ord, t)))] for t in returned]),
            "lexical_weights": [{t: 1.0} for t in returned],
        }


@pytest.fixture
def make_encoder():
    made: list[tuple[BgeM3Encoder, FakeModel]] = []

    def make(model: FakeModel | None = None, **kwargs: object):
        model = model or FakeModel()
        encoder = BgeM3Encoder(**kwargs)
        encoder._model = model
        encoder.start()
        made.append((encoder, model))
        return encoder, model

    yield make
    for encoder, model in made:
        model.release.set()
        encoder.stop()


def hold_worker(encoder: BgeM3Encoder, model: FakeModel) -> list[Future]:
    futures = encoder.submit([HOLD])
    assert model.entered.wait(timeout=10)
    return futures


def drain(*groups: list[Future]) -> None:
    wait([future for group in groups for future in group], timeout=10)


def called(model: FakeModel) -> list[str]:
    return [text for call in model.calls for text in call]


def test_query_runs_before_queued_index(make_encoder) -> None:
    encoder, model = make_encoder()
    held = hold_worker(encoder, model)
    index = [encoder.submit([name], "index") for name in ("i1", "i2", "i3")]
    query = encoder.submit(["q"], "query")

    model.release.set()
    drain(held, query, *index)

    assert called(model) == [HOLD, "q", "i1", "i2", "i3"]


def test_same_priority_keeps_arrival_order(make_encoder) -> None:
    encoder, model = make_encoder()
    held = hold_worker(encoder, model)
    groups = [
        encoder.submit(["i1"], "index"),
        encoder.submit(["q1"], "query"),
        encoder.submit(["i2"], "index"),
        encoder.submit(["q2"], "query"),
    ]

    model.release.set()
    drain(held, *groups)

    assert called(model) == [HOLD, "q1", "q2", "i1", "i2"]


@pytest.mark.parametrize("batch", [1, 4])
def test_query_cuts_in_between_the_texts_of_a_large_request(
    make_encoder, batch: int
) -> None:
    encoder, model = make_encoder(max_index_batch=batch)
    held = hold_worker(encoder, model)
    big = encoder.submit([f"t{i}" for i in range(32)], "index")
    query = encoder.submit(["q"], "query")

    model.release.set()
    drain(held, big, query)

    assert called(model).index("q") == 1


def test_index_texts_are_batched_when_no_query_waits(make_encoder) -> None:
    encoder, model = make_encoder(max_index_batch=2)
    held = hold_worker(encoder, model)
    index = [encoder.submit([name], "index") for name in ("i1", "i2", "i3")]

    model.release.set()
    drain(held, *index)

    assert model.calls == [[HOLD], ["i1", "i2"], ["i3"]]


def test_waiting_query_stops_index_batching(make_encoder) -> None:
    encoder, model = make_encoder(max_index_batch=2)
    held = hold_worker(encoder, model)
    index = [encoder.submit([name], "index") for name in ("i1", "i2")]
    query = encoder.submit(["q"], "query")

    model.release.set()
    drain(held, query, *index)

    assert model.calls == [[HOLD], ["q"], ["i1", "i2"]]


def test_cancelled_query_does_not_block_index_batching(make_encoder) -> None:
    encoder, model = make_encoder(max_index_batch=2)
    held = hold_worker(encoder, model)
    query = encoder.submit(["q"], "query")
    assert query[0].cancel()
    index = [encoder.submit([name], "index") for name in ("i1", "i2")]

    model.release.set()
    drain(held, *index)

    assert model.calls == [[HOLD], ["i1", "i2"]]


def test_batched_results_go_back_to_their_own_texts(make_encoder) -> None:
    encoder, model = make_encoder(max_index_batch=3)
    held = hold_worker(encoder, model)
    first = encoder.submit(["a", "b"], "index")
    second = encoder.submit(["c"], "index")

    model.release.set()
    drain(held, first, second)

    assert [f.result().sparse for f in first + second] == [
        {"a": 1.0},
        {"b": 1.0},
        {"c": 1.0},
    ]
    assert [f.result().dense for f in first + second] == [[97.0], [98.0], [99.0]]


def test_expired_item_never_reaches_the_model(make_encoder) -> None:
    clock = FakeClock()
    encoder, model = make_encoder(clock=clock)
    held = hold_worker(encoder, model)
    late = encoder.submit(["late"], "query", max_wait=1.0)

    clock.now = 5.0
    model.release.set()
    drain(held, late)

    assert isinstance(late[0].exception(), ModelBusyError)
    assert called(model) == [HOLD]


def test_item_that_is_still_queued_at_its_deadline_is_cancelled(make_encoder) -> None:
    encoder, model = make_encoder()
    held = hold_worker(encoder, model)

    with pytest.raises(ModelBusyError):
        encoder.encode(["x"], max_wait=0, priority="query")

    model.release.set()
    drain(held)
    encoder.encode(["after"])
    assert called(model) == [HOLD, "after"]


def test_deadline_stops_applying_once_the_first_text_has_started(
    make_encoder,
) -> None:
    clock = FakeClock()
    encoder, model = make_encoder(clock=clock)
    futures = encoder.submit([HOLD, "second"], "index", max_wait=1.0)
    assert model.entered.wait(timeout=10)

    clock.now = 5.0
    encoder._await_start(futures, deadline=1.0)

    assert not futures[1].cancelled()
    model.release.set()
    drain(futures)
    result = encoder._collect(futures)
    assert result.sparse == [{HOLD: 1.0}, {"second": 1.0}]
    assert called(model) == [HOLD, "second"]


def test_cancelled_item_is_reported_as_busy() -> None:
    encoder = BgeM3Encoder()
    future: Future = Future()
    future.cancel()

    with pytest.raises(ModelBusyError):
        encoder._collect([future])


def test_stop_cancels_waiting_requests_as_busy(make_encoder) -> None:
    encoder, model = make_encoder()
    held = hold_worker(encoder, model)
    waiting = encoder.submit(["w"], "index")

    stopper = threading.Thread(target=encoder.stop)
    stopper.start()
    with pytest.raises(CancelledError):
        waiting[0].exception(timeout=10)
    model.release.set()
    stopper.join(timeout=10)
    drain(held)

    assert waiting[0].cancelled()
    with pytest.raises(ModelBusyError):
        encoder._collect(waiting)
    assert called(model) == [HOLD]


def test_failed_text_cancels_the_rest_of_its_request() -> None:
    encoder = BgeM3Encoder()
    failed: Future = Future()
    failed.set_exception(RuntimeError("model failed"))
    rest = [Future(), Future()]

    with pytest.raises(RuntimeError, match="model failed"):
        encoder._collect([failed, *rest])

    assert all(future.cancelled() for future in rest)


def test_model_error_reaches_the_request_and_the_worker_survives(make_encoder) -> None:
    encoder, model = make_encoder()
    model.fail_on = {"boom"}

    with pytest.raises(RuntimeError, match="model failed"):
        encoder.encode(["boom"])

    assert encoder.ready
    assert encoder.encode(["ok"]).sparse == [{"ok": 1.0}]


def test_short_model_output_fails_the_request_and_the_worker_survives(
    make_encoder,
) -> None:
    encoder, model = make_encoder()
    model.drop_last_result = True

    with pytest.raises(RuntimeError, match="0 dense and 0 sparse vectors for 1 texts"):
        encoder.encode(["a"])

    model.drop_last_result = False
    assert encoder._thread.is_alive()
    assert encoder.ready
    assert encoder.encode(["ok"]).sparse == [{"ok": 1.0}]


def test_unexpected_error_in_the_worker_loop_fails_only_that_batch(
    make_encoder,
) -> None:
    encoder, _ = make_encoder()
    original = encoder._process
    failed: list[bool] = []

    def process(batch):
        if not failed:
            failed.append(True)
            raise ValueError("unexpected")
        original(batch)

    encoder._process = process

    with pytest.raises(ValueError, match="unexpected"):
        encoder.encode(["a"])

    assert encoder._thread.is_alive()
    assert encoder.encode(["ok"]).sparse == [{"ok": 1.0}]


def test_encode_reports_queue_wait_and_inference_time(make_encoder) -> None:
    clock = FakeClock()
    model = FakeModel(clock)
    encoder, _ = make_encoder(model, clock=clock)
    held = hold_worker(encoder, model)
    futures = encoder.submit(["a"], "index")

    clock.now = 3.0
    model.release.set()
    drain(held, futures)
    result = encoder._collect(futures, queue_len=2)

    assert result.queue_wait_ms == 3500
    assert result.infer_ms == 500
    assert result.queue_len == 2


def test_start_is_idempotent_and_stop_makes_the_encoder_unavailable(
    make_encoder,
) -> None:
    encoder, _ = make_encoder()
    worker = encoder._thread

    encoder.start()
    assert encoder._thread is worker
    assert encoder.ready

    encoder.stop()
    encoder.stop()
    assert not encoder.ready
    with pytest.raises(ModelNotLoadedError):
        encoder.encode(["x"])


def test_encode_without_a_loaded_model_raises() -> None:
    with pytest.raises(ModelNotLoadedError):
        BgeM3Encoder().encode(["x"])


def test_normalize_converts_numpy_outputs_to_python_types() -> None:
    dense, sparse = _normalize(
        {
            "dense_vecs": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "lexical_weights": [
                {"one": np.float32(0.5)},
                {"two": np.float64(0.25)},
            ],
        }
    )

    assert dense == [[1.0, 2.0], [3.0, 4.0]]
    assert sparse == [{"one": 0.5}, {"two": 0.25}]
    assert all(isinstance(value, float) for row in dense for value in row)
    assert all(
        isinstance(value, float) for weights in sparse for value in weights.values()
    )
