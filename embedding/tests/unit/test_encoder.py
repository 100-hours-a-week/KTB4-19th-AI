import numpy as np

from embedding.encoder import _normalize


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
