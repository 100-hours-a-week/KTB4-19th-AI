from typing import Any

from zipsai.settings import EMBEDDING_MODEL_ID


def _normalize(
    output: dict[str, Any],
) -> tuple[list[list[float]], list[dict[str, float]]]:
    dense = [[float(value) for value in vector] for vector in output["dense_vecs"]]
    sparse = [
        {str(token): float(weight) for token, weight in weights.items()}
        for weights in output["lexical_weights"]
    ]
    return dense, sparse


class BgeM3Encoder:
    def __init__(self, model_id: str = EMBEDDING_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None

    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_id, use_fp16=False)
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return _normalize(output)
