import json
from pathlib import Path

import numpy as np

from src.models.classification import ClassificationResult, TaskDefinition
from src.models.usage_logs import TokensUsage
from src.repositories.openai_embedding_repository import OpenAIEmbeddingRepository


def build_support_matrix(items: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Stack the support vectors, class by class, into one matrix of unit-length rows.

    `items` are the support set entries ({"label": ..., "embedding": [...]}).
    Returns (labels, matrix, group_starts):
      - labels:       the classes, in the order their groups appear in the matrix
      - matrix:       one normalised support vector per row
      - group_starts: the row where each class group starts, for the per-class maximum
    """
    groups: dict[str, list[list[float]]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item["embedding"])

    matrix = np.array([vector for vectors in groups.values() for vector in vectors], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A support vector is all zeros")
    group_starts = np.cumsum([0] + [len(vectors) for vectors in groups.values()])[:-1]
    return list(groups), matrix / norms, group_starts


class FewShotLearningClassifier:
    """Classifies a text by the class whose support vectors are most similar to it.

    1. Embed the text with the embedding repository.
    2. Compare that embedding (cosine similarity) with the support vectors of every class at once:
       the support vectors are stacked, normalised, into one matrix, so a single matrix-vector product
       gives all the similarities and each class group is then a slice of the result.
    3. A class scores the highest similarity among its own vectors (not the mean), and the
       class with the highest score is the label.

    `support_set_path` is a file written by scripts/create_support_set_vector.py:
        {"model": ..., "dimension": ..., "items": [{"review": ..., "label": ..., "embedding": [...]}, ...]}
    Use one support set per task (intent or sentiment). Its labels must be exactly the classes
    of the task passed to `classify`.
    """

    def __init__(
        self,
        repository: OpenAIEmbeddingRepository,
        support_set_path: Path | str,
        embedding_model: str,
    ):
        self.__repository = repository
        self.__embedding_model = embedding_model

        with open(support_set_path, encoding="utf-8") as f:
            support_set = json.load(f)
        # Vectors from different models live in different spaces, so comparing them is meaningless.
        if support_set["model"] != embedding_model:
            raise ValueError(
                f"{support_set_path}: the support vectors were made with {support_set['model']!r} "
                f"but this classifier embeds texts with {embedding_model!r}"
            )
        self.__dimension: int = support_set["dimension"]

        self.__labels, self.__matrix, self.__group_starts = build_support_matrix(support_set["items"])

    @property
    def labels(self) -> list[str]:
        return list(self.__labels)

    def scores(self, text: str) -> dict[str, float]:
        """The best cosine similarity between `text` and the support vectors of each class."""
        query = self.__repository.get_embedding(text, model=self.__embedding_model)
        if len(query) != self.__dimension:
            raise ValueError(f"The text embedding has {len(query)} dimensions, the support vectors have {self.__dimension}")
        query = np.asarray(query, dtype=np.float64)
        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("The text embedding is all zeros")
        
        similarities = self.__matrix @ (query / norm)  # cosine similarity with every support vector
        best = np.maximum.reduceat(similarities, self.__group_starts)  # the highest one in each class group
        return dict(zip(self.__labels, best.tolist()))

    def classify(self, task: TaskDefinition, text: str) -> ClassificationResult:
        self.__check_classes(task)
        scores = self.scores(text)
        # Embedding calls report no token usage here, so TokensUsage stays empty.
        return ClassificationResult(label=max(scores, key=scores.get), usage=TokensUsage())

    def __check_classes(self, task: TaskDefinition) -> None:
        unknown = sorted(set(self.__labels) - set(task.criteria))
        missing = sorted(set(task.criteria) - set(self.__labels))
        if unknown or missing:
            raise ValueError(
                f"The support set does not match the {task.key!r} task: "
                f"labels not in the task: {unknown}; task classes without support vectors: {missing}"
            )
