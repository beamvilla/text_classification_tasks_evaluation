import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.models.classification import ClassificationResult, TaskDefinition
from src.models.usage_logs import TokensUsage
from src.repositories.openai_embedding_repository import OpenAIEmbeddingRepository


class SupportMatrix(NamedTuple):
    labels: list[str]            # the classes, in the order their groups appear in the matrix
    matrix: np.ndarray           # one unit-length support vector per row
    slices: list[slice]          # the rows of each class group, same order as `labels`
    reviews: list[str]           # the text of each row


def build_support_matrix(items: list[dict]) -> SupportMatrix:
    """Stack the support vectors, class by class, into one matrix of unit-length rows.

    `items` are the support set entries ({"review": ..., "label": ..., "embedding": [...]}).
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item)

    rows = [item for group in groups.values() for item in group]
    matrix = np.array([item["embedding"] for item in rows], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A support vector is all zeros")

    slices, start = [], 0
    for group in groups.values():
        slices.append(slice(start, start + len(group)))
        start += len(group)
    return SupportMatrix(list(groups), matrix / norms, slices, [item["review"] for item in rows])


class FewShotLearningClassifier:
    """Classifies a text by the class whose support vectors are most similar to it.

    1. Embed the text with the embedding repository.
    2. Compare that embedding (cosine similarity) with the support vectors of every class at once:
       the support vectors are stacked, normalised, into one matrix, so a single matrix-vector product
       gives all the similarities and each class group is then a slice of the result.
    3. A class scores the mean of its `top_k` highest similarities (the default top_k=1 is simply
       the highest one), and the class with the highest score is the label.

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
        top_k: int = 1,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        self.__repository = repository
        self.__embedding_model = embedding_model
        self.__top_k = top_k

        with open(support_set_path, encoding="utf-8") as f:
            support_set = json.load(f)
        # Vectors from different models live in different spaces, so comparing them is meaningless.
        if support_set["model"] != embedding_model:
            raise ValueError(
                f"{support_set_path}: the support vectors were made with {support_set['model']!r} "
                f"but this classifier embeds texts with {embedding_model!r}"
            )
        self.__dimension: int = support_set["dimension"]
        self.__support = build_support_matrix(support_set["items"])

    @property
    def labels(self) -> list[str]:
        return list(self.__support.labels)

    def scores(self, text: str) -> dict[str, float]:
        """The score of each class for `text`: the mean of its top_k highest similarities."""
        similarities = self.__similarities(text)
        return {
            label: float(np.sort(similarities[rows])[-self.__top_k:].mean())
            for label, rows in zip(self.__support.labels, self.__support.slices)
        }

    def explain(self, text: str, n: int = 3) -> dict[str, list[tuple[float, str]]]:
        """For each class, its `n` nearest support sentences to `text` as (similarity, sentence), best first.

        Shows which support sentences decided a prediction.
        """
        similarities = self.__similarities(text)
        explanation = {}
        for label, rows in zip(self.__support.labels, self.__support.slices):
            order = np.argsort(similarities[rows])[::-1][:n] + rows.start
            explanation[label] = [(float(similarities[i]), self.__support.reviews[i]) for i in order]
        return explanation

    def classify(self, task: TaskDefinition, text: str) -> ClassificationResult:
        self.__check_classes(task)
        scores = self.scores(text)
        # Embedding calls report no token usage here, so TokensUsage stays empty.
        return ClassificationResult(label=max(scores, key=scores.get), usage=TokensUsage())

    def __similarities(self, text: str) -> np.ndarray:
        """Cosine similarity between `text` and every support vector, in matrix row order."""
        query = self.__repository.get_embedding(text, model=self.__embedding_model)
        if len(query) != self.__dimension:
            raise ValueError(f"The text embedding has {len(query)} dimensions, the support vectors have {self.__dimension}")
        query = np.asarray(query, dtype=np.float64)
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("The text embedding is all zeros")
        return self.__support.matrix @ (query / norm)

    def __check_classes(self, task: TaskDefinition) -> None:
        labels = set(self.__support.labels)
        unknown = sorted(labels - set(task.criteria))
        missing = sorted(set(task.criteria) - labels)
        if unknown or missing:
            raise ValueError(
                f"The support set does not match the {task.key!r} task: "
                f"labels not in the task: {unknown}; task classes without support vectors: {missing}"
            )
