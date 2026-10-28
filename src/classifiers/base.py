from typing import Protocol

from src.models.classification import ClassificationResult, TaskDefinition


class Classifier(Protocol):
    """Runs a classification task on one text with a specific provider."""

    def classify(self, task: TaskDefinition, text: str) -> ClassificationResult: ...
