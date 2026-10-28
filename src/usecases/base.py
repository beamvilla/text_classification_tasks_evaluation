import time

from src.classifiers.base import Classifier
from src.models.classification import ClassificationResult
from src.usecases.prompt_builder import build_task


class ClassificationUsecase:
    """Runs one classification task (prompt + classes from prompt_templates/) through a classifier."""

    task_key: str
    template_file: str
    definition_file: str

    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.task = build_task(
            task_key=self.task_key,
            template_file=self.template_file,
            definition_file=self.definition_file,
        )

    def predict(self, text: str) -> ClassificationResult:
        """Label, token usage and inference time (seconds) for one text."""
        started = time.perf_counter()
        result = self.classifier.classify(self.task, text)
        return result.model_copy(update={"inference_time": time.perf_counter() - started})
