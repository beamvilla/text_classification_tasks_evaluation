from src.classifiers.base import Classifier
from src.models.classification import ClassificationResult
from src.usecases.prompt_builder import build_task


class IntentClassificationUsecase:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.task = build_task(
            task_key="intent",
            template_file="intent_system_prompt.md",
            definition_file="intent_classes_definition_en.json",
        )

    def predict(self, text: str) -> ClassificationResult:
        return self.classifier.classify(self.task, text)
