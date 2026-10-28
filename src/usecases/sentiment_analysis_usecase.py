from src.classifiers.base import Classifier
from src.models.classification import ClassificationResult
from src.usecases.prompt_builder import build_task


class SentimentAnalysisUsecase:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.task = build_task(
            task_key="sentiment",
            template_file="sentiment_system_prompt.md",
            definition_file="sentiment_classes_definition_en.json",
        )

    def predict(self, text: str) -> ClassificationResult:
        return self.classifier.classify(self.task, text)
