from src.usecases.base import ClassificationUsecase


class SentimentAnalysisUsecase(ClassificationUsecase):
    task_key = "sentiment"
    template_file = "sentiment_system_prompt.md"
    definition_file = "sentiment_classes_definition_en.json"
