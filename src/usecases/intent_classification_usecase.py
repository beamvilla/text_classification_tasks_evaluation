from src.usecases.base import ClassificationUsecase


class IntentClassificationUsecase(ClassificationUsecase):
    task_key = "intent"
    template_file = "intent_system_prompt.md"
    definition_file = "intent_classes_definition_en.json"
