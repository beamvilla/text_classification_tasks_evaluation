import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from src.models.classification import ClassificationResult
from src.models.usage_logs import TokensUsage
from src.usecases.intent_classification_usecase import IntentClassificationUsecase
from src.usecases.prompt_builder import load_definition
from src.usecases.sentiment_analysis_usecase import SentimentAnalysisUsecase

CASES = [
    (IntentClassificationUsecase, "intent_classes_definition_en.json", "intent"),
    (SentimentAnalysisUsecase, "sentiment_classes_definition_en.json", "sentiment"),
]


class TaskDefinitionTest(unittest.TestCase):
    def test_task_matches_definition_file(self):
        for usecase_class, definition_file, key in CASES:
            with self.subTest(usecase=usecase_class.__name__):
                definition = load_definition(definition_file, key)
                task = usecase_class(MagicMock()).task
                self.assertEqual(task.key, key)
                self.assertEqual(task.instructions, definition["instructions"])
                self.assertEqual(task.criteria, definition["criteria"])

    def test_prompt_contains_every_class_and_definition(self):
        for usecase_class, definition_file, key in CASES:
            with self.subTest(usecase=usecase_class.__name__):
                prompt = usecase_class(MagicMock()).task.system_prompt
                self.assertNotIn("{classes}", prompt)
                for name, definition in load_definition(definition_file, key)["criteria"].items():
                    self.assertIn(f"- {name}: {definition}", prompt)

    def test_output_model_accepts_only_defined_classes(self):
        for usecase_class, definition_file, key in CASES:
            with self.subTest(usecase=usecase_class.__name__):
                output_model = usecase_class(MagicMock()).task.output_model
                for name in load_definition(definition_file, key)["criteria"]:
                    self.assertEqual(getattr(output_model(**{key: name}), key), name)
                with self.assertRaises(ValidationError):
                    output_model(**{key: "not_a_class"})
                self.assertEqual(list(output_model.model_fields), [key])


class PredictTest(unittest.TestCase):
    def test_predict_delegates_to_classifier_with_task_and_text(self):
        for usecase_class, _, _ in CASES:
            with self.subTest(usecase=usecase_class.__name__):
                expected = ClassificationResult(label="x", usage=TokensUsage())
                classifier = MagicMock()
                classifier.classify.return_value = expected

                usecase = usecase_class(classifier)
                result = usecase.predict("สินค้าดีมาก")

                classifier.classify.assert_called_once_with(usecase.task, "สินค้าดีมาก")
                self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
