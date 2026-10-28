import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.classifiers.openai_classifier import OpenAIClassifier
from src.classifiers.typesafe_classifier import TypesafeClassifier
from src.models.typesafe_input import JevResultType
from src.usecases.prompt_builder import build_task

TASK = build_task("intent", "intent_system_prompt.md", "intent_classes_definition_en.json")


class OpenAIClassifierTest(unittest.TestCase):
    def setUp(self):
        self.repo = MagicMock()
        self.repo.send_request.return_value = SimpleNamespace(
            output_parsed=TASK.output_model(intent="complaint"),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        )

    def test_returns_label_and_usage(self):
        result = OpenAIClassifier(self.repo).classify(TASK, "แย่มาก")
        self.assertEqual(result.label, "complaint")
        self.assertEqual(
            (result.usage.input_tokens, result.usage.output_tokens, result.usage.reasoning_tokens),
            (10, 5, 2),
        )

    def test_sends_system_prompt_then_text_with_task_schema(self):
        OpenAIClassifier(self.repo).classify(TASK, "แย่มาก")
        kwargs = self.repo.send_request.call_args.kwargs
        self.assertEqual([(m.role, m.content) for m in kwargs["messages"]],
                         [("system", TASK.system_prompt), ("user", "แย่มาก")])
        self.assertIs(kwargs["text_format"], TASK.output_model)

    def test_only_set_options_are_forwarded(self):
        OpenAIClassifier(self.repo).classify(TASK, "x")
        kwargs = self.repo.send_request.call_args.kwargs
        for option in ("model", "temperature", "reasoning_effort"):
            self.assertNotIn(option, kwargs)

        OpenAIClassifier(self.repo, model="gpt-5-mini", reasoning_effort="low").classify(TASK, "x")
        kwargs = self.repo.send_request.call_args.kwargs
        self.assertEqual((kwargs["model"], kwargs["reasoning_effort"]), ("gpt-5-mini", "low"))
        self.assertNotIn("temperature", kwargs)


class TypesafeClassifierTest(unittest.TestCase):
    def setUp(self):
        self.repo = MagicMock()
        self.repo.send_request.return_value = SimpleNamespace(
            model="jev-latest",
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
            answers={"intent": SimpleNamespace(
                type="choice", choice="order_status", confidence=0.9, probabilities={"order_status": 0.9},
            )},
        )

    def test_returns_label_and_usage(self):
        result = TypesafeClassifier(self.repo).classify(TASK, "พัสดุอยู่ไหน")
        self.assertEqual(result.label, "order_status")
        self.assertEqual((result.usage.input_tokens, result.usage.output_tokens), (7, 3))
        self.assertIsNone(result.usage.reasoning_tokens)

    def test_sends_text_as_ticket_and_task_as_choice_question(self):
        TypesafeClassifier(self.repo).classify(TASK, "พัสดุอยู่ไหน")
        kwargs = self.repo.send_request.call_args.kwargs
        self.assertEqual(kwargs["ticket"], "พัสดุอยู่ไหน")
        (question,) = kwargs["question_configs"]
        self.assertEqual(question.title, TASK.key)
        self.assertEqual(question.result_type, JevResultType.CHOICE)
        self.assertEqual(question.instructions, TASK.instructions)
        self.assertEqual(question.criteria, TASK.criteria)
        self.assertNotIn("model", kwargs)

    def test_model_is_forwarded_when_set(self):
        TypesafeClassifier(self.repo, model="jev-latest").classify(TASK, "x")
        self.assertEqual(self.repo.send_request.call_args.kwargs["model"], "jev-latest")


if __name__ == "__main__":
    unittest.main()
