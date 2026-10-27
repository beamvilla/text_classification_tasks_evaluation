import unittest

from typesafe_sdk import Choice, Noul, Score

from src.repositories.typesafe_repository import (
    TypesafeRepository,
    _build_choice,
    _build_noul,
    _build_score,
)
from src.models.typesafe_input import QuestionConfig, JevResultType


class ConvertConfigsToQuestionsTest(unittest.TestCase):
    def convert(self, configs):
        return TypesafeRepository.convert_configs_to_questions(configs)

    def test_choice_config_becomes_choice_question(self):
        config = QuestionConfig(
            title="intent",
            result_type=JevResultType.CHOICE,
            instructions="What does the customer want?",
            criteria={"order_status": "Asks where the order is", "other": "Anything else"},
        )

        question = self.convert([config])[config.title]

        self.assertIsInstance(question, Choice)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(
            question.criteria,
            config.criteria,
        )

    def test_score_config_with_list_criteria_keeps_order(self):
        config = QuestionConfig(
            title="frustration",
            result_type=JevResultType.SCORE,
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated but civil", "Very angry"],
        )

        question = self.convert([config])[config.title]

        self.assertIsInstance(question, Score)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(list(question.criteria), config.criteria)


    def test_noul_config_with_criteria(self):
        config = QuestionConfig(
            title="is_urgent",
            result_type=JevResultType.NOUL,
            instructions="Is the message urgent?",
            criteria={"true": "Mentions a deadline", "false": "No time pressure"},
        )

        question = self.convert([config])[config.title]

        self.assertIsInstance(question, Noul)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(question.criteria, config.criteria)

    def test_noul_config_without_criteria(self):
        config = QuestionConfig(
            title="is_urgent", 
            result_type=JevResultType.NOUL, 
            instructions="Is the message urgent?"
        )

        question = self.convert([config])[config.title]

        self.assertIsInstance(question, Noul)
        self.assertIsNone(question.criteria)

    def test_multiple_configs_are_keyed_by_title_in_input_order(self):
        configs = [
            QuestionConfig(
                title="intent", 
                result_type=JevResultType.CHOICE, 
                instructions="i", 
                criteria={"a": "A"}
            ),
            QuestionConfig(
                title="frustration", 
                result_type=JevResultType.SCORE, 
                instructions="f", 
                criteria=["low", "high"]
            ),
            QuestionConfig(
                title="is_urgent", 
                result_type=JevResultType.NOUL, 
                instructions="u"
            ),
        ]

        result = self.convert(configs)

        self.assertEqual(list(result), ["intent", "frustration", "is_urgent"])
        self.assertIsInstance(result["intent"], Choice)
        self.assertIsInstance(result["frustration"], Score)
        self.assertIsInstance(result["is_urgent"], Noul)

    def test_empty_config_list_returns_empty_dict(self):
        self.assertEqual(self.convert([]), {})


class BuildQuestionTest(unittest.TestCase):
    def test_build_choice_copies_instructions_and_criteria(self):
        config = QuestionConfig(
            title="intent",
            result_type=JevResultType.CHOICE,
            instructions="What does the customer want?",
            criteria={"order_status": "Asks where the order is", "other": "Anything else"},
        )

        question = _build_choice(config)

        self.assertIsInstance(question, Choice)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(question.criteria, config.criteria)

    def test_build_score_copies_instructions_and_keeps_criteria_order(self):
        config = QuestionConfig(
            title="frustration",
            result_type=JevResultType.SCORE,
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated but civil", "Very angry"],
        )

        question = _build_score(config)

        self.assertIsInstance(question, Score)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(list(question.criteria), config.criteria)

    def test_build_noul_copies_instructions_and_criteria(self):
        config = QuestionConfig(
            title="is_urgent",
            result_type=JevResultType.NOUL,
            instructions="Is the message urgent?",
            criteria={"true": "Mentions a deadline", "false": "No time pressure"},
        )

        question = _build_noul(config)

        self.assertIsInstance(question, Noul)
        self.assertEqual(question.instructions, config.instructions)
        self.assertEqual(question.criteria, config.criteria)

    def test_build_noul_without_criteria_leaves_criteria_unset(self):
        config = QuestionConfig(
            title="is_urgent",
            result_type=JevResultType.NOUL,
            instructions="Is the message urgent?",
        )

        question = _build_noul(config)

        self.assertIsInstance(question, Noul)
        self.assertEqual(question.instructions, config.instructions)
        self.assertIsNone(question.criteria)


if __name__ == "__main__":
    unittest.main()
