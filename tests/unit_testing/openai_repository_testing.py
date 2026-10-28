import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from src.exceptions import ModelNotFoundError
from src.repositories.openai_repository import Message, OpenAIRepository


class Sentiment(BaseModel):
    label: str


MESSAGES = [Message(role="user", content="hi")]
REGULAR = "gpt-4o-mini"
REASONING = "gpt-5-mini"


class SendRequestTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.repositories.openai_repository.OpenAI")
        self.openai_class = patcher.start()
        self.client = self.openai_class.return_value
        self.addCleanup(patcher.stop)
        self.client.responses.parse.return_value.output_parsed = Sentiment(label="pos")
        self.repo = OpenAIRepository()

    def sent_kwargs(self):
        return self.client.responses.parse.call_args.kwargs

    def test_client_gets_default_max_retries(self):
        self.openai_class.assert_called_once_with(max_retries=3)

    def test_max_retries_is_configurable(self):
        OpenAIRepository(max_retries=0)
        self.assertEqual(self.openai_class.call_args.kwargs, {"max_retries": 0})

    def test_returns_parsed_output(self):
        result = self.repo.send_request(MESSAGES, Sentiment, model=REGULAR)
        self.assertEqual(result, Sentiment(label="pos"))

    def test_regular_model_passes_temperature(self):
        self.repo.send_request(MESSAGES, Sentiment, model=REGULAR, temperature=0.0)
        self.assertEqual(self.sent_kwargs()["temperature"], 0.0)
        self.assertNotIn("reasoning", self.sent_kwargs())

    def test_regular_model_ignores_reasoning_effort(self):
        with self.assertLogs("src.repositories.openai_repository", level="WARNING") as logs:
            self.repo.send_request(MESSAGES, Sentiment, model=REGULAR, reasoning_effort="low")
        self.assertIn("Bypassing reasoning_effort", logs.output[0])
        self.assertNotIn("reasoning", self.sent_kwargs())

    def test_reasoning_model_passes_effort(self):
        self.repo.send_request(MESSAGES, Sentiment, model=REASONING, reasoning_effort="low")
        self.assertEqual(self.sent_kwargs()["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", self.sent_kwargs())

    def test_reasoning_model_ignores_temperature(self):
        with self.assertLogs("src.repositories.openai_repository", level="WARNING") as logs:
            self.repo.send_request(MESSAGES, Sentiment, model=REASONING, temperature=0.0)
        self.assertIn("Bypassing temperature", logs.output[0])
        self.assertNotIn("temperature", self.sent_kwargs())

    def test_no_optional_params_sends_neither(self):
        self.repo.send_request(MESSAGES, Sentiment, model=REASONING)
        self.assertNotIn("reasoning", self.sent_kwargs())
        self.assertNotIn("temperature", self.sent_kwargs())

    def test_unknown_model_raises(self):
        with self.assertRaises(ModelNotFoundError):
            self.repo.send_request(MESSAGES, Sentiment, model="nope")
        self.client.responses.parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
