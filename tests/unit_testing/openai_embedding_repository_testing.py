import unittest
from unittest.mock import patch

from src.repositories.openai_embedding_repository import OpenAIEmbeddingRepository
from src.repositories.openai_repository import OpenAIRepository


class OpenAIBaseRepositoryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.repositories.openai_base.OpenAI")
        self.openai_class = patcher.start()
        self.client = self.openai_class.return_value
        self.addCleanup(patcher.stop)

    def test_defaults_leave_base_url_and_api_key_to_the_sdk(self):
        OpenAIEmbeddingRepository()
        self.openai_class.assert_called_once_with(max_retries=3)

    def test_base_url_and_api_key_are_passed_through(self):
        OpenAIEmbeddingRepository(base_url="http://localhost:8080/v1", api_key="-")
        self.openai_class.assert_called_once_with(max_retries=3, base_url="http://localhost:8080/v1", api_key="-")

    def test_chat_repository_shares_the_same_client_setup(self):
        OpenAIRepository(base_url="http://localhost:8080/v1")
        self.openai_class.assert_called_once_with(max_retries=3, base_url="http://localhost:8080/v1")

    def test_get_embedding_returns_the_vector(self):
        self.client.embeddings.create.return_value.data = [type("D", (), {"embedding": [0.1, 0.2]})()]
        result = OpenAIEmbeddingRepository().get_embedding("hello", model="m")
        self.assertEqual(result, [0.1, 0.2])
        self.client.embeddings.create.assert_called_once_with(model="m", input="hello")


if __name__ == "__main__":
    unittest.main()
