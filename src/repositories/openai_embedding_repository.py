from typing import List

from src.repositories.openai_base import OpenAIBaseRepository


class OpenAIEmbeddingRepository(OpenAIBaseRepository):
    def get_embedding(self, text: str, model: str) -> List[float]:
        """Return the embedding vector of `text`.

        `model` is the embedding model name the server expects (e.g. "text-embedding-3-small" on OpenAI,
        "text-embeddings-inference" on a local server); it is not checked against config/models.toml.
        """
        response = self._client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
