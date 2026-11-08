from openai import OpenAI


DEFAULT_MAX_RETRIES = 3


class OpenAIBaseRepository:
    """Shared base for repositories that talk to an OpenAI-compatible server.

    `base_url` / `api_key` default to the SDK's own lookup (OPENAI_BASE_URL / OPENAI_API_KEY, else the real
    OpenAI service). Pass them to target any OpenAI-compatible server, e.g. a local text-embeddings-inference
    at http://localhost:8080/v1 with api_key="-".

    The SDK retries connection errors, timeouts, 408, 409, 429 and 5xx with exponential backoff
    (honouring Retry-After); max_retries=0 disables retrying.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        kwargs: dict = {"max_retries": max_retries}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)
