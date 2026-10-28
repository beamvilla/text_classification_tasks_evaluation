import logging

from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel
from typing import List, Type, TypeVar, Literal

from src.config.model_config import load_models


logger = logging.getLogger(__name__)

MODELS = load_models("openai")
DEFAULT_MAX_RETRIES = 3

ParsedOutputT = TypeVar("ParsedOutputT", bound=BaseModel)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIRepository:
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        # Reads OPENAI_API_KEY. The SDK retries connection errors, timeouts, 408, 409, 429 and 5xx
        # with exponential backoff (honouring Retry-After); 0 disables retrying.
        self.__client = OpenAI(max_retries=max_retries)

    def send_request(
        self,
        messages: List[Message],
        text_format: Type[ParsedOutputT],
        model: str = MODELS.default,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ParsedResponse[ParsedOutputT]:
        """Send `messages` and return the full response; `.output_parsed` is an instance of `text_format`
        and `.usage` holds the token counts.

        Works with both kinds of models (listed in config/models.toml):
        - reasoning models use `reasoning_effort`; `temperature` is ignored;
        - regular models use `temperature`; `reasoning_effort` is ignored.
        An ignored parameter is not an error: a warning is logged and it is not sent.

        Raises ModelNotFoundError if `model` is not in config/models.toml.
        """
        MODELS.validate(model)

        params: dict = {}
        if MODELS.is_reasoning(model):
            if temperature is not None:
                logger.warning("Bypassing temperature=%s: %r is a reasoning model and does not support it", temperature, model)
            if reasoning_effort is not None:
                params["reasoning"] = {"effort": reasoning_effort}
        else:
            if reasoning_effort is not None:
                logger.warning("Bypassing reasoning_effort=%s: %r is not a reasoning model and does not support it", reasoning_effort, model)
            if temperature is not None:
                params["temperature"] = temperature

        response = self.__client.responses.parse(
            model=model,
            input=[m.model_dump() for m in messages],
            text_format=text_format,
            **params,
        )
        return response
