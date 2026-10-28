import logging

from openai import OpenAI
from pydantic import BaseModel
from typing import List, Type, TypeVar, Literal

from src.config.model_config import load_models


logger = logging.getLogger(__name__)

MODELS = load_models("openai")

ParsedOutputT = TypeVar("ParsedOutputT", bound=BaseModel)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIRepository:
    def __init__(self):
        self.__client = OpenAI()  # Reads OPENAI_API_KEY.

    def send_request(
        self,
        messages: List[Message],
        text_format: Type[ParsedOutputT],
        model: str = MODELS.default,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ParsedOutputT:
        """Send `messages` and return the response parsed into `text_format`.

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
        return response.output_parsed
