from openai import OpenAI
from pydantic import BaseModel
from typing import List, Type, TypeVar, Literal

from src.config.model_config import load_models


MODELS = load_models("openai")

ParsedOutputT = TypeVar("ParsedOutputT", bound=BaseModel)


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
    ) -> ParsedOutputT:
        """Send `input` messages and return the response parsed into `text_format`.

        Raises ModelNotFoundError if `model` is not in config/models.toml.
        """
        MODELS.validate(model)
        response = self.__client.responses.parse(
            model=model,
            input=[m.model_dump() for m in messages],
            text_format=text_format,
        )
        return response.output_parsed
