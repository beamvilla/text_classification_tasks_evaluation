from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, List, Type, TypeVar

from src.config.model_config import load_models


MODELS = load_models("openai")

ParsedOutputT = TypeVar("ParsedOutputT", bound=BaseModel)


class OpenAIRepository:
    def __init__(self):
        self.__client = OpenAI()  # Reads OPENAI_API_KEY.

    def send_request(
        self,
        input: List[Dict[str, str]],
        text_format: Type[ParsedOutputT],
        model: str = MODELS.default,
    ) -> ParsedOutputT:
        """Send `input` messages and return the response parsed into `text_format`.

        Raises ModelNotFoundError if `model` is not in config/models.toml.
        """
        MODELS.validate(model)
        response = self.__client.responses.parse(
            model=model,
            input=input,
            text_format=text_format,
        )
        return response.output_parsed
