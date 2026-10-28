from openai import OpenAI

from src.config.model_config import load_models


MODELS = load_models("openai")


class OpenAIRepository:
    def __init__(self):
        self.__client = OpenAI()


    def send_request(self, model: str = MODELS.default):
        MODELS.validate(model)
        response = self.__client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": "Alice and Bob are going to a science fair on Friday.",
                },
            ],
            text_format=CalendarEvent,
        )