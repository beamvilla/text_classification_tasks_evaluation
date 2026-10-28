from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient, SystemOneResponse
from typing import Dict, List, Union

from src.config.model_config import load_models
from src.models.typesafe_input import QuestionConfig, JevResultType


def _build_choice(config: QuestionConfig) -> Choice:
    return Choice(instructions=config.instructions, criteria=config.criteria)


def _build_score(config: QuestionConfig) -> Score:
    return Score(instructions=config.instructions, criteria=config.criteria)


def _build_noul(config: QuestionConfig) -> Noul:
    return Noul(instructions=config.instructions, criteria=config.criteria)


_QUESTION_BUILDERS = {
    JevResultType.CHOICE: _build_choice,
    JevResultType.SCORE: _build_score,
    JevResultType.NOUL: _build_noul,
}

MODELS = load_models("typesafe")
DEFAULT_MAX_RETRIES = 3


class TypesafeRepository:
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        # The SDK retries connection errors, timeouts, 408, 429 and 5xx with exponential
        # backoff (honouring Retry-After); 0 disables retrying.
        self.__retry = RetryPolicy(max_retries=max_retries)

    @staticmethod
    def convert_configs_to_questions(
        question_configs: List[QuestionConfig]
    ) -> Dict[str, Union[Choice, Noul, Score]]:
        return {q.title: _QUESTION_BUILDERS[q.result_type](q) for q in question_configs}

    def send_request(
        self,
        ticket: str,
        question_configs: List[QuestionConfig],
        model: str = MODELS.default
    ) -> SystemOneResponse:
        MODELS.validate(model)

        with TypeSafeClient(model=model, retry=self.__retry) as client:  # Reads TYPESAFE_API_KEY.
            response = client.system_one(
                state=ticket,
                questions=self.convert_configs_to_questions(question_configs)
            )
        return response
