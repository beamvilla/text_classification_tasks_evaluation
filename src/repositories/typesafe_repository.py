from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, SystemOneResponse
from typing import Dict, List, Union

from src.exceptions import ModelNotFoundError
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

AVAILABLE_MODELS = ["jev-latest"]


class TypesafeRepository:
    def __init__(self):
        pass

    @staticmethod
    def convert_configs_to_questions(
        question_configs: List[QuestionConfig]
    ) -> Dict[str, Union[Choice, Noul, Score]]:
        return {q.title: _QUESTION_BUILDERS[q.result_type](q) for q in question_configs}

    def send_request(
        self,
        ticket: str,
        question_configs: List[QuestionConfig],
        model: str = "jev-latest"
    ) -> SystemOneResponse:
        if model not in AVAILABLE_MODELS:
            raise ModelNotFoundError
        
        with TypeSafeClient(model=model) as client:  # Reads TYPESAFE_API_KEY.
            response = client.system_one(
                state=ticket,
                questions=self.convert_configs_to_questions(question_configs)
            )
        return response
