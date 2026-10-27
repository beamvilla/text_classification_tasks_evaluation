from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, SystemOneResponse
from typing import Dict, List, Union

from src.exceptions import ModelNotFoundError
from src.models.typesafe_input import QuestionConfig, JevResultType


AVAILABLE_MODELS = ["jev-latest"]


class TypesafeRepository:
    def __init__(self):
        pass

    @staticmethod
    def convert_configs_to_questions(
        question_configs: List[QuestionConfig]
    ) -> Dict[str, Union[Choice, Noul, Score]]:
        revised_question_config_format = {}
        for q in question_configs:
            if q.result_type == JevResultType.CHOICE:
                revised_question_config_format[q.title] = Choice(
                    instructions=q.instructions,
                    criteria=q.criteria,
                )
            elif q.result_type == JevResultType.SCORE:
                revised_question_config_format[q.title] = Score(
                    instructions=q.instructions,
                    criteria=q.criteria if isinstance(q.criteria, list) else list(q.criteria.values()),
                )
            elif q.result_type == JevResultType.NOUL:
                revised_question_config_format[q.title] = Noul(
                    instructions=q.instructions,
                    criteria=q.criteria,
                )
            else:
                raise ValueError(f"Unsupported result_type: {q.result_type}")
        return revised_question_config_format

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
