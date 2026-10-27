from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, SystemOneResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from enum import Enum

from exceptions.model_not_found_exception import ModelNotFoundError



AVAILABLE_MODELS = ["jev-latest"]


class JevResultTypes(Enum):
    Choice = 1
    Score = 2
    Noul = 3


class QuestionConfig(BaseModel):
    title: str
    result_type: JevResultTypes
    instructions: str
    # choice: {label: description} | score: [description, ...] ordered from score 0 | noul: {"true": ..., "false": ...} (optional)
    criteria: Optional[Union[Dict[str, str], List[str]]] = None


class TypesafeRepository:
    def __init__(self):
        pass

    @staticmethod
    def convert_configs_to_questions(
        question_configs: List[QuestionConfig]
    ) -> Dict[str, Union[Choice, Noul, Score]]:
        revised_question_config_format = {}
        for q in question_configs:
            if q.result_type == JevResultTypes.Choice:
                revised_question_config_format[q.title] = Choice(
                    instructions=q.instructions,
                    criteria=q.criteria,
                )
            elif q.result_type == JevResultTypes.Score:
                revised_question_config_format[q.title] = Score(
                    instructions=q.instructions,
                    criteria=q.criteria if isinstance(q.criteria, list) else list(q.criteria.values()),
                )
            elif q.result_type == JevResultTypes.Noul:
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
