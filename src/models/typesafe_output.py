from pydantic import BaseModel, Field
from typing import Annotated, Dict, Literal, Union


class ChoiceOutput(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Dict[str, float]


class ScoreOutput(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: Dict[int, str]
    probabilities: Dict[int, float]


class NoulOutput(BaseModel):
    type: Literal["noul"]
    noul: float


# The `type` field decides which model each answer is parsed into.
Answer = Annotated[Union[ChoiceOutput, ScoreOutput, NoulOutput], Field(discriminator="type")]


class TypesafeOutput(BaseModel):
    model: str
    answers: Dict[str, Answer]
