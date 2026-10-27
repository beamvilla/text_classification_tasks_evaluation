from enum import Enum
from pydantic import BaseModel
from typing import Optional, Union, Dict, List


class JevResultType(Enum):
    CHOICE = 1
    SCORE = 2
    NOUL = 3


class QuestionConfig(BaseModel):
    title: str
    result_type: JevResultType
    instructions: str
    # choice: {label: description} | score: [description, ...] ordered from score 0 | noul: {"true": ..., "false": ...} (optional)
    criteria: Optional[Union[Dict[str, str], List[str]]] = None