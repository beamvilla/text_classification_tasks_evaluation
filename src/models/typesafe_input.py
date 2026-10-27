from enum import Enum
from pydantic import BaseModel, model_validator
from typing import Optional, Union, Dict, List


class JevResultType(Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class QuestionConfig(BaseModel):
    title: str
    result_type: JevResultType
    instructions: str
    # choice: {label: description} | score: [description, ...] ordered from score 0 | noul: {"true": ..., "false": ...} (optional)
    criteria: Optional[Union[Dict[str, str], List[str]]] = None

    @model_validator(mode="after")
    def check_criteria_matches_result_type(self) -> "QuestionConfig":
        kind = self.result_type
        if kind == JevResultType.CHOICE:
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError(f"'{self.title}': choice needs criteria as a non-empty {{label: description}} dict")
        elif kind == JevResultType.SCORE:
            if not isinstance(self.criteria, list) or not self.criteria:
                raise ValueError(f"'{self.title}': score needs criteria as a non-empty list of descriptions")
        elif kind == JevResultType.NOUL:
            if self.criteria is not None and not (isinstance(self.criteria, dict) and set(self.criteria) <= {"true", "false"}):
                raise ValueError(f"'{self.title}': noul criteria must be omitted or a dict with keys 'true' and/or 'false'")
        return self