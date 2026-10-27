from pydantic import BaseModel
from typing import Optional


class TokensUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class InferenceTime(BaseModel):
    inference_time: float