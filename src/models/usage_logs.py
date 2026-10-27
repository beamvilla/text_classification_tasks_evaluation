from pydantic import BaseModel
from typing import Optional


class TokensUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    reasoning_tokens: Optional[int] = None