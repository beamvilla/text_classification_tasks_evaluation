from dataclasses import dataclass

from pydantic import BaseModel

from src.models.usage_logs import TokensUsage


@dataclass(frozen=True)
class TaskDefinition:
    """A classification task, independent of which provider runs it."""
    key: str                      # e.g. "intent"
    instructions: str             # the question being asked, from the definition file
    criteria: dict[str, str]      # {class_name: definition}
    system_prompt: str            # used by providers that take a prompt (OpenAI)
    output_model: type[BaseModel] # used by providers that take a schema (OpenAI)


class ClassificationResult(BaseModel):
    label: str
    usage: TokensUsage
    # Seconds the usecase waited for the classifier (includes any retries); set by the usecase.
    inference_time: float | None = None
