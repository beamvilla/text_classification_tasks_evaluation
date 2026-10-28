from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.model_config import load_models
from src.exceptions import ModelNotFoundError
from src.repositories.openai_repository import DEFAULT_MAX_RETRIES as OPENAI_DEFAULT_MAX_RETRIES
from src.repositories.openai_repository import ReasoningEffort
from src.repositories.typesafe_repository import DEFAULT_MAX_RETRIES as TYPESAFE_DEFAULT_MAX_RETRIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

UsecaseName = Literal["intent", "sentiment"]


class _Strict(BaseModel):
    # A typo such as `temprature:` fails loading instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    path: Path
    delimiter: str = ","
    text_column: str
    label_column: str
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = 42


class _RunConfig(_Strict):
    model: str | None = None  # None -> the provider's default in config/models.toml

    @model_validator(mode="after")
    def _check_model(self):
        models = load_models(self.provider)
        if self.model is None:
            self.model = models.default
        try:
            models.validate(self.model)
        except ModelNotFoundError as e:
            raise ValueError(str(e)) from e
        return self


class OpenAIRunConfig(_RunConfig):
    provider: Literal["openai"]
    temperature: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_retries: int = Field(default=OPENAI_DEFAULT_MAX_RETRIES, ge=0)


class TypesafeRunConfig(_RunConfig):
    provider: Literal["typesafe"]
    max_retries: int = Field(default=TYPESAFE_DEFAULT_MAX_RETRIES, ge=0)


RunConfig = Annotated[Union[OpenAIRunConfig, TypesafeRunConfig], Field(discriminator="provider")]


class UsecaseConfig(_Strict):
    dataset: DatasetConfig
    runs: list[RunConfig] = Field(min_length=1)


class EvalConfig(_Strict):
    output_dir: Path = Path("results")
    usecases: dict[UsecaseName, UsecaseConfig] = Field(min_length=1)


def load_eval_config(path: Path | str = DEFAULT_CONFIG_PATH) -> EvalConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is empty or is not a YAML mapping")
    return EvalConfig.model_validate(raw)
