from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.model_config import load_models
from src.exceptions import ModelNotFoundError
from src.repositories.openai_base import DEFAULT_MAX_RETRIES as EMBEDDING_DEFAULT_MAX_RETRIES
from src.repositories.openai_repository import DEFAULT_MAX_RETRIES as OPENAI_DEFAULT_MAX_RETRIES
from src.repositories.openai_repository import ReasoningEffort
from src.repositories.typesafe_repository import DEFAULT_MAX_RETRIES as TYPESAFE_DEFAULT_MAX_RETRIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

UsecaseName = Literal["intent", "sentiment"]


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


class FewShotRunConfig(_Strict):
    """Few-shot classification by embedding similarity (see src/classifiers/few_shot_learning_classifier.py).

    `model` and `base_url` default to the top-level `embedding` section of the config file.
    """
    provider: Literal["few_shot"]
    model: str | None = None      # embedding model; must be the one that made the support vectors
    base_url: str | None = None   # OpenAI-compatible embedding server
    support_set: Path             # written by scripts/create_support_set_vector.py; one file per usecase
    top_k: int = Field(default=1, ge=1)   # a class scores the mean of its top_k most similar support vectors
    max_retries: int = Field(default=EMBEDDING_DEFAULT_MAX_RETRIES, ge=0)

    @model_validator(mode="after")
    def _check_support_set(self):
        if not resolve_project_path(self.support_set).is_file():
            raise ValueError(f"support_set {str(self.support_set)!r} does not exist (create it with scripts/create_support_set_vector.py)")
        return self


RunConfig = Annotated[Union[OpenAIRunConfig, TypesafeRunConfig, FewShotRunConfig], Field(discriminator="provider")]


class UsecaseConfig(_Strict):
    dataset: DatasetConfig
    runs: list[RunConfig] = Field(min_length=1)


class EmbeddingConfig(_Strict):
    """The embedding server and model, used by scripts/create_support_set_vector.py and by few_shot runs."""
    base_url: str
    model: str


class EvalConfig(_Strict):
    output_dir: Path = Path("results")
    embedding: EmbeddingConfig | None = None
    usecases: dict[UsecaseName, UsecaseConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _fill_few_shot_runs_from_embedding(self):
        for name, usecase in self.usecases.items():
            for run in usecase.runs:
                if not isinstance(run, FewShotRunConfig):
                    continue
                if self.embedding is not None:
                    run.model = run.model or self.embedding.model
                    run.base_url = run.base_url or self.embedding.base_url
                if run.model is None or run.base_url is None:
                    raise ValueError(
                        f"usecase {name!r}: a few_shot run needs `model` and `base_url`, "
                        "set them on the run or in the top-level `embedding` section"
                    )
        return self


def load_eval_config(path: Path | str = DEFAULT_CONFIG_PATH) -> EvalConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is empty or is not a YAML mapping")
    return EvalConfig.model_validate(raw)


def load_embedding_config(path: Path | str = DEFAULT_CONFIG_PATH) -> EmbeddingConfig:
    """Read only the top-level `embedding` section (for scripts that do not run an evaluation)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "embedding" not in raw:
        raise ValueError(f"{path} has no top-level `embedding` section (base_url and model)")
    return EmbeddingConfig.model_validate(raw["embedding"])
