import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.exceptions import ModelNotFoundError


MODELS_CONFIG_PATH = "./config/models.toml"


@dataclass(frozen=True)
class ProviderModels:
    default: str
    available: tuple[str, ...]
    reasoning: tuple[str, ...] = ()

    def validate(self, model: str) -> str:
        """Return `model` if it is allowed, otherwise raise ModelNotFoundError."""
        if model not in self.available:
            raise ModelNotFoundError(f"{model!r} is not in available models: {list(self.available)}")
        return model

    def is_reasoning(self, model: str) -> bool:
        return model in self.reasoning


def load_models(provider: str, path: Path = MODELS_CONFIG_PATH) -> ProviderModels:
    with open(path, "rb") as f:
        section = tomllib.load(f)[provider]
    models = ProviderModels(
        default=section["default"],
        available=tuple(section["available"]),
        reasoning=tuple(section.get("reasoning", [])),
    )
    models.validate(models.default)  # the default must always be allowed
    for model in models.reasoning:
        models.validate(model)  # reasoning models must also be allowed models
    return models
