import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, create_model

from src.models.classification import TaskDefinition

PROMPT_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "prompt_templates"


def load_definition(definition_file: str, task_key: str) -> dict:
    """Read the `task_key` entry ({type, instructions, criteria}) of a file in prompt_templates/."""
    with open(PROMPT_TEMPLATES_DIR / definition_file, encoding="utf-8") as f:
        return json.load(f)[task_key]


def load_criteria(definition_file: str, task_key: str) -> dict[str, str]:
    """Read {class_name: definition} for `task_key` from a definition file in prompt_templates/."""
    return load_definition(definition_file, task_key)["criteria"]


def format_classes(criteria: dict[str, str]) -> str:
    """Render {class_name: definition} as one '- name: definition' line per class."""
    return "\n".join(f"- {name}: {definition}" for name, definition in criteria.items())


def build_system_prompt(template_file: str, criteria: dict[str, str]) -> str:
    """Fill the `{classes}` placeholder of a template in prompt_templates/ with the classes."""
    template = (PROMPT_TEMPLATES_DIR / template_file).read_text(encoding="utf-8")
    return template.format(classes=format_classes(criteria))


def build_output_model(task_key: str, criteria: dict[str, str]) -> type[BaseModel]:
    """Create the `text_format` for a task: one field named `task_key` restricted to the class names.

    e.g. build_output_model("sentiment", {"pos": ..., "neg": ...}) behaves like
        class SentimentOutput(BaseModel):
            sentiment: Literal["pos", "neg"]
    so the model can only answer with a class that exists in the definition file.
    """
    label_type = Literal[tuple(criteria)]
    return create_model(f"{task_key.capitalize()}Output", **{task_key: (label_type, ...)})


def build_task(task_key: str, template_file: str, definition_file: str) -> TaskDefinition:
    """Load a definition file once and derive everything a provider might need from it."""
    definition = load_definition(definition_file, task_key)
    criteria = definition["criteria"]
    return TaskDefinition(
        key=task_key,
        instructions=definition["instructions"],
        criteria=criteria,
        system_prompt=build_system_prompt(template_file, criteria),
        output_model=build_output_model(task_key, criteria),
    )
