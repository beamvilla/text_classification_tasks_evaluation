from src.models.classification import ClassificationResult, TaskDefinition
from src.models.typesafe_input import JevResultType, QuestionConfig
from src.models.typesafe_output import ChoiceOutput, TypesafeOutput
from src.models.usage_logs import TokensUsage
from src.repositories.typesafe_repository import TypesafeRepository


class TypesafeClassifier:
    def __init__(self, repository: TypesafeRepository, model: str | None = None):
        self.__repository = repository
        self.__options = {} if model is None else {"model": model}

    def classify(self, task: TaskDefinition, text: str) -> ClassificationResult:
        # TypeSafe builds its own prompt from instructions + criteria, so the
        # task's system_prompt and output_model are not used here.
        question = QuestionConfig(
            title=task.key,
            result_type=JevResultType.CHOICE,
            instructions=task.instructions,
            criteria=task.criteria,
        )
        response = self.__repository.send_request(
            ticket=text,
            question_configs=[question],
            **self.__options,
        )
        answer = TypesafeOutput.model_validate(response, from_attributes=True).answers[task.key]
        if not isinstance(answer, ChoiceOutput):
            raise TypeError(f"Expected a choice answer for {task.key!r}, got {answer.type!r}")
        return ClassificationResult(
            label=answer.choice,
            usage=TokensUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
