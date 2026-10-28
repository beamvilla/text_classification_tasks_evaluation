from src.models.classification import ClassificationResult, TaskDefinition
from src.models.usage_logs import TokensUsage
from src.repositories.openai_repository import Message, OpenAIRepository, ReasoningEffort


class OpenAIClassifier:
    def __init__(
        self,
        repository: OpenAIRepository,
        model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        self.__repository = repository
        # Only the options that were set are forwarded; the repository supplies the defaults.
        self.__options = {
            key: value
            for key, value in (
                ("model", model),
                ("temperature", temperature),
                ("reasoning_effort", reasoning_effort),
            )
            if value is not None
        }

    def classify(self, task: TaskDefinition, text: str) -> ClassificationResult:
        response = self.__repository.send_request(
            messages=[
                Message(role="system", content=task.system_prompt),
                Message(role="user", content=text),
            ],
            text_format=task.output_model,
            **self.__options,
        )
        usage = response.usage
        reasoning_details = getattr(usage, "output_tokens_details", None)
        return ClassificationResult(
            label=getattr(response.output_parsed, task.key),
            usage=TokensUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=getattr(reasoning_details, "reasoning_tokens", None),
            ),
        )
