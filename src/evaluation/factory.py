import logging

from src.classifiers.base import Classifier
from src.classifiers.openai_classifier import OpenAIClassifier
from src.classifiers.typesafe_classifier import TypesafeClassifier
from src.config.config import OpenAIRunConfig, RunConfig
from src.repositories.openai_repository import MODELS as OPENAI_MODELS
from src.repositories.openai_repository import OpenAIRepository
from src.repositories.typesafe_repository import TypesafeRepository
from src.usecases import IntentClassificationUsecase, SentimentAnalysisUsecase

logger = logging.getLogger(__name__)

USECASES = {
    "intent": IntentClassificationUsecase,
    "sentiment": SentimentAnalysisUsecase,
}


def build_classifier(run: RunConfig) -> Classifier:
    if isinstance(run, OpenAIRunConfig):
        temperature, reasoning_effort = run.temperature, run.reasoning_effort
        # Drop the parameter the model does not support here, once, instead of letting the
        # repository warn on every sample.
        if OPENAI_MODELS.is_reasoning(run.model):
            if temperature is not None:
                logger.warning("Bypassing temperature=%s: %r is a reasoning model", temperature, run.model)
                temperature = None
        elif reasoning_effort is not None:
            logger.warning("Bypassing reasoning_effort=%s: %r is not a reasoning model", reasoning_effort, run.model)
            reasoning_effort = None
        return OpenAIClassifier(
            OpenAIRepository(max_retries=run.max_retries),
            model=run.model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    return TypesafeClassifier(TypesafeRepository(max_retries=run.max_retries), model=run.model)
