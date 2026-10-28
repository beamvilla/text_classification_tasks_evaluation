from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("./")

from src.usecases import IntentClassificationUsecase, SentimentAnalysisUsecase
from src.classifiers import OpenAIClassifier, TypesafeClassifier
from src.repositories import OpenAIRepository, TypesafeRepository


text = "สั่งของไปหลายอาทิตแล้ว ยังไม่เห็นมาเลย ช้ามาก"
# classifier = OpenAIClassifier(
#     repository=OpenAIRepository(),
#     model="gpt-5",
#     reasoning_effort="minimal"
# )

classifier = TypesafeClassifier(
    repository=TypesafeRepository(),
    model="jev-latest"
)

usecase = SentimentAnalysisUsecase(classifier)


result = usecase.predict(text)
print(result)