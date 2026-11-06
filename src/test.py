import sys
sys.path.append("./")

from src.classifiers import FewShotLearningClassifier
from src.repositories import OpenAIEmbeddingRepository

import os
from dotenv import load_dotenv
load_dotenv()


few_shot_learning_classifier = FewShotLearningClassifier(
    repository=OpenAIEmbeddingRepository(
        max_retries=3, 
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        api_key="-"
    ),
    support_set_path="data/support_set_vectors/support_set_intent.json",
    embedding_model=os.getenv("EMBEDDING_MODEL")
)

text = "ส่งช้ามาก ตอนนี้ยังไม่ได้ของ"

# 1) Use it like any other classifier: wrap it in a usecase (the same way OpenAIClassifier is used).
from src.usecases import IntentClassificationUsecase, SentimentAnalysisUsecase


intent_usecase = IntentClassificationUsecase(few_shot_learning_classifier)
result = intent_usecase.predict(text)
print(f"text: {text}")
print(f"label: {result.label}")
print(f"inference time: {result.inference_time:.3f}s")  # embedding request + cosine comparison

# 2) Look at the score of every class (the highest similarity among that class's support vectors).
scores = few_shot_learning_classifier.scores(text)
for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
    print(f"  {label:<18} {score:.4f}")

# 3) Another task needs its own support set, so build another classifier for it.
sentiment_usecase = SentimentAnalysisUsecase(
    FewShotLearningClassifier(
        repository=OpenAIEmbeddingRepository(base_url=os.getenv("EMBEDDING_BASE_URL"), api_key="-"),
        support_set_path="data/support_set_vectors/support_set_sentiment.json",
        embedding_model=os.getenv("EMBEDDING_MODEL"),
    )
)
print(f"sentiment: {sentiment_usecase.predict('สินค้าดีมาก ส่งไวด้วย').label}")
