import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.classifiers.few_shot_learning_classifier import FewShotLearningClassifier
from src.usecases import SentimentAnalysisUsecase
from src.utils.similarity import cosine_similarity

MODEL = "embed-model"


def write_support_set(directory: str, items: list[tuple[str, list[float]]], model: str = MODEL) -> Path:
    path = Path(directory) / "support.json"
    path.write_text(json.dumps({
        "model": model,
        "dimension": len(items[0][1]),
        "items": [{"review": "x", "label": label, "embedding": vector} for label, vector in items],
    }), encoding="utf-8")
    return path


class FewShotLearningClassifierTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.repository = MagicMock()

    def build(self, items, model=MODEL):
        return FewShotLearningClassifier(self.repository, write_support_set(self.dir.name, items), model)

    def embed_as(self, vector):
        self.repository.get_embedding.return_value = vector

    def task(self):
        return SentimentAnalysisUsecase(MagicMock()).task  # classes: pos, neg

    def test_picks_the_class_with_the_closest_support_vector(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1])])
        self.embed_as([0.9, 0.1])
        self.assertEqual(classifier.classify(self.task(), "t").label, "pos")
        self.embed_as([0.1, 0.9])
        self.assertEqual(classifier.classify(self.task(), "t").label, "neg")

    def test_class_score_is_its_best_vector_not_the_mean(self):
        # pos has one vector right on the query and one far away (mean ~0.5);
        # neg has two vectors at 45 degrees (mean = max ~0.71). The max rule must pick pos.
        classifier = self.build([("pos", [1, 0]), ("pos", [-1, 0]), ("neg", [1, 1]), ("neg", [1, 1])])
        self.embed_as([1, 0])
        scores = classifier.scores("t")
        self.assertAlmostEqual(scores["pos"], 1.0)
        self.assertAlmostEqual(scores["neg"], 0.7071, places=3)
        self.assertEqual(classifier.classify(self.task(), "t").label, "pos")

    def test_scores_match_the_plain_python_cosine_similarity(self):
        rng = random.Random(0)
        vectors = {label: [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(n)] for label, n in (("pos", 5), ("neg", 3))}
        classifier = self.build([(label, v) for label, vs in vectors.items() for v in vs])
        query = [rng.uniform(-1, 1) for _ in range(16)]
        self.embed_as(query)
        scores = classifier.scores("t")
        for label, vs in vectors.items():
            self.assertAlmostEqual(scores[label], max(cosine_similarity(query, v) for v in vs))

    def test_embeds_the_text_with_the_configured_model(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1])])
        self.embed_as([1, 0])
        classifier.classify(self.task(), "hello")
        self.repository.get_embedding.assert_called_once_with("hello", model=MODEL)

    def test_result_has_label_and_empty_usage(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1])])
        self.embed_as([1, 0])
        result = classifier.classify(self.task(), "t")
        self.assertEqual((result.usage.input_tokens, result.usage.output_tokens, result.usage.reasoning_tokens), (None, None, None))

    def test_works_inside_a_usecase_and_reports_inference_time(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1])])
        self.embed_as([0, 1])
        result = SentimentAnalysisUsecase(classifier).predict("t")
        self.assertEqual(result.label, "neg")
        self.assertIsNotNone(result.inference_time)

    def test_rejects_a_support_set_made_with_another_model(self):
        with self.assertRaisesRegex(ValueError, "made with"):
            self.build([("pos", [1, 0]), ("neg", [0, 1])], model="other-model")

    def test_rejects_a_class_without_support_vectors(self):
        classifier = self.build([("pos", [1, 0])])
        self.embed_as([1, 0])
        with self.assertRaisesRegex(ValueError, "without support vectors: \\['neg'\\]"):
            classifier.classify(self.task(), "t")

    def test_rejects_a_label_that_is_not_a_task_class(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1]), ("order_status", [1, 1])])
        self.embed_as([1, 0])
        with self.assertRaisesRegex(ValueError, "not in the task: \\['order_status'\\]"):
            classifier.classify(self.task(), "t")

    def test_rejects_an_embedding_of_a_different_size(self):
        classifier = self.build([("pos", [1, 0]), ("neg", [0, 1])])
        self.embed_as([1, 0, 0])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            classifier.classify(self.task(), "t")


if __name__ == "__main__":
    unittest.main()
