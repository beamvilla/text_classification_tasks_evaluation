import csv
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.config import DatasetConfig, EvalConfig, OpenAIRunConfig, TypesafeRunConfig
from src.evaluation import factory
from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.runner import (
    MAX_CONSECUTIVE_ERRORS, Sample, load_samples, run_evaluation, validate_labels,
)
from src.models.classification import ClassificationResult
from src.models.usage_logs import TokensUsage
from src.usecases.prompt_builder import build_task


class MetricsTest(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(accuracy(["a", "b", "a", "b"], ["a", "b", "b", "b"]), 0.75)

    def test_macro_f1_is_mean_of_per_class_f1(self):
        # class a: tp=1 fp=0 fn=1 -> 2/3 ; class b: tp=2 fp=1 fn=0 -> 4/5
        self.assertAlmostEqual(macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"]), (2 / 3 + 4 / 5) / 2)

    def test_failed_prediction_is_a_miss_and_adds_no_class(self):
        self.assertAlmostEqual(macro_f1(["a", "b"], [None, "b"]), 0.5)


class DatasetFileMixin:
    def write(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def dataset(self, text, delimiter=",", **kwargs):
        return DatasetConfig(path=self.write(text), delimiter=delimiter,
                             text_column="review", label_column="label", **kwargs)


class LoadSamplesTest(DatasetFileMixin, unittest.TestCase):
    def test_reads_columns_by_name(self):
        samples = load_samples(self.dataset("label,review,extra\npos,ดี,x\nneg,แย่,y\n"))
        self.assertEqual(samples, [Sample("ดี", "pos"), Sample("แย่", "neg")])

    def test_tolerates_leading_blank_line_and_single_cell_header(self):
        # the shipped sentiment dataset is written like this
        samples = load_samples(self.dataset('\n"review\tlabel"\nดี\tpos\nแย่\tneg\n', delimiter="\t"))
        self.assertEqual([s.label for s in samples], ["pos", "neg"])

    def test_missing_column_names_available_columns(self):
        with self.assertRaisesRegex(ValueError, "columns are"):
            load_samples(self.dataset("a,b\n1,2\n"))

    def test_max_samples_is_a_reproducible_subset(self):
        text = "review,label\n" + "".join(f"t{i},pos\n" for i in range(50))
        first = load_samples(self.dataset(text, max_samples=5, seed=1))
        again = load_samples(self.dataset(text, max_samples=5, seed=1))
        other = load_samples(self.dataset(text, max_samples=5, seed=2))
        self.assertEqual(len(first), 5)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_shipped_datasets_only_use_defined_classes(self):
        from src.config.config import load_eval_config
        from src.evaluation.factory import USECASES
        config = load_eval_config()
        for name, usecase_config in config.usecases.items():
            with self.subTest(usecase=name):
                samples = load_samples(usecase_config.dataset)
                self.assertGreater(len(samples), 100)
                validate_labels(samples, USECASES[name](classifier=None).task, usecase_config.dataset.path)

    def test_unknown_label_is_rejected(self):
        task = build_task("sentiment", "sentiment_system_prompt.md", "sentiment_classes_definition_en.json")
        with self.assertRaisesRegex(ValueError, "neutral"):
            validate_labels([Sample("x", "neutral")], task, Path("d.csv"))


class FakeClassifier:
    """Predicts the label given in `answers` by text, or fails for texts mapped to an Exception."""
    def __init__(self, answers):
        self.answers = answers

    def classify(self, task, text):
        answer = self.answers[text]
        if isinstance(answer, Exception):
            raise answer
        return ClassificationResult(label=answer, usage=TokensUsage(input_tokens=10, output_tokens=2))


class RunEvaluationTest(DatasetFileMixin, unittest.TestCase):
    def config(self, dataset_text, runs, output_dir):
        dataset = {"path": self.write(dataset_text), "delimiter": ",", "text_column": "review", "label_column": "label"}
        return EvalConfig.model_validate({
            "output_dir": output_dir,
            "usecases": {"sentiment": {"dataset": dataset, "runs": runs}},
        })

    def test_writes_predictions_and_summary_per_run(self):
        with tempfile.TemporaryDirectory() as out:
            config = self.config("review,label\nดี,pos\nแย่,neg\nเฉยๆ,neg\n",
                                 [{"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0},
                                  {"provider": "typesafe"}], out)
            answers = {"ดี": "pos", "แย่": "neg", "เฉยๆ": "pos"}  # last one wrong
            output_dir, rows = run_evaluation(config, classifier_factory=lambda run: FakeClassifier(answers),
                                              log=lambda msg: None)

            self.assertEqual([r["provider"] for r in rows], ["openai", "typesafe"])
            self.assertAlmostEqual(rows[0]["accuracy"], 0.6667, places=3)
            self.assertEqual((rows[0]["samples"], rows[0]["completed"], rows[0]["errors"]), (3, 3, 0))
            self.assertEqual(rows[0]["input_tokens"], 30)
            self.assertGreaterEqual(rows[0]["mean_inference_time_s"], 0)
            self.assertIsNone(rows[0]["reasoning_tokens"])
            self.assertEqual(rows[0]["temperature"], 0)

            names = sorted(p.name for p in output_dir.iterdir())
            self.assertEqual(len(names), 3)  # 2 predictions files + summary.csv
            with open(output_dir / "summary.csv", encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            predictions = next(p for p in output_dir.iterdir() if p.name.startswith("sentiment__1_openai"))
            with open(predictions, encoding="utf-8-sig", newline="") as f:
                predictions_rows = list(csv.DictReader(f))
            self.assertEqual([r["correct"] for r in predictions_rows], ["True", "True", "False"])
            self.assertTrue(all(r["inference_time_s"] != "" for r in predictions_rows))

    def test_a_failed_call_is_recorded_and_the_run_continues(self):
        with tempfile.TemporaryDirectory() as out:
            config = self.config("review,label\nดี,pos\nแย่,neg\n", [{"provider": "typesafe"}], out)
            answers = {"ดี": RuntimeError("boom"), "แย่": "neg"}
            _, rows = run_evaluation(config, classifier_factory=lambda run: FakeClassifier(answers),
                                     log=lambda msg: None)
            self.assertEqual((rows[0]["completed"], rows[0]["errors"], rows[0]["aborted"]), (2, 1, False))
            self.assertEqual(rows[0]["accuracy"], 0.5)

    def test_run_aborts_after_consecutive_errors(self):
        n = MAX_CONSECUTIVE_ERRORS + 3
        text = "review,label\n" + "".join(f"t{i},pos\n" for i in range(n))
        with tempfile.TemporaryDirectory() as out:
            config = self.config(text, [{"provider": "typesafe"}], out)
            answers = {f"t{i}": RuntimeError("bad key") for i in range(n)}
            _, rows = run_evaluation(config, classifier_factory=lambda run: FakeClassifier(answers),
                                     log=lambda msg: None)
            self.assertTrue(rows[0]["aborted"])
            self.assertEqual(rows[0]["completed"], MAX_CONSECUTIVE_ERRORS)

    def test_dry_run_calls_nothing_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as out:
            config = self.config("review,label\nดี,pos\n", [{"provider": "typesafe"}], out)
            messages = []
            result = run_evaluation(config, dry_run=True, classifier_factory=lambda run: self.fail("called"),
                                    log=messages.append)
            self.assertEqual(result, (None, []))
            self.assertEqual(list(Path(out).iterdir()), [])
            self.assertIn("Plan: 1 API calls", messages[0])

    def test_unknown_dataset_label_fails_before_any_call(self):
        with tempfile.TemporaryDirectory() as out:
            config = self.config("review,label\nดี,neutral\n", [{"provider": "typesafe"}], out)
            with self.assertRaisesRegex(ValueError, "neutral"):
                run_evaluation(config, classifier_factory=lambda run: self.fail("called"), log=lambda m: None)


class BuildClassifierTest(unittest.TestCase):
    def build(self, run):
        with patch.object(factory, "OpenAIRepository"), patch.object(factory, "OpenAIClassifier") as classifier_class:
            factory.build_classifier(run)
        return classifier_class.call_args.kwargs

    def test_reasoning_model_drops_temperature_with_one_warning(self):
        run = OpenAIRunConfig(provider="openai", model="gpt-5-mini", temperature=0, reasoning_effort="low")
        with self.assertLogs(factory.logger, level=logging.WARNING) as logs:
            kwargs = self.build(run)
        self.assertEqual(len(logs.records), 1)
        self.assertIsNone(kwargs["temperature"])
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_regular_model_drops_reasoning_effort(self):
        run = OpenAIRunConfig(provider="openai", model="gpt-4.1-mini", temperature=0, reasoning_effort="low")
        with self.assertLogs(factory.logger, level=logging.WARNING):
            kwargs = self.build(run)
        self.assertIsNone(kwargs["reasoning_effort"])
        self.assertEqual(kwargs["temperature"], 0)

    def test_typesafe_run_builds_typesafe_classifier(self):
        with patch.object(factory, "TypesafeRepository") as repo_class, \
             patch.object(factory, "TypesafeClassifier") as classifier_class:
            factory.build_classifier(TypesafeRunConfig(provider="typesafe", max_retries=1))
        repo_class.assert_called_once_with(max_retries=1)
        self.assertEqual(classifier_class.call_args.kwargs["model"], "jev-latest")


if __name__ == "__main__":
    unittest.main()
