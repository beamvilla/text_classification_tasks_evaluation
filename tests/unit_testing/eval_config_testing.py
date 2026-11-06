import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config.config import (DEFAULT_CONFIG_PATH, FewShotRunConfig, OpenAIRunConfig, load_embedding_config,
                               load_eval_config)

DATASET = "dataset: {path: d.csv, text_column: t, label_column: l}"


class LoadEvalConfigTest(unittest.TestCase):
    def load(self, yaml_text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        tmp.write(yaml_text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return load_eval_config(tmp.name)

    def config_with_run(self, run_yaml):
        return f"usecases:\n  intent:\n    {DATASET}\n    runs:\n      - {run_yaml}\n"

    def test_project_config_loads(self):
        config = load_eval_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(set(config.usecases), {"intent", "sentiment"})

    def test_missing_model_gets_provider_default(self):
        config = self.load(self.config_with_run("{provider: openai}"))
        self.assertEqual(config.usecases["intent"].runs[0].model, "gpt-4o-mini")

    def test_run_parameters_are_parsed(self):
        config = self.load(self.config_with_run("{provider: openai, model: gpt-5-mini, reasoning_effort: low, max_retries: 1}"))
        run = config.usecases["intent"].runs[0]
        self.assertIsInstance(run, OpenAIRunConfig)
        self.assertEqual((run.reasoning_effort, run.max_retries), ("low", 1))

    def test_model_outside_allowlist_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.load(self.config_with_run("{provider: openai, model: nope}"))

    def test_misspelled_parameter_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.load(self.config_with_run("{provider: openai, temprature: 0}"))

    def test_parameter_of_another_provider_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.load(self.config_with_run("{provider: typesafe, temperature: 0}"))

    def test_unknown_provider_usecase_and_empty_runs_are_rejected(self):
        bad_configs = [
            self.config_with_run("{provider: anthropic}"),
            f"usecases:\n  topic:\n    {DATASET}\n    runs:\n      - {{provider: openai}}\n",
            f"usecases:\n  intent:\n    {DATASET}\n    runs: []\n",
        ]
        for text in bad_configs:
            with self.subTest(text=text), self.assertRaises(ValidationError):
                self.load(text)

    def test_project_config_has_an_embedding_section(self):
        embedding = load_embedding_config(DEFAULT_CONFIG_PATH)
        self.assertTrue(embedding.base_url and embedding.model)

    def few_shot_config(self, run_yaml, embedding=True):
        support = Path(tempfile.mkdtemp()) / "support.json"
        support.write_text("{}", encoding="utf-8")
        embedding_yaml = "embedding: {base_url: 'http://e/v1', model: m1}\n" if embedding else ""
        return embedding_yaml + self.config_with_run(run_yaml.format(support=support)), support

    def test_few_shot_run_inherits_model_and_base_url_from_the_embedding_section(self):
        text, support = self.few_shot_config("{{provider: few_shot, support_set: {support}, top_k: 3}}")
        run = self.load(text).usecases["intent"].runs[0]
        self.assertIsInstance(run, FewShotRunConfig)
        self.assertEqual((run.model, run.base_url, run.top_k, run.support_set), ("m1", "http://e/v1", 3, support))

    def test_few_shot_run_can_override_the_embedding_section(self):
        text, _ = self.few_shot_config("{{provider: few_shot, support_set: {support}, model: m2, base_url: 'http://o/v1'}}")
        run = self.load(text).usecases["intent"].runs[0]
        self.assertEqual((run.model, run.base_url), ("m2", "http://o/v1"))

    def test_few_shot_run_without_model_or_base_url_is_rejected(self):
        text, _ = self.few_shot_config("{{provider: few_shot, support_set: {support}}}", embedding=False)
        with self.assertRaisesRegex(ValueError, "embedding"):
            self.load(text)

    def test_few_shot_run_with_missing_support_set_or_bad_top_k_is_rejected(self):
        for run in ("{provider: few_shot, support_set: /nope/support.json}",
                    "{provider: few_shot, support_set: /nope/support.json, top_k: 0}"):
            with self.subTest(run=run), self.assertRaises(ValidationError):
                self.load("embedding: {base_url: 'http://e/v1', model: m}\n" + self.config_with_run(run))

    def test_load_embedding_config_requires_the_section(self):
        tmp = Path(tempfile.mkdtemp()) / "c.yaml"
        tmp.write_text("usecases: {}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "embedding"):
            load_embedding_config(tmp)

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load("")


if __name__ == "__main__":
    unittest.main()
