import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config.config import DEFAULT_CONFIG_PATH, OpenAIRunConfig, load_eval_config

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

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load("")


if __name__ == "__main__":
    unittest.main()
