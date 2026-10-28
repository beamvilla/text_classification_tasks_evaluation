import tempfile
import unittest
from pathlib import Path

from src.config.model_config import MODELS_CONFIG_PATH, load_models
from src.exceptions import ModelNotFoundError


class LoadModelsTest(unittest.TestCase):
    def write_config(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_real_config_has_valid_defaults_for_every_provider(self):
        for provider in ("openai", "typesafe"):
            models = load_models(provider, MODELS_CONFIG_PATH)
            self.assertIn(models.default, models.available)

    def test_validate_returns_allowed_model(self):
        path = self.write_config('[p]\ndefault = "a"\navailable = ["a", "b"]\n')
        self.assertEqual(load_models("p", path).validate("b"), "b")

    def test_validate_rejects_unknown_model(self):
        path = self.write_config('[p]\ndefault = "a"\navailable = ["a"]\n')
        with self.assertRaises(ModelNotFoundError):
            load_models("p", path).validate("zzz")

    def test_default_not_in_available_raises(self):
        path = self.write_config('[p]\ndefault = "x"\navailable = ["a"]\n')
        with self.assertRaises(ModelNotFoundError):
            load_models("p", path)


if __name__ == "__main__":
    unittest.main()
