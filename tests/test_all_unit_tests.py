import unittest
from pathlib import Path
import sys
sys.path.append("./")

UNIT_TEST_DIR = Path(__file__).resolve().parent / "unit_testing"


# unittest calls `load_tests(loader, standard_tests, pattern)` automatically when it loads this module
# (e.g. `python -m unittest discover -s tests` or `python tests/test_all_unit_tests.py`).
# Whatever suite it returns is used as this module's tests, so the tests here are the ones
# discovered in tests/unit_testing instead of any tests defined in this file.
# Only the function name `load_tests` is recognised; renaming it means unittest finds 0 tests.
def load_tests(
    loader: unittest.TestLoader,
    # Required by the load_tests protocol but unused: tests unittest already found in this
    # module (there are none) and the `-p` discovery pattern (we use our own pattern below).
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Collect every `*_testing.py` module under tests/unit_testing into one suite."""
    return loader.discover(
        start_dir=str(UNIT_TEST_DIR),
        pattern="*_testing.py",
        top_level_dir=str(UNIT_TEST_DIR),
    )


if __name__ == "__main__":
    unittest.main()
