import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paths import load_project_env


class EnvLoadingTests(unittest.TestCase):
    def test_load_project_env_overrides_shell_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=from_dotenv_file\n", encoding="utf-8")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "from_shell"}, clear=False):
                with patch("src.paths.ENV_FILE", env_path):
                    load_project_env(override=True)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "from_dotenv_file")


if __name__ == "__main__":
    unittest.main()
