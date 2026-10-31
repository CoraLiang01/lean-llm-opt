"""Protocol detection must not confuse missing CA Ollama models with another API."""
import contextlib
import io
import unittest
from unittest.mock import patch

import check_server


class ServerDiagnosticTests(unittest.TestCase):
    def test_ollama_missing_models_is_not_misidentified_as_vllm(self):
        with patch.object(check_server, "preflight", side_effect=check_server.MissingModelsError("missing nomic")), \
             patch.object(check_server, "api") as api, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(check_server.diagnose("http://server:11434"), 1)
            api.assert_not_called()
            self.assertIn("API TYPE: Ollama", stdout.getvalue())

    def test_openai_compatible_service_is_reported_without_claiming_backend(self):
        with patch.object(check_server, "api", return_value={"data": [{"id": "gpt-oss-20b"}]}), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(check_server.diagnose("http://server:8000/v1/"), 2)
            self.assertIn("OpenAI-compatible", stdout.getvalue())

    def test_ollama_passes_without_inference(self):
        with patch.object(check_server, "preflight", return_value={}), \
             patch.object(check_server, "api") as api:
            self.assertEqual(check_server.diagnose("http://server:11434"), 0)
            api.assert_not_called()

    def test_unknown_or_invalid_address_fails_clearly(self):
        with patch.object(check_server, "api", side_effect=ValueError("invalid URL")), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(check_server.diagnose("not-a-url"), 1)
            self.assertIn("CHECK FAILED", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
