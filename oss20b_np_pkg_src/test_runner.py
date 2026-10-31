"""Runner regressions: exact tags, smoke isolation, resume and failure reporting."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("np_server_runner", Path(__file__).with_name("run_airnrm_np.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ServerRunnerTests(unittest.TestCase):
    def test_real_json_parser_accepts_compact_and_pretty_tags(self):
        tags = {"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]}
        for encoding in (json.dumps(tags, separators=(",", ":")), json.dumps(tags, indent=2)):
            with self.subTest(encoding=encoding):
                def urlopen(request, timeout):
                    payload = encoding if request.full_url.endswith("/api/tags") else '{}'
                    return io.BytesIO(payload.encode())
                with patch.object(runner.urllib.request, "urlopen", side_effect=urlopen), \
                     patch.object(runner.shutil, "which", return_value=None):
                    self.assertEqual(runner.preflight("http://remote:11434")["available_tags"],
                                     ["gpt-oss:20b", "nomic-embed-text:latest"])

    def test_unavailable_metadata_does_not_block_existing_models(self):
        def api(base, path, payload=None):
            if path == "/api/tags":
                return {"models": [{"model": "gpt-oss:20b"}, {"model": "nomic-embed-text"}]}
            raise RuntimeError("metadata endpoint disabled by proxy")
        with patch.object(runner, "api", side_effect=api), patch.object(runner.shutil, "which", return_value=None):
            report = runner.preflight("http://remote:11434/api/")
        self.assertEqual(report["ollama_url"], "http://remote:11434")
        self.assertIn("unavailable", report["ollama_version"])

    def test_unreachable_server_has_actionable_error(self):
        with patch.object(runner.urllib.request, "urlopen", side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "--ollama-url"):
                runner.preflight("http://remote:11434")

    def test_html_and_wrong_api_schema_are_distinguished(self):
        with patch.object(runner.urllib.request, "urlopen", return_value=io.BytesIO(b'<html>login</html>')):
            with self.assertRaisesRegex(RuntimeError, "valid JSON"):
                runner.preflight("http://remote:11434")
        with patch.object(runner, "api", return_value={"data": []}):
            with self.assertRaisesRegex(RuntimeError, "server API type"):
                runner.preflight("http://remote:11434")
        with self.assertRaisesRegex(ValueError, "OpenAI-compatible"):
            runner.normalize_ollama_url("http://remote:8000/v1/")

    def test_old_ollama_url_environment_variable_is_supported(self):
        with patch.dict(runner.os.environ, {"OLLAMA_URL": "http://old-server:11434"}, clear=True), \
             patch.object(runner.sys, "argv", ["run_airnrm_np.py", "--dry-run"]), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(runner.main(), 0)
            self.assertIn("http://old-server:11434", stdout.getvalue())

    def test_wrong_oss_tag_is_rejected(self):
        with patch.object(runner, "api", return_value={"models": [
            {"name": "gpt-oss:120b"}, {"name": "nomic-embed-text:latest"}]}):
            with self.assertRaisesRegex(RuntimeError, "gpt-oss:20b"):
                runner.preflight("http://remote:11434")

    def test_remote_endpoint_is_used_for_every_model_check(self):
        requests = []

        def api(base, path, payload=None):
            requests.append((base, path))
            if path == "/api/tags":
                return {"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]}
            return {"version": "test"}

        with patch.object(runner, "api", side_effect=api), patch.object(runner.shutil, "which", return_value=None):
            result = runner.preflight("http://remote:11434")
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(base == "http://remote:11434" for base, _ in requests))
        self.assertIn("runner_host_gpu", result)

    def test_notebook_exit_zero_does_not_hide_failed_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "custom_results"
            previous = output / "LEAN-LLM-OPT__gpt-oss-20b__Air-NRM-NP-LS__r0__old"
            previous.mkdir(parents=True)
            (previous / "instances.jsonl").write_text('{"status":"failed"}\n')
            log = root / "test.log"
            log.write_text("test failure log")
            args = argparse.Namespace(check=False, preflight=False, smoke=False,
                                      ollama_url="http://remote:11434", output_dir=str(output),
                                      repeats=3, fresh=False)
            commands = []

            def run(command, env=None):
                commands.append((command, env))
                return 0

            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "run_command", side_effect=run), \
                 patch.object(runner, "preflight", return_value={"test": True}), \
                 patch.object(runner, "query_rows", return_value=[{}, {}]), \
                 patch.object(runner, "validate_run", return_value=False), \
                 patch.dict(runner.os.environ):
                self.assertEqual(runner.execute(args, log, "test"), 1)
            self.assertEqual(len(commands), 2)  # verifier + first repeat; fail fast
            command, env = commands[-1]
            self.assertEqual(command[command.index("--resume") + 1], previous.name)
            self.assertEqual(env["LEANOPT_OUT_DIR"], str(output.resolve()))
            with tarfile.open(root / "results_airnrm_np_test.tar.gz") as tar:
                names = tar.getnames()
                self.assertIn("runs/" + previous.name + "/instances.jsonl", names)
                self.assertIn("logs/test.log", names)
                self.assertIn("reports/hardware_test.json", names)

    def test_smoke_and_full_cannot_share_custom_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "shared"
            output.mkdir()
            (output / ".airnrm_mode").write_text("smoke")
            args = argparse.Namespace(check=False, preflight=False, smoke=False,
                                      ollama_url="http://remote:11434", output_dir=str(output),
                                      repeats=3, fresh=False)
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "run_command", return_value=0), \
                 patch.object(runner, "preflight", return_value={}), \
                 patch.dict(runner.os.environ):
                with self.assertRaisesRegex(ValueError, "different mode"):
                    runner.execute(args, root / "log", "test")

    def test_dry_run_has_no_side_effects_or_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner.sys, "argv", ["run_airnrm_np.py", "--dry-run"]):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_failed_and_incomplete_raw_outputs_are_rejected(self):
        repo = Path(__file__).resolve().parents[1]
        import sys
        sys.path.insert(0, str(repo))
        from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
            AirNrmLargeScaleData, EXAMPLE_MODEL_CODE, format_parameters,
        )
        with patch.object(runner, "ROOT", repo):
            query = runner.query_rows()[0]["Query"]
            data = AirNrmLargeScaleData.from_path(repo / runner.DATA)
            code = format_parameters(data.assemble(query)) + EXAMPLE_MODEL_CODE.rstrip()
            answer = ("### Objective Function\nRevenue\n### Constraints\n"
                      "Decision variable domains, capacity, balance, scale, selection, cardinality and flow.\n"
                      "### Generated Code\n```python\n" + code + "\n```")
            record = {"instance_id": 0, "status": "ok", "pred_type": "Policy_Flow",
                      "model_output": answer, "code_output": code}
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                records_file = path / "instances.jsonl"
                records_file.write_text(json.dumps(record) + "\n")
                self.assertTrue(runner.validate_run(path, [0]))
                record["code_output"] = "# injected replacement"
                records_file.write_text(json.dumps(record) + "\n")
                self.assertFalse(runner.validate_run(path, [0]))
                record["status"] = "failed"
                record["errors"] = ["a provider error"]
                records_file.write_text(json.dumps(record) + "\n")
                self.assertFalse(runner.validate_run(path, [0]))


if __name__ == "__main__":
    unittest.main()
