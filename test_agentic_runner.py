"""Offline regression tests for the Air-NRM agentic benchmark runner."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from google.genai import types


HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(
    HERE,
    "Test_Dataset",
    "Large-scale-or",
    "run_test_universe_agentic.py",
)
SPEC = importlib.util.spec_from_file_location("agentic_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

REPORT_PATH = os.path.join(
    HERE,
    "Test_Dataset",
    "Air_NRM",
    "large_scale",
    "report_agentic_benchmark.py",
)
REPORT_SPEC = importlib.util.spec_from_file_location("agentic_report", REPORT_PATH)
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(REPORT)

BENCHMARK_PATH = os.path.join(
    HERE,
    "Test_Dataset",
    "Air_NRM",
    "large_scale",
    "run_agentic_benchmark.py",
)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "logged_agentic_benchmark", BENCHMARK_PATH
)
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)

LSOR_BENCHMARK_PATH = os.path.join(
    HERE,
    "Test_Dataset",
    "Large-scale-or",
    "run_singlecall_cost_benchmark.py",
)
LSOR_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "lsor_singlecall_benchmark", LSOR_BENCHMARK_PATH
)
LSOR_BENCHMARK = importlib.util.module_from_spec(LSOR_BENCHMARK_SPEC)
LSOR_BENCHMARK_SPEC.loader.exec_module(LSOR_BENCHMARK)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.valid = "\n".join([
            "import gurobipy as gp",
            "model = gp.Model()",
            "model.optimize()",
        ])

    def assert_valid_extraction(self, response):
        _, code, error = RUNNER.extract_code_from_response(response)
        self.assertEqual(error, "")
        self.assertEqual(RUNNER.validate_gurobi_code(code), "")

    def test_supported_fence_variants_and_fallbacks(self):
        responses = [
            f"### Python Code\n```python\n{self.valid}\n```",
            f"### Python Code\n```PYTHON\n{self.valid}\n```",
            f"### Python Code\n```py\n{self.valid}\n```",
            f"### Python Code\n```\n{self.valid}\n```",
            f"### Python Code\n```python\n{self.valid}",
            f"### Python Code\n{self.valid}",
            self.valid,
            (
                "```python\nprint(1)\n```\n"
                f"```Python\n{self.valid}\n```"
            ),
        ]
        for response in responses:
            with self.subTest(response=response[:40]):
                self.assert_valid_extraction(response)

    def test_math_model_and_failure_taxonomy(self):
        response = (
            "### Mathematical Model\nminimize x\n"
            f"### Python Code\n```python\n{self.valid}\n```"
        )
        math_model, _, error = RUNNER.extract_code_from_response(response)
        self.assertEqual(math_model, "minimize x")
        self.assertEqual(error, "")

        failures = [
            ("", "empty_response"),
            ("Only prose", "no_code_block"),
            ("```python\nprint(1)\n```", "no_gurobi_code:"),
            (
                "```python\nimport gurobipy as gp\n"
                "model = gp.Model(\nmodel.optimize()\n```",
                "python_syntax_error:",
            ),
        ]
        for response, expected in failures:
            with self.subTest(expected=expected):
                _, _, extraction_error = RUNNER.extract_code_from_response(response)
                self.assertTrue(extraction_error.startswith(expected))

    def test_prompt_contains_shared_contract(self):
        prompt = RUNNER.build_prompt("test query")
        self.assertIn("### Mathematical Model", prompt)
        self.assertIn("### Python Code", prompt)
        self.assertIn("Exactly one fenced Python code block", prompt)

    def test_path_mapping_ignores_quoted_csv_diagnostic_prose(self):
        code = (
            "import gurobipy as gp\n"
            "print(f'OD {od} not found in v1.csv or v2.csv')\n"
            "df = pd.read_csv('flight_capacity.csv')\n"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            mapped = RUNNER.replace_file_paths_in_code(
                code, RUNNER.AIR_NRM_DATA_FILES
            )
        self.assertEqual(output.getvalue(), "")
        self.assertIn(RUNNER.AIR_NRM_DATA_FILES[-1], mapped)
        self.assertIn("OD {od} not found in v1.csv or v2.csv", mapped)


class TwoStageWorkflowTests(unittest.TestCase):
    def test_query_is_preserved_and_gold_is_never_added_to_prompts(self):
        query = "A UNIQUE QUERY STRING"
        formulation = "minimize c^T x subject to Ax <= b"
        execution = {
            "ok": False,
            "error": "NameError: x is not defined",
            "stdout": "",
            "stderr": "traceback",
        }
        prompts = [
            BENCHMARK.build_formulation_prompt(query),
            BENCHMARK.build_code_prompt(query, formulation),
            BENCHMARK.build_repair_prompt(
                query, formulation, "import gurobipy", execution
            ),
        ]
        for prompt in prompts:
            self.assertIn(query, prompt)
            self.assertNotIn("Label-objective", prompt)
            self.assertNotIn("gold_objective", prompt)
        self.assertIn(formulation, prompts[1])
        self.assertIn(formulation, prompts[2])
        self.assertIn(execution["error"], prompts[2])

    def test_local_validation_failure_is_one_failed_tool_attempt(self):
        rec = SimpleNamespace(calls=[])
        rec.t_start = 0.0

        def add_call(call):
            rec.calls.append(call)

        rec.add_call = add_call
        result, ok = BENCHMARK.execute_local_attempt(
            rec, "initial_local_execution", "", "no_code_block", 1.0
        )
        self.assertFalse(ok)
        self.assertFalse(result["ok"])
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["type"], "tool")
        self.assertEqual(rec.calls[0]["tool_name"], "local_gurobipy_execution")


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        code = "\n".join([
            "import gurobipy as gp",
            "model = gp.Model()",
            "model.optimize()",
        ])
        self.response_text = f"### Python Code\n```python\n{code}\n```"

    def test_gemini_request_is_single_generation_without_tools(self):
        class Models:
            def __init__(self, response_text):
                self.response_text = response_text
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    text=self.response_text,
                    candidates=[SimpleNamespace(
                        finish_reason=types.FinishReason.STOP,
                        content=SimpleNamespace(parts=[
                            SimpleNamespace(
                                text=self.response_text, thought=False,
                                executable_code=None,
                                code_execution_result=None,
                            ),
                        ]),
                    )],
                    usage_metadata=None,
                    model_version="gemini-3.1-pro-preview",
                    response_id="offline-response",
                    prompt_feedback=None,
                )

        backend = SimpleNamespace(
            models=Models(self.response_text),
        )
        config = {
            "max_retries": 1,
            "gemini_api_key": "offline-test",
            "model_name": "gemini-3.1-pro-preview",
        }
        with patch("google.genai.Client", return_value=backend) as client_ctor:
            client = RUNNER.GeminiClient(config)
            metadata = client.generate_with_metadata(
                "prompt", RUNNER.AIR_NRM_DATA_FILES
            )

        self.assertEqual(metadata["text"], self.response_text)
        self.assertEqual(metadata["finish_reasons"], ["STOP"])
        self.assertEqual(metadata["tool_calls"], 0)
        self.assertEqual(
            metadata["provider_response_parts"][0]["content"]["parts"][0]
            ["text"],
            self.response_text,
        )
        call = backend.models.calls[0]
        self.assertIn("### Python Code", str(call["config"].system_instruction))
        self.assertIsNone(call["config"].tools)
        self.assertIsNone(call["config"].thinking_config)
        self.assertIsNone(call["config"].temperature)
        self.assertIsNone(call["config"].top_p)
        retry_options = client_ctor.call_args.kwargs["http_options"].retry_options
        self.assertEqual(retry_options.attempts, 1)
        request_text = call["contents"]
        for path in RUNNER.AIR_NRM_DATA_FILES:
            self.assertIn(f"### DATA FILE: {os.path.basename(path)}", request_text)
            contents = Path(path).read_text(encoding="utf-8-sig")
            self.assertIn(f"```csv\n{contents}\n```", request_text)
        self.assertIn("### TASK\nprompt", request_text)

    def test_openai_request_uses_same_contract(self):
        class Responses:
            def __init__(self, response_text):
                self.response_text = response_text
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    status="completed",
                    output_text=self.response_text,
                    error=None,
                )

        backend = SimpleNamespace(
            responses=Responses(self.response_text),
        )
        config = {
            "max_retries": 1,
            "openai_api_key": "offline-test",
            "model_name": "gpt-5.4",
        }
        with patch("openai.OpenAI", return_value=backend) as client_ctor:
            client = RUNNER.OpenAIClient(config)
            response = client.generate("prompt", RUNNER.AIR_NRM_DATA_FILES)

        self.assertEqual(response, self.response_text)
        self.assertEqual(client_ctor.call_args.kwargs["max_retries"], 0)
        call = backend.responses.calls[0]
        self.assertEqual(call["instructions"], RUNNER.SYSTEM_INSTRUCTIONS)
        self.assertNotIn("tools", call)
        content = call["input"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "input_text")
        request_text = content[0]["text"]
        for path in RUNNER.AIR_NRM_DATA_FILES:
            self.assertIn(f"### DATA FILE: {os.path.basename(path)}", request_text)
            contents = Path(path).read_text(encoding="utf-8-sig")
            self.assertIn(f"```csv\n{contents}\n```", request_text)
        self.assertIn("### TASK\nprompt", request_text)

    def test_full_flight_csv_is_not_truncated_at_1000_rows(self):
        flight = Path(RUNNER.AIR_NRM_DATA_FILES[0])
        lines = flight.read_text(encoding="utf-8-sig").splitlines()
        self.assertGreater(len(lines) - 1, 1000)
        request_text = RUNNER.prompt_with_file_contents(
            "prompt", RUNNER.AIR_NRM_DATA_FILES
        )
        self.assertIn(lines[-1], request_text)


class ReportTests(unittest.TestCase):
    def test_report_contains_scores_calls_tokens_and_cost(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "synthetic-run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({
                "run_id": "synthetic-run",
                "dataset": "Air-NRM-CA-LS",
                "method": "Base-Agentic",
                "model_profile": "gpt-5.4",
                "models": {"modeler": {"model": "gpt-5.4"}},
            }))
            record = {
                "run_id": "synthetic-run",
                "instance_id": 0,
                "status": "ok",
                "summary": {
                    "n_llm_calls": 1,
                    "n_tool_calls": 2,
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 20,
                    "completion_tokens": 30,
                    "reasoning_tokens": 10,
                    "total_tokens": 130,
                    "cost_usd": 0.0307,
                    "llm_latency_s": 2.0,
                    "wall_s": 2.1,
                },
            }
            (run_dir / "instances.jsonl").write_text(json.dumps(record) + "\n")
            scored = pd.DataFrame([{
                "dataset": "Air-NRM-CA-LS", "instance_id": 0,
                "run_id": "synthetic-run", "method": "Base-Agentic",
                "model_profile": "gpt-5.4", "gold_type": None,
                "gold_objective": 100.0, "objective": 100.005,
                "correct_optimal": 0, "correct_formulation": "",
                "status": "ok", "solver_status": 2, "n_vars": 1,
                "n_constrs": 0, "n_nonzeros": 0, "is_mip": False,
                "solve_runtime_s": 0.01, "solve_error": None,
                "workdir_used": "repo_root", "size_class": 1,
            }])
            with patch.object(REPORT, "score_run", return_value=scored), \
                    patch.object(REPORT, "run_existing_aggregate"), \
                    patch.object(REPORT, "run_parameter_diagnosis"):
                report = REPORT.generate_report(run_dir, diagnose=False)

            self.assertEqual(
                report["scores"]["relative_tolerance_1e-4"]["correct"], 1
            )
            self.assertEqual(
                report["scores"]["relative_tolerance_1e-6"]["correct"], 0
            )
            self.assertEqual(report["resources"]["n_tool_calls"], 2)
            self.assertTrue((run_dir / "report.md").exists())
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "per_instance_report.csv").exists())


class KeyPromptTests(unittest.TestCase):
    def test_key_is_hidden_and_only_last_four_are_printed(self):
        key = "sk-test-1234567890abcdefghijkl"
        spec = BENCHMARK.lx.ModelSpec(
            provider="openai", model="gpt-5.4", api_key_env="TEST_AGENTIC_KEY"
        )
        output = io.StringIO()
        try:
            with patch.object(BENCHMARK.getpass, "getpass", return_value=key), \
                    redirect_stdout(output):
                BENCHMARK.prompt_for_api_key(spec)
            self.assertEqual(os.environ["TEST_AGENTIC_KEY"], key)
            self.assertNotIn(key, output.getvalue())
            self.assertIn("****ijkl", output.getvalue())
        finally:
            os.environ.pop("TEST_AGENTIC_KEY", None)


class CostLimitTests(unittest.TestCase):
    def test_medium_preflight_supplies_capacity_to_model_and_records_protocol(self):
        seen = []
        original_load = BENCHMARK.lx.load_config

        def capture_config(*args, **kwargs):
            cfg = original_load(*args, **kwargs)
            seen.append(cfg)
            return cfg

        with patch.object(BENCHMARK, "DATA_FILES", list(BENCHMARK.DATA_FILES)), \
                patch.object(BENCHMARK, "CODE_SYSTEM_INSTRUCTIONS", BENCHMARK.CODE_SYSTEM_INSTRUCTIONS), \
                patch.object(BENCHMARK.lx, "load_config", side_effect=capture_config), \
                patch.object(BENCHMARK.sys, "argv", ["run_agentic_benchmark.py", "gpt-5.2",
                    "--scale", "medium", "--dataset", "np", "--dry-run"]), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(BENCHMARK.main(), 0)
            payload = BENCHMARK.CORE.prompt_with_file_contents(
                "test query", [str(p) for p in BENCHMARK.DATA_FILES])
            self.assertIn("DATA FILE: flight_capacity.csv", payload)
            self.assertIn(",187", payload)
            self.assertNotIn("Label-objective", payload)
        self.assertEqual(len(seen), 1)
        cfg = seen[0]
        self.assertEqual(len(cfg.manifest["inline_data_files"]), 5)
        self.assertEqual(cfg.manifest["benchmark_adapter"], "medium_resources_v2_capacity_input")
        self.assertIn("medium_resources_v2_capacity_input", cfg.notes)
        self.assertTrue(cfg.run_id.endswith(cfg.config_hash()))

    def test_medium_limits_cover_ca_and_np_column_layouts(self):
        args = SimpleNamespace(max_instance_cost_usd=None, max_run_cost_usd=30.0)
        for family, expected in (("CA", 15), ("NP_Flow", 21)):
            with self.subTest(family=family):
                data = pd.read_csv(Path(HERE) / "Test_Dataset/Air_NRM/medium_scale"
                                   / f"query_{family}.csv")
                self.assertEqual(len(data), expected)
                limits = [BENCHMARK.instance_cost_limit(row, data, args)
                          for _, row in data.iterrows()]
                self.assertAlmostEqual(sum(limits), 30.0)
                self.assertTrue(all(value > 0 for value in limits))
                if family == "CA":
                    self.assertTrue(all(value == 2.0 for value in limits))
                else:
                    self.assertGreater(max(limits), min(limits))

    def test_dynamic_limits_sum_to_run_budget_and_increase_with_size(self):
        data = pd.read_csv(
            Path(HERE) / "Test_Dataset/Air_NRM/large_scale/query_CA.csv"
        )
        args = SimpleNamespace(
            max_instance_cost_usd=None, max_run_cost_usd=30.0
        )
        limits = [
            BENCHMARK.instance_cost_limit(row, data, args)
            for _, row in data.iterrows()
        ]
        self.assertAlmostEqual(sum(limits), 30.0)
        self.assertTrue(all(a < b for a, b in zip(limits, limits[1:])))
        self.assertAlmostEqual(limits[0], 1.155226, places=5)
        self.assertAlmostEqual(limits[-1], 2.962949, places=5)


class LargeScaleORSingleCallTests(unittest.TestCase):
    def test_all_101_rows_and_data_files_validate(self):
        data = pd.read_csv(LSOR_BENCHMARK.QUERY_CSV)
        resolved = LSOR_BENCHMARK.validate_dataset(data)
        self.assertEqual(len(data), 101)
        self.assertEqual(len(resolved), 101)
        self.assertTrue(all(files for files in resolved))
        self.assertTrue(all(path.is_file() for files in resolved for path in files))

    def test_profiles_resolve_to_requested_models(self):
        for alias, expected in {
            "openai": "gpt-5.2",
            "gpt-5.2": "gpt-5.2",
            "gemini": "gemini-3.1-pro",
            "gemini-3.1-pro": "gemini-3.1-pro",
            "gemini-3-pro": "gemini-3.1-pro",
        }.items():
            self.assertEqual(LSOR_BENCHMARK.PROFILES[alias], expected)

    def test_gemini_fixed_instance_cost_default(self):
        self.assertEqual(
            LSOR_BENCHMARK.GEMINI_DEFAULT_INSTANCE_COST_USD,
            0.40,
        )

    def test_nonrecoverable_provider_errors_fail_fast(self):
        fatal = RuntimeError(
            "404 NOT_FOUND: model is not found for API version v1beta"
        )
        transient = RuntimeError("connection reset by peer")
        self.assertTrue(LSOR_BENCHMARK.is_fatal_provider_error(fatal))
        self.assertFalse(LSOR_BENCHMARK.is_fatal_provider_error(transient))

    def test_token_weighted_limits_sum_to_run_budget(self):
        data = pd.read_csv(LSOR_BENCHMARK.QUERY_CSV)
        args = SimpleNamespace(
            max_instance_cost_usd=None, max_run_cost_usd=50.0
        )
        limits = [
            LSOR_BENCHMARK.instance_cost_limit(row, data, args)
            for _, row in data.iterrows()
        ]
        self.assertAlmostEqual(sum(limits), 50.0)
        self.assertGreater(max(limits), min(limits))

    def test_one_row_smoke_test_receives_the_whole_run_budget(self):
        data = pd.read_csv(LSOR_BENCHMARK.QUERY_CSV).iloc[[0]]
        args = SimpleNamespace(
            max_instance_cost_usd=None, max_run_cost_usd=1.0
        )
        limit = LSOR_BENCHMARK.instance_cost_limit(data.iloc[0], data, args)
        self.assertAlmostEqual(limit, 1.0)


if __name__ == "__main__":
    unittest.main()
