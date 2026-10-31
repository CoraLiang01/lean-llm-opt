"""Offline checks for Gemini resource estimates and this repository's log format."""
import json
import tempfile
import unittest
from pathlib import Path

from estimate_gemini_3pro import Pricing, Usage, estimate_call, estimate_run, estimate_totals, main


class CostAccountingTests(unittest.TestCase):
    def test_cache_is_subset_and_thinking_is_billable_output(self):
        usage = Usage(100_000, 1_000, 2_000, 50_000)
        baseline = estimate_call(usage)
        self.assertEqual(baseline["total_tokens"], 103_000)
        self.assertAlmostEqual(baseline["token_cost_usd"], 0.146)
        more_thinking = estimate_call(usage, thinking_ratio=1.25)
        self.assertEqual(more_thinking["total_tokens"], 103_500)
        self.assertAlmostEqual(more_thinking["token_cost_usd"], 0.152)

    def test_removing_cache_changes_cost_but_not_total_tokens(self):
        usage = Usage(100_000, 1_000, 2_000, 50_000)
        no_cache = estimate_call(usage, preserve_cache=False)
        self.assertEqual(no_cache["total_tokens"], 103_000)
        self.assertEqual(no_cache["cached_prompt_tokens"], 0)
        self.assertAlmostEqual(no_cache["token_cost_usd"], 0.236)

    def test_exact_200k_boundary_and_full_prompt_including_cache(self):
        pricing = Pricing()
        self.assertAlmostEqual(pricing.cost(Usage(200_000, 0)), 0.4)
        self.assertAlmostEqual(pricing.cost(Usage(200_001, 0)), 0.800004)
        # Even though the uncached part is small, the full prompt exceeds 200k.
        self.assertAlmostEqual(pricing.cost(Usage(200_001, 0, 0, 200_000)), 0.080004)

    def test_totals_only_does_not_invent_a_token_cost_conversion(self):
        result = estimate_totals(1_000, 0.1, token_ratio=1.2, cost_ratio=1.5)
        self.assertEqual(result["estimated_total_tokens"], 1_200)
        self.assertAlmostEqual(result["estimated_api_cost_usd"], 0.15)
        self.assertEqual(estimate_totals(810_416, 4.896942)["estimated_api_cost_usd"], 4.896942)

    def test_rejects_invalid_or_unknown_usage(self):
        for args in [(1, 0, 0, 2), (-1, 0), (float("nan"), 0), (True, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Usage(*args)
        with self.assertRaisesRegex(ValueError, "missing reasoning_tokens"):
            Usage.from_call({"prompt_tokens": 10, "completion_tokens": 5})
        with self.assertRaisesRegex(ValueError, "total_tokens must equal"):
            Usage.from_call({"prompt_tokens": 10, "completion_tokens": 5,
                             "reasoning_tokens": 2, "total_tokens": 15})
        for ratio in [-1, float("inf"), float("nan")]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                estimate_call(Usage(1, 1), ratio)


class LogIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "source_run"
        self.directory.mkdir()
        call = dict(type="llm", provider="google", model="gemini-3.1-pro-preview",
                    prompt_tokens=150_000, completion_tokens=1_000, reasoning_tokens=0,
                    cached_prompt_tokens=0, total_tokens=151_000, cost_usd=0.312,
                    token_source="provider", ok=True)
        self.calls = [call, {"type": "tool", "cost_usd": 0}, {**call, "ok": False}]
        self.summary = dict(run_id="source_run", dataset="test", method="SingleCall",
                            n_instances=2, total_n_llm_calls=2, total_prompt_tokens=300_000,
                            total_completion_tokens=2_000, total_reasoning_tokens=0,
                            total_cached_prompt_tokens=0, total_total_tokens=302_000,
                            total_cost_usd=0.624)
        self.save()

    def save(self):
        (self.directory / "calls.jsonl").write_text(
            "\n".join(json.dumps(c) for c in self.calls) + "\n")
        (self.directory / "summary.json").write_text(json.dumps(self.summary))

    def test_per_request_tier_tool_exclusion_and_failed_call_usage(self):
        before = (self.directory / "calls.jsonl").read_bytes()
        result = estimate_run(self.directory, thinking_ratios=[1])[0]
        self.assertEqual(result["source_n_llm_calls"], 2)
        self.assertEqual(result["source_failed_calls"], 1)
        self.assertEqual(result["source_long_context_calls"], 0)
        self.assertEqual(result["estimated_total_tokens"], 302_000)
        self.assertAlmostEqual(result["estimated_api_cost_usd"], 0.624)
        self.assertEqual((self.directory / "calls.jsonl").read_bytes(), before)

    def test_rejects_summary_mismatch(self):
        self.summary["total_total_tokens"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "calls disagree with summary"):
            estimate_run(self.directory)

    def test_rejects_wrong_model(self):
        self.calls[0]["model"] = "gemini-3.5-flash"
        self.save()
        with self.assertRaisesRegex(ValueError, "expected Gemini 3.1 Pro"):
            estimate_run(self.directory)

    def test_cli_exports_usage_without_raw_responses(self):
        self.calls[0]["response"] = "do not export model answers"
        self.save()
        out = Path(self.temp.name) / "output"
        self.assertEqual(main(["--run-dir", str(self.directory), "--thinking-ratios", "1",
                               "--output-dir", str(out)]), 0)
        text = (out / "estimates.json").read_text()
        self.assertNotIn("do not export model answers", text)
        self.assertEqual(len(json.loads(text)["estimates"]), 1)
        self.assertTrue((out / "estimates.csv").is_file())


if __name__ == "__main__":
    unittest.main()
