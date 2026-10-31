#!/usr/bin/env python3
"""Offline regression tests for Task 4; these tests never construct an LLM."""

from __future__ import annotations

import ast
import csv
import contextlib
import io
import unittest
from pathlib import Path

from np_lean_workflow import (
    AirNrmLargeScaleData,
    EXAMPLE_MODEL_CODE,
    PRODUCTS,
    format_parameters,
    parse_np_query,
    validate_complete_generated_answer,
)


HERE = Path(__file__).resolve().parent


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class LargeScaleNPLeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = AirNrmLargeScaleData.from_path(HERE)
        cls.queries = rows(HERE / "query_NP_Flow.csv")

    def test_all_queries_preserve_model_fields(self):
        self.assertEqual(len(self.queries), 15)
        for row in self.queries:
            parsed = parse_np_query(row["Query"])
            parameters = self.data.assemble(row["Query"])
            self.assertEqual(parameters["products"], list(PRODUCTS))
            self.assertEqual(parameters["option_num"], parsed.z)
            self.assertEqual(len(parameters["flight_info"]), len(parsed.flights))
            self.assertEqual(len(parameters["avg_price"]), 4 * len(parsed.flights))
            self.assertEqual(len(parameters["flight_capacity"]), len(parsed.flights))
            self.assertEqual(
                int(row["n_variables"]),
                5 * len(parsed.flights) + len({flight[:2] for flight in parsed.flights}),
            )

    def test_capacity_and_tight_link_inputs(self):
        for row in self.queries:
            parsed = parse_np_query(row["Query"])
            parameters = self.data.assemble(row["Query"])
            self.assertEqual(parsed.capacity_fraction, 0.28)
            self.assertEqual(set(parameters["capacity_consum"]), {0, 1, 2, 3})
            for fid, capacity in parameters["flight_capacity"].items():
                self.assertGreater(capacity, 0)
                for pid, consumption in parameters["capacity_consum"].items():
                    self.assertGreater(capacity / consumption, 0)

    def test_airports_are_dynamic_and_flow_is_possible(self):
        for row in self.queries:
            parsed = parse_np_query(row["Query"])
            airports = {code for flight in parsed.flights for code in flight[:2]}
            self.assertTrue(all(len(code) > 1 for code in airports))
            ods = {flight[:2] for flight in parsed.flights}
            self.assertTrue(all((destination, origin) in ods for origin, destination in ods))

    def test_gurobi_template_matches_all_15_labels(self):
        """Execute the exact compact-key code template, not a second model."""
        quiet_template = EXAMPLE_MODEL_CODE.replace(
            'model = gp.Model("air_nrm_np_flow")\n',
            'model = gp.Model("air_nrm_np_flow")\nmodel.Params.OutputFlag = 0\n',
            1,
        )
        for number, row in enumerate(self.queries, 1):
            parameters = self.data.assemble(row["Query"])
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(format_parameters(parameters) + quiet_template, namespace)  # noqa: S102
            model = namespace["model"]
            self.assertEqual(model.Status, 2, f"query {number} was not optimal")
            expected = float(row["Label-objective"])
            self.assertLessEqual(
                abs(model.ObjVal - expected),
                1e-5 * max(1.0, abs(expected)),
                f"query {number} objective differs from the Task 3 label",
            )

    def test_raw_answer_must_contain_exact_complete_parameters(self):
        parameters = self.data.assemble(self.queries[0]["Query"])
        model = """### Objective Function
The objective maximizes ticket-sales revenue over the decision variables.

### Constraints
Decision variable domains and the capacity, OD balance, SBLP scale,
sales/selection linking, at-most-Z cardinality, and airport flow conservation
constraints are enforced.

### Generated Code
"""
        answer = model + "```python\n" + format_parameters(parameters) + EXAMPLE_MODEL_CODE + "\n```"
        extracted = validate_complete_generated_answer(answer, parameters)
        self.assertEqual(extracted, format_parameters(parameters) + EXAMPLE_MODEL_CODE.rstrip())

        with self.assertRaisesRegex(ValueError, "ellipsis placeholder"):
            validate_complete_generated_answer(
                answer.replace("option_num=10", "option_num=..."),
                parameters,
            )
        with self.assertRaisesRegex(ValueError, "changed parameters"):
            validate_complete_generated_answer(
                answer.replace("option_num=10", "option_num=999"),
                parameters,
            )
        with self.assertRaisesRegex(ValueError, "missing parameters"):
            validate_complete_generated_answer(
                answer.replace("option_num=10\n", ""),
                parameters,
            )
        with self.assertRaisesRegex(ValueError, "exactly one tagged Python"):
            validate_complete_generated_answer(
                answer + "\n" + answer,
                parameters,
            )
        answer_with_math_fence = answer.replace(
            "### Generated Code",
            "```\nsum x <= capacity\n```\n\n### Generated Code",
        )
        self.assertEqual(
            validate_complete_generated_answer(answer_with_math_fence, parameters),
            extracted,
        )
        with self.assertRaisesRegex(ValueError, "must contain Objective Function"):
            validate_complete_generated_answer(
                answer.replace("### Objective Function", "### Model"),
                parameters,
            )
        with self.assertRaisesRegex(ValueError, "do not duplicate parameters"):
            validate_complete_generated_answer(
                answer.replace("### Generated Code", "### Retrieved Information\n"
                               "avg_price={...}\n\n### Generated Code"),
                parameters,
            )

    def test_exemplar_is_label_free_and_disjoint(self):
        with (HERE / "RAG_Example_SBLP_Flow_LargeScale_NP.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            exemplar_rows = list(reader)
            self.assertEqual(reader.fieldnames, ["prompt"])
        self.assertEqual(len(exemplar_rows), 1)
        exemplar = parse_np_query(exemplar_rows[0]["prompt"])
        used = {flight for row in self.queries for flight in parse_np_query(row["Query"]).flights}
        self.assertFalse(set(exemplar.flights) & used)
        self.data.assemble(exemplar_rows[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
