"""Regression cases from the failed CA output protocol; no real model calls."""
import ast
import csv
import re
import unittest
from pathlib import Path

from ca_lean_workflow import (AirNrmCAData, parse_ca_query, PARAMETER_ORDER,
                             example_answer, validate_complete_generated_answer)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'Test_Dataset/Air_NRM/large_scale'


class CAWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = AirNrmCAData.from_path(DATA)
        with (DATA / 'query_CA.csv').open() as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.parameters = cls.data.assemble(cls.rows[0]['Query'])
        cls.good = example_answer(cls.parameters)

    def test_all_fifteen_inputs_and_literal_outputs(self):
        for row in self.rows:
            p = self.data.assemble(row['Query'])
            parsed = parse_ca_query(row['Query'])
            self.assertEqual(len(p['avg_price']), 4 * len(parsed.flights))
            self.assertEqual(4 * len(parsed.flights) + len(p['avg_pax']), int(row['n_variables']))
            self.assertNotIn('option_num', p)
            self.assertEqual(set(p), set(PARAMETER_ORDER))
            code = validate_complete_generated_answer(example_answer(p), p)
            ast.parse(code)

    def test_price_only_answer_is_rejected(self):
        answer = re.sub(r'^value_0_list=.*\n', '', self.good, flags=re.M)
        with self.assertRaisesRegex(ValueError, 'missing parameters'):
            validate_complete_generated_answer(answer, self.parameters)

    def test_assignments_squeezed_onto_one_line_are_rejected(self):
        answer = self.good.replace('\nvalue_list=', ' value_list=', 1)
        with self.assertRaises(SyntaxError):
            validate_complete_generated_answer(answer, self.parameters)

    def test_observation_label_in_program_is_rejected(self):
        answer = self.good.replace('avg_price=', 'Observation: avg_price=', 1)
        with self.assertRaisesRegex(ValueError, 'missing parameters'):
            validate_complete_generated_answer(answer, self.parameters)

    def test_agent_stop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'agent stopped'):
            validate_complete_generated_answer('Agent stopped due to iteration limit or time limit.', self.parameters)

    def test_changed_values_and_ellipsis_are_rejected(self):
        answer = self.good.replace('capacity_consum={0: 2.0', 'capacity_consum={0: 9.0', 1)
        self.assertNotEqual(answer, self.good)
        with self.assertRaisesRegex(ValueError, 'changed parameters'):
            validate_complete_generated_answer(answer, self.parameters)
        with self.assertRaisesRegex(ValueError, 'ellipsis'):
            validate_complete_generated_answer(self.good.replace('model.optimize()', '...\nmodel.optimize()'), self.parameters)

    def test_code_is_extracted_verbatim(self):
        code = validate_complete_generated_answer(self.good, self.parameters)
        self.assertEqual(code, self.good.split('```python\n', 1)[1].split('\n```', 1)[0].strip())

    def test_np_dataset_is_rejected(self):
        with (DATA / 'query_NP_Flow.csv').open() as handle:
            np_row = next(csv.DictReader(handle))
        with self.assertRaisesRegex(ValueError, 'NP/Flow'):
            self.data.assemble(np_row['Query'])


if __name__ == '__main__':
    unittest.main()
