"""Real Gurobi interface regression tests; no model API or benchmark CSV access."""
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import unittest
from pathlib import Path
import gurobipy as gp

ROOT = Path(__file__).resolve().parents[1]
COPIES = sorted(ROOT.glob('*Large-scale_Model_Interface_V2.ipynb'))


def executor_namespace(path):
    book = json.loads(path.read_text())
    tree = ast.parse(''.join(book['cells'][29]['source']))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in {
            'invoke_formulation', 'case_fields', 'execute_pipeline_case'}:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
            'MODEL_INTERFACE_VERSION', 'MODEL_RETURN_CONTRACT'} for t in node.targets):
            keep.append(node)
    ns = dict(ast=ast, gp=gp, hashlib=hashlib, json=json, math=math, re=re)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class ModelInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.ns = executor_namespace(COPIES[0])

    def run_code(self, source):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns['execute_code'](source, return_trace=True)

    def test_all_three_copies_share_changes_and_compile(self):
        self.assertEqual(len(COPIES), 3)
        books = [json.loads(p.read_text()) for p in COPIES]
        for index in (25, 27, 29, 33):
            self.assertEqual(len({''.join(b['cells'][index]['source']) for b in books}), 1)
        for path, book in zip(COPIES, books):
            for i, cell in enumerate(book['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    compile(source, f'{path}:{i}', 'exec')
                    self.assertFalse(re.search(r'(?m)^RUN_\w+ = True$', source))

    def test_global_model_is_unchanged(self):
        obj, _, trace = self.run_code('import gurobipy as gp\nm=gp.Model()\nm.Params.OutputFlag=0\nx=m.addVar(ub=3)\nm.setObjective(x,gp.GRB.MAXIMIZE)\nm.optimize()')
        self.assertEqual(obj, 3)
        self.assertEqual(trace['model_return_strategy'], 'global_m')
        self.assertFalse(trace['model_contract_recovered'])
        self.assertEqual(trace['model_count'], 1)

    def test_missing_return_is_recovered_without_math_change(self):
        source = 'from gurobipy import Model, GRB\ndef solve():\n    local=Model()\n    local.Params.OutputFlag=0\n    x=local.addVar(ub=7)\n    local.setObjective(x,GRB.MAXIMIZE)\n    local.optimize()\nm=solve()'
        obj, _, trace = self.run_code(source)
        self.assertEqual(obj, 7)
        self.assertEqual(trace['model_return_strategy'], 'captured_constructor')
        self.assertTrue(trace['model_contract_recovered'])
        self.assertEqual(trace['executed_source'].count('.optimize()'), 1)
        self.assertNotIn('return local', trace['executed_source'])

    def test_model_alias_and_future_import(self):
        obj, _, trace = self.run_code('"module doc"\nfrom __future__ import annotations\nfrom gurobipy import Model as Make\ndef solve():\n    local=Make()\n    local.Params.OutputFlag=0\n    local.optimize()\nsolve()')
        self.assertEqual(obj, 0)
        self.assertTrue(trace['model_contract_recovered'])

    def test_unique_other_global_name(self):
        _, _, trace = self.run_code('import gurobipy as gp\nsolver=gp.Model()\nsolver.Params.OutputFlag=0\nsolver.optimize()\nm=None')
        self.assertEqual(trace['model_return_strategy'], 'unique_global_model')

    def test_uncalled_function_is_not_invented(self):
        with self.assertRaises(self.ns['ModelContractError']) as caught:
            self.run_code('import gurobipy as gp\ndef solve():\n    return gp.Model()')
        self.assertEqual(caught.exception.execution_trace['model_count'], 0)

    def test_ambiguous_models_fail_even_with_named_m(self):
        with self.assertRaises(self.ns['ModelContractError']) as caught:
            self.run_code('import gurobipy as gp\nm=gp.Model()\nother=gp.Model()')
        self.assertEqual(caught.exception.execution_trace['model_count'], 2)

    def test_unoptimized_model_is_not_optimized_by_executor(self):
        with self.assertRaises(self.ns['SolverStatusError']) as caught:
            self.run_code('import gurobipy as gp\nm=gp.Model()')
        self.assertEqual(caught.exception.execution_trace['solver_status'], gp.GRB.LOADED)

    def test_infeasibility_is_not_rescued(self):
        with self.assertRaises(self.ns['SolverStatusError']) as caught:
            self.run_code('import gurobipy as gp\nm=gp.Model()\nm.Params.OutputFlag=0\nx=m.addVar(lb=0)\nm.addConstr(x<=-1)\nm.optimize()')
        self.assertEqual(caught.exception.execution_trace['solver_status'], gp.GRB.INFEASIBLE)

    def test_disposed_model_fails_clearly(self):
        with self.assertRaises(self.ns['ModelContractError']):
            self.run_code('import gurobipy as gp\nm=gp.Model()\nm.dispose()')

    def test_original_runtime_error_is_preserved(self):
        with self.assertRaises(ZeroDivisionError) as caught:
            self.run_code('import gurobipy as gp\nm=gp.Model()\n1/0')
        self.assertFalse(caught.exception.execution_trace['generated_program_completed'])
        self.assertFalse(caught.exception.execution_trace['execution_ok'])
        self.assertEqual(caught.exception.execution_trace['model_count'], 1)

    def test_trace_is_recorded_as_json_and_source(self):
        _, _, trace = self.run_code('import gurobipy as gp\nm=gp.Model()\nm.Params.OutputFlag=0\nm.optimize()')
        record = {}
        self.ns['_record_execution_trace'](record, trace)
        self.assertTrue(json.loads(record['execution_details'])['execution_ok'])
        self.assertIn('__lean_capture_v2', record['execution_code'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
