"""Rebuild reviewed CSV Code fields in memory and compare them with reference LPs.

No optimize call is executed, and no input is rewritten. The reviewed code is
executed only up to its first optimize call, replaced by an uncaught capture.
This verifies Code-to-LP correspondence, not natural-language modeling validity.
"""
import ast
import contextlib
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import gurobipy as gp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CSV_PATH = ROOT / 'Large_Scale_Or_Files/RAG_Examples_All.csv'
LP_DIR = ROOT / 'Ref_Data_Large_Scale_LP'


class CapturedModel(BaseException):
    def __init__(self, model):
        self.model = model


def capture(model):
    model.update()
    raise CapturedModel(model)


class CaptureBeforeSolve(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'optimize':
            return ast.copy_location(ast.Call(ast.Name('_capture', ast.Load()), [node.func.value], []), node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'gp' and node.func.attr == 'Model':
                node.keywords.append(ast.keyword('env', ast.Name('_env', ast.Load())))
        return node


def snapshot(model):
    model.update()
    assert not (model.NumQConstrs or model.NumGenConstrs or model.NumSOS or model.NumQNZs)
    variables, rows = model.getVars(), model.getConstrs()
    normalize = lambda name: re.sub(r'\s', '_', name)
    vnames = [normalize(v.VarName) for v in variables]
    rnames = [normalize(r.ConstrName) for r in rows]
    assert len(set(vnames)) == len(vnames) and len(set(rnames)) == len(rnames)
    vi, ri = np.argsort(vnames), np.argsort(rnames)
    variables, rows = [variables[i] for i in vi], [rows[i] for i in ri]
    exact = dict(variable_names=sorted(vnames), constraint_names=sorted(rnames),
                 variable_types=model.getAttr('VType', variables),
                 constraint_senses=model.getAttr('Sense', rows),
                 objective_sense=model.ModelSense, number_of_objectives=model.NumObj)
    numeric = dict(A=model.getA().toarray()[ri][:, vi],
                   lower_bounds=model.getAttr('LB', variables),
                   upper_bounds=model.getAttr('UB', variables),
                   RHS=model.getAttr('RHS', rows))
    if model.NumObj > 1:
        for i in range(model.NumObj):
            model.Params.ObjNumber = i
            numeric[f'objective_{i}_coefficients'] = model.getAttr('ObjN', variables)
            numeric[f'objective_{i}_settings'] = [model.ObjNCon, model.ObjNPriority,
                model.ObjNWeight, model.ObjNAbsTol, model.ObjNRelTol]
    else:
        numeric['objective_coefficients'] = model.getAttr('Obj', variables)
        numeric['objective_constant'] = [model.ObjCon]
    return exact, numeric


def main():
    records = []
    with CSV_PATH.open(encoding='utf-8-sig', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for i, row in enumerate(csv_rows, 1):
            tree = ast.fix_missing_locations(CaptureBeforeSolve().visit(ast.parse(row['Code'])))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(tree, f'RAG_Examples_All.csv:row{i}', 'exec'), {'_env': env, '_capture': capture})
            except CapturedModel as captured:
                rebuilt = captured.model
            else:
                raise RuntimeError(f'No model captured for CSV row {i}')
            lp_path = LP_DIR / f'{i}.lp'
            loaded = gp.read(str(lp_path), env=env)
            a, an = snapshot(rebuilt)
            b, bn = snapshot(loaded)
            mismatches = [key for key in a if a[key] != b[key]]
            if set(an) != set(bn):
                mismatches.append('numeric_field_set')
            errors = {}
            for key in an.keys() & bn.keys():
                x, y = np.asarray(an[key]), np.asarray(bn[key])
                if x.shape != y.shape or not np.allclose(x, y, atol=1e-10, rtol=1e-12):
                    mismatches.append(key)
                if x.shape == y.shape:
                    finite = np.isfinite(x) & np.isfinite(y)
                    errors[key] = float(np.max(np.abs(x[finite] - y[finite]), initial=0))
            rec = dict(csv_row=i, category=row['Type'].strip(), lp_path=str(lp_path),
                variables=rebuilt.NumVars, constraints=rebuilt.NumConstrs,
                nonzeros=rebuilt.NumNZs, objectives=rebuilt.NumObj,
                matches=not mismatches, mismatches=mismatches,
                maximum_absolute_errors=errors,
                code_sha256=hashlib.sha256(row['Code'].encode()).hexdigest(),
                lp_sha256=hashlib.sha256(lp_path.read_bytes()).hexdigest())
            records.append(rec)
            print(f'{i:02d} {rec["category"]:7s} -> {i}.lp: {"PASS" if rec["matches"] else mismatches}')
            rebuilt.dispose()
            loaded.dispose()
    result = dict(csv_path=str(CSV_PATH), csv_sha256=hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
        method='Rebuild CSV Code before optimize; align variable/constraint names after whitespace-to-underscore LP export conversion; compare matrix, types, bounds, row senses, RHS, all objectives and multiobjective settings.',
        absolute_tolerance=1e-10, relative_tolerance=1e-12,
        optimization_executed=False, inputs_modified=False,
        all_match=all(r['matches'] for r in records), count=len(records), records=records,
        limitation='Does not validate that Code faithfully expresses all natural-language prompt/Label requirements, or audit the generation of the 101 test LPs.')
    (OUT / 'reference_correspondence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['all_match'], 'Some CSV Code fields differ from their reference LPs; see JSON.'


if __name__ == '__main__':
    main()
