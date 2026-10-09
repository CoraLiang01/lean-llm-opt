"""Canonicalize singleton restrictions into native bounds and exclude them from WL.

Uses exactly the input inventory, domains, WL labels and depths of
canonicalize_and_recompute_wl.py. Original files are never overwritten.
"""
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp
import numpy as np

from canonicalize_and_recompute_wl import (
    ROOT, comparisons, digest, direct_signature, effective_form, expression,
    inputs, number, same_form, wrapped, write_csv, write_json,
)
from wl_bipartite import read_linear_model, typed_graph, wl_kernels

OUT = ROOT / 'outputs/wl_bounds_excluded_20261006'
INCLUDED = ROOT / 'outputs/wl_canonical_bounds_20261006'


def integer_lattice_form(form):
    """Compare bounds on their feasible integer lattice, not fractional endpoints.

    Gurobi's LP reader rounds native integer bounds. For an integer x,
    31.7 <= x <= 40.9 has exactly the same feasible values as 32 <= x <= 40.
    """
    projected = dict(form)
    projected['bounds'] = {name: list(values) for name, values in form['bounds'].items()}
    for name, domain in form['domains'].items():
        if domain in ('I', 'B'):
            lo, hi = projected['bounds'][name]
            projected['bounds'][name] = [math.ceil(lo) if math.isfinite(lo) else lo,
                                         math.floor(hi) if math.isfinite(hi) else hi]
    return projected


def same_feasible_form(left, right):
    return same_form(integer_lattice_form(left), integer_lattice_form(right))


def native_bound_form(model):
    original = effective_form(model)
    lattice = integer_lattice_form(original)
    converted = model.copy()
    rows = {c.ConstrName: c for c in converted.getConstrs()}
    converted.remove([rows[r['row']] for r in original['singleton']])
    converted.update()
    for v in converted.getVars():
        v.LB, v.UB = lattice['bounds'][v.VarName]
    converted.update()
    checked = same_feasible_form(original, effective_form(converted))
    assert checked['passed'], checked
    return converted, original, checked


def write_lp(model, path):
    """Full-precision linear LP export including effective native bounds."""
    variables = model.getVars()
    assert variables
    zero_var = variables[0].VarName
    lines = [r'\ Canonical native-bound formulation; bounds excluded from WL graph.']
    header = 'Minimize' if model.ModelSense == 1 else 'Maximize'
    lines.append(header + (' multi-objectives' if model.NumObj > 1 else ''))
    has_constant = False
    for i in range(model.NumObj):
        if model.NumObj > 1:
            model.Params.ObjNumber = i
            lines.append(f'  OBJ{i}: Priority={model.ObjNPriority} Weight={number(model.ObjNWeight)} '
                         f'AbsTol={number(model.ObjNAbsTol)} RelTol={number(model.ObjNRelTol)}')
            expr = model.getObjective(i)
            prefix = '    '
        else:
            expr = model.getObjective()
            prefix = '  obj: '
        has_constant |= bool(expr.getConstant())
        lines.extend(wrapped(prefix, expression(
            [(expr.getVar(j).VarName, expr.getCoeff(j)) for j in range(expr.size())],
            expr.getConstant(), zero_var)))
    model.Params.ObjNumber = 0
    lines.append('Subject To')
    matrix = model.getA().tocsr()
    for i, row in enumerate(model.getConstrs()):
        start, stop = matrix.indptr[i:i+2]
        terms = [(variables[int(j)].VarName, float(a)) for j, a in
                 zip(matrix.indices[start:stop], matrix.data[start:stop])]
        operator = {'<': '<=', '>': '>=', '=': '='}[row.Sense]
        lines.extend(wrapped(f'  {row.ConstrName}: ', expression(terms, zero_var=zero_var),
                             f' {operator} {number(row.RHS)}'))
    lines.append('Bounds')
    if has_constant:
        assert 'Constant' not in {v.VarName for v in variables}
        lines.append('  Constant = 1')
    for v in variables:
        if not math.isfinite(v.LB) and not math.isfinite(v.UB):
            lines.append(f'  {v.VarName} free')
        elif v.LB == v.UB:
            lines.append(f'  {v.VarName} = {number(v.LB)}')
        else:
            lower = number(v.LB) if math.isfinite(v.LB) else '-inf'
            upper = number(v.UB) if math.isfinite(v.UB) else '+inf'
            lines.append(f'  {lower} <= {v.VarName} <= {upper}')
    for kind, section in [('I', 'Generals'), ('B', 'Binaries')]:
        names = [v.VarName for v in variables if v.VType == kind]
        if names:
            lines.append(section)
            lines.extend(wrapped('  ', names))
    lines.append('End')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, csvs = inputs()
    hashes = {str(p): digest(p) for p in csvs + [r['path'] for r in records]}
    manifests, validations, graphs = [], [], []
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for r in records:
            source = gp.read(str(r['path']), env=env)
            native, old_form, semantic = native_bound_form(source)
            path = OUT / 'models' / r['relative']
            write_lp(native, path)
            mps = OUT / 'models_mps' / r['relative'].with_suffix('.mps')
            mps.parent.mkdir(parents=True, exist_ok=True)
            native.write(str(mps))
            reread = gp.read(str(path), env=env)
            mps_read = gp.read(str(mps), env=env)
            lp_exact = direct_signature(native) == direct_signature(reread)
            mps_exact = direct_signature(native) == direct_signature(mps_read)
            semantic = same_feasible_form(old_form, effective_form(reread))
            # Compare with the already-validated bound-included model too.
            included = gp.read(str(INCLUDED / 'models' / r['relative']), env=env)
            cross_mode = same_feasible_form(effective_form(included), effective_form(reread))
            assert lp_exact and mps_exact and semantic['passed'] and cross_mode['passed'], (r, semantic)
            assert reread.NumVars == source.NumVars
            assert all(reread.getRow(c).size() != 1 for c in reread.getConstrs())
            graph = typed_graph(read_linear_model(path, env))
            graphs.append(graph)
            manifests.append(dict(dataset=r['dataset'], instance=r['instance'], category=r['category'],
                source_path=str(r['path']), source_sha256=digest(r['path']),
                native_bound_path=str(path), native_bound_sha256=digest(path),
                mps_path=str(mps), variables=reread.NumVars,
                source_constraints=source.NumConstrs, graph_constraints=reread.NumConstrs,
                graph_edges=reread.NumNZs, absorbed_singleton_rows=len(old_form['singleton']),
                isolated_variable_nodes=sum(not a for a in graph[1][:reread.NumVars]),
                integer_variables_with_rounded_bounds=sum(
                    old_form['bounds'][v.VarName] != [v.LB,v.UB]
                    for v in reread.getVars() if v.VType in ('B','I'))))
            validations.append(dict(dataset=r['dataset'], instance=r['instance'],
                lp_roundtrip_exact=lp_exact, mps_roundtrip_exact=mps_exact,
                source_semantics_preserved=semantic['passed'],
                same_semantics_as_explicit_bound_model=cross_mode['passed'],
                maximum_absolute_errors=semantic['maximum_absolute_errors']))
            source.dispose(); native.dispose(); reread.dispose(); mps_read.dispose(); included.dispose()
        # Explain the NRM effect with a counterfactual graph only; keep source files intact.
        ref = gp.read(str(OUT/'models/RefData_LP/3.lp'), env=env)
        dispatch = ref.getConstrByName('dispatch_capacity')
        assert dispatch is not None
        ref.remove(dispatch); ref.update()
        control_path = OUT/'validation_examples/Ref03_without_dispatch_capacity.lp'
        write_lp(ref, control_path)
        control = typed_graph(read_linear_model(control_path, env))
        nrm_indices = [i for i, r in enumerate(records) if r['dataset']=='large_scale_101' and r['category']=='NRM']
        control_scores = wl_kernels([control]+[graphs[i] for i in nrm_indices], 3)[2][0,1:]
        assert np.allclose(control_scores, 1, atol=1e-12, rtol=0)
        ref.dispose()
    summary = comparisons(records, graphs, OUT)
    write_csv(OUT/'normalization_manifest.csv', manifests)
    write_json(OUT/'conversion_validation.json', validations)
    primary = [r for r in summary if r['h']==2]
    write_csv(OUT/'category_summary_h2.csv', primary)
    changes = []
    for dataset in ('large_scale_101', 'variants'):
        with (INCLUDED/dataset/'category_summary.csv').open(encoding='utf-8-sig', newline='') as f:
            included_rows = {(int(q['h']), q['category']):q for q in csv.DictReader(f)}
        for r in [q for q in summary if q['dataset']==dataset]:
            previous = included_rows[(r['h'],r['category'])]
            a = float(previous['median_same']) if previous['median_same'] else None
            b = float(previous['median_all']) if previous['median_all'] else None
            changes.append(dict(dataset=dataset,h=r['h'],category=r['category'],instances=r['instances'],
                bounds_included_median_same=a, bounds_excluded_median_same=r['median_same'],
                same_change=r['median_same']-a if a is not None else None,
                bounds_included_median_all=b,bounds_excluded_median_all=r['median_all']))
    write_csv(OUT/'included_vs_excluded.csv', changes)
    unchanged = all(digest(p)==h for p,h in hashes.items())
    assert unchanged
    audit = dict(computed_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=dict(Counter(r['dataset'] for r in records)),
        source_files_unchanged=unchanged, source_sha256=hashes,
        conversion_passed=len(validations), gurobi_version=list(gp.gurobi.version()),
        conventions=dict(singletons='Intersect singleton rows with native bounds; remove all singleton rows.',
            bound_values='Retained in Bounds and excluded from the WL graph.',
            integer_bounds='Normalize finite integer bounds with ceil(lower)/floor(upper), preserving the feasible integer lattice and avoiding LP-reader rounding differences.',
            variable_types='Retain original B/I/C domains.',
            isolated_variables='Retained as graph nodes; no variable elimination or general presolve.',
            other_rows='All non-singleton and zero-nonzero rows retained unchanged.',
            ignored='Bounds, objective, coefficient values/signs, RHS, variable/row names.',
            primary_h=2,sensitivity_h=[1,3],normalization='Cosine of cumulative round 0..h WL counts'),
        nrm_control=dict(description='Graph-only removal of the previously added dispatch_capacity reference row.',
            instances=25, median_without_dispatch_capacity=float(np.median(control_scores)),
            original_reference_file_modified=False),
        solver_validation='No new solver experiment; deterministic semantic checks against original and already solved explicit-bound formulations.',
        script_sha256=digest(__file__),wl_module_sha256=digest(ROOT/'wl_bipartite.py'))
    write_json(OUT/'audit.json',audit)
    print(json.dumps(dict(audit={k:audit[k] for k in ['inputs','conversion_passed','source_files_unchanged']},results_h2=primary),indent=2),flush=True)


if __name__ == '__main__':
    main()
