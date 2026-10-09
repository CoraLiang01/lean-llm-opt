"""Canonicalize finite bounds as explicit LP rows; recompute typed WL.

Original inputs are never overwritten. The convention merges singleton rows
with native bounds and emits one row per effective finite lower/upper bound.
Binary variables retain their intrinsic {0,1} domain, also exposed as rows.
WL remains V:B/I/C, R:E/I, cumulative rounds 0..h and cosine normalization.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp
import numpy as np

from wl_bipartite import read_linear_model, self_test, typed_graph, wl_kernels

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return '+infinity' if value > 0 else '-infinity'
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def category(value):
    value = value.strip()
    if value.startswith('Others'):
        return 'Others'
    if value in ('FLP', 'UFLP'):
        return 'UFLP'
    if value not in CATEGORIES:
        raise ValueError(f'Unknown problem category: {value}')
    return value


def inputs():
    ref_csv = ROOT / 'Large_Scale_Or_Files/RAG_Examples_All.csv'
    test_csv = ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101.csv'
    variant_csv = ROOT / 'benchmark_dataset/questions.csv'
    refs = [dict(dataset='reference', instance=f'Ref{i:02d}',
                 category=category(r['Type']),
                 path=ROOT / 'Large_Scale_Or_Files/Ref_Data_Large_Scale_LP' / f'{i}.lp',
                 relative=Path('RefData_LP') / f'{i}.lp')
            for i, r in enumerate(csv_rows(ref_csv), 1)]
    tests = []
    for i, r in enumerate(csv_rows(test_csv), 1):
        found = set(re.findall(r'/([A-Za-z]+)_(?:testing|example)/([^/\s]+)/',
                               r['Dataset_address']))
        assert len(found) == 1, (i, found)
        cat, name = next(iter(found))
        assert cat == r['Problem Type'].strip().split()[0]
        if cat == 'Others':
            name = name.replace('Others', 'Other')
        rel = Path(cat) / name / f'{name}.lp'
        tests.append(dict(dataset='large_scale_101', instance=f'{cat}/{name}',
                          category=cat, problem_id=f'OR-{i:03d}',
                          path=ROOT / 'generated_label_models' / rel,
                          relative=Path('Large_Scale_OR_101') / rel))
    variants = []
    for i, r in enumerate(csv_rows(variant_csv), 1):
        ids = set(map(int, re.findall(r'/Variant(\d+)/', r['Dataset_address'])))
        assert len(ids) == 1, (i, ids)
        vid = next(iter(ids))
        variants.append(dict(dataset='variants', instance=f'Variants{vid}',
            category=category(r['Problem Type']), original_category=r['Problem Type'],
            questions_row=i,
            path=ROOT / 'benchmark_dataset/Variants_1_36_lp' / f'Variants{vid}.lp',
            relative=Path('Variants_1_36_lp') / f'Variants{vid}.lp'))
    variants.sort(key=lambda r: int(r['instance'].replace('Variants', '')))
    assert len(refs) == 15 and len(tests) == 101 and len(variants) == 36
    assert {v['instance'] for v in variants} == {f'Variants{i}' for i in range(1, 37)}
    for directory, records in [
        (ROOT / 'generated_label_models', tests),
        (ROOT / 'Large_Scale_Or_Files/Ref_Data_Large_Scale_LP', refs),
        (ROOT / 'benchmark_dataset/Variants_1_36_lp', variants),
    ]:
        assert {r['path'].resolve() for r in records} == {p.resolve() for p in directory.rglob('*.lp')}
    return refs + tests + variants, [ref_csv, test_csv, variant_csv]


def check_linear(model):
    if model.NumQConstrs or model.NumQNZs or model.NumGenConstrs or model.NumSOS or model.NumPWLObjVars:
        raise ValueError('Only linear models without SOS, general constraints or PWL objectives are supported')
    if any(v.VType not in ('B', 'I', 'C') for v in model.getVars()):
        raise ValueError('Unsupported variable domain')


def effective_form(model):
    """Numerical signature preserving the feasible set in the original domains.

    Division of non-unit singleton coefficients is performed in double precision.
    Variable domains, all non-singleton rows, objective coefficients/constants,
    priorities, weights and tolerances are preserved and subsequently checked.
    """
    check_linear(model)
    vs = model.getVars()
    matrix = model.getA().tocsr()
    matrix.eliminate_zeros()
    bounds = {v.VarName: [float(v.LB), float(v.UB)] for v in vs}
    remaining, singleton = {}, []
    for i, row in enumerate(model.getConstrs()):
        start, stop = matrix.indptr[i:i+2]
        terms = {vs[int(j)].VarName: float(a) for j, a in
                 zip(matrix.indices[start:stop], matrix.data[start:stop])}
        if len(terms) != 1:
            remaining[row.ConstrName] = dict(sense=row.Sense, rhs=float(row.RHS), terms=terms)
            continue
        name, coefficient = next(iter(terms.items()))
        value = float(row.RHS) / coefficient
        sense = row.Sense
        if coefficient < 0:
            sense = {'<': '>', '>': '<', '=': '='}[sense]
        if sense in ('>', '='):
            bounds[name][0] = max(bounds[name][0], value)
        if sense in ('<', '='):
            bounds[name][1] = min(bounds[name][1], value)
        singleton.append(dict(row=row.ConstrName, variable=name,
             coefficient=coefficient, original_sense=row.Sense, rhs=float(row.RHS),
             normalized_sense=sense, normalized_value=value))
    objectives = []
    for i in range(model.NumObj):
        expr = model.getObjective(i) if model.NumObj > 1 else model.getObjective()
        terms = {expr.getVar(j).VarName: float(expr.getCoeff(j)) for j in range(expr.size())}
        terms = {name: x for name, x in terms.items() if x != 0}
        obj = dict(terms=terms, constant=float(expr.getConstant()))
        if model.NumObj > 1:
            model.Params.ObjNumber = i
            obj.update(priority=int(model.ObjNPriority), weight=float(model.ObjNWeight),
                       abstol=float(model.ObjNAbsTol), reltol=float(model.ObjNRelTol))
        objectives.append(obj)
    model.Params.ObjNumber = 0
    return dict(domains={v.VarName: v.VType for v in vs}, bounds=bounds,
                remaining=remaining, objectives=objectives,
                sense=int(model.ModelSense), singleton=singleton)


def same_form(left, right):
    errors = []
    maxima = {'bounds': 0.0, 'matrix': 0.0, 'rhs': 0.0, 'objective': 0.0}
    def numeric(a, b, group, context):
        if a == b:
            return
        if not (math.isfinite(a) and math.isfinite(b)):
            errors.append(context)
            return
        maxima[group] = max(maxima[group], abs(a-b))
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10):
            errors.append(context)
    if left['domains'] != right['domains']:
        errors.append('variable names/domains')
    if left['sense'] != right['sense']:
        errors.append('objective sense')
    if left['bounds'].keys() != right['bounds'].keys():
        errors.append('bound variables')
    else:
        for name, values in left['bounds'].items():
            for i, a in enumerate(values):
                numeric(a, right['bounds'][name][i], 'bounds', f'bound {name}[{i}]')
    if left['remaining'].keys() != right['remaining'].keys():
        errors.append('non-singleton row names')
    else:
        for name, a in left['remaining'].items():
            b = right['remaining'][name]
            if a['sense'] != b['sense']:
                errors.append(f'sense {name}')
            numeric(a['rhs'], b['rhs'], 'rhs', f'RHS {name}')
            if a['terms'].keys() != b['terms'].keys():
                errors.append(f'support {name}')
            else:
                for var, val in a['terms'].items():
                    numeric(val, b['terms'][var], 'matrix', f'coefficient {name}/{var}')
    if len(left['objectives']) != len(right['objectives']):
        errors.append('objective count')
    else:
        for i, a in enumerate(left['objectives']):
            b = right['objectives'][i]
            if a['terms'].keys() != b['terms'].keys():
                errors.append(f'objective support {i}')
            else:
                for var, val in a['terms'].items():
                    numeric(val, b['terms'][var], 'objective', f'objective {i}/{var}')
            numeric(a['constant'], b['constant'], 'objective', f'objective constant {i}')
            for key in ('priority', 'weight', 'abstol', 'reltol'):
                if a.get(key) != b.get(key):
                    errors.append(f'objective {i} {key}')
    return dict(passed=not errors, errors=errors, maximum_absolute_errors=maxima)


def canonicalize(model):
    original = effective_form(model)
    normalized = model.copy()
    old_rows = {c.ConstrName: c for c in normalized.getConstrs()}
    normalized.remove([old_rows[r['row']] for r in original['singleton']])
    normalized.update()
    occupied = {c.ConstrName for c in normalized.getConstrs()}
    bound_rows = []
    for i, var in enumerate(normalized.getVars()):
        lo, hi = original['bounds'][var.VarName]
        if var.VType == 'B':
            # Intrinsic binary domain cannot be removed while retaining V:B.
            var.LB, var.UB = 0.0, 1.0
        else:
            var.LB, var.UB = -gp.GRB.INFINITY, gp.GRB.INFINITY
        for kind, value in [('lower', lo), ('upper', hi)]:
            if not math.isfinite(value):
                continue
            name = f'__wl_bound_{i+1:05d}_{kind}'
            while name in occupied:
                name += '_'
            occupied.add(name)
            normalized.addConstr(var >= value if kind == 'lower' else var <= value, name=name)
            bound_rows.append(dict(variable=var.VarName, kind=kind, value=value, row=name))
    normalized.update()
    validation = same_form(original, effective_form(normalized))
    if not validation['passed']:
        raise AssertionError(validation)
    return normalized, original, bound_rows, validation


def number(value):
    if not math.isfinite(value):
        raise ValueError('An LP row/objective cannot contain an infinite coefficient')
    return format(value, '.17g') if value else '0'


def expression(terms, constant=0.0, zero_var=None):
    tokens = []
    for name, coefficient in terms:
        if coefficient:
            tokens.append(f'{"+" if coefficient > 0 else "-"} {number(abs(coefficient))} {name}')
    if constant:
        # Gurobi's LP convention encodes an objective offset with the
        # reserved, fixed "Constant" symbol, folded away on reading.
        tokens.append(f'{"+" if constant > 0 else "-"} {number(abs(constant))} Constant')
    # A bare "0" can be parsed as a variable name by the LP reader.
    return tokens or [f'0 {zero_var}']


def wrapped(prefix, tokens, suffix=''):
    lines = []
    current = prefix
    for token in tokens:
        if len(current) + len(token) + 1 > 850:
            lines.append(current.rstrip())
            current = '    '
        current += (' ' if current and not current.endswith(' ') else '') + token
    if len(current) + len(suffix) > 950:
        lines.append(current.rstrip())
        current = '    '
    lines.append(current + suffix)
    return lines


def write_full_precision_lp(model, path):
    """Write supported linear LP/MILP models with round-trip double precision."""
    if model.NumVars == 0:
        raise ValueError('The explicit-bound LP writer requires at least one variable')
    zero_var = model.getVars()[0].VarName
    lines = [r'\ Canonical explicit-bound formulation for WL comparison.',
             r'\ Finite bounds and singleton restrictions merged; original variable domains retained.']
    header = 'Minimize' if model.ModelSense == 1 else 'Maximize'
    has_objective_constant = False
    if model.NumObj > 1:
        lines.append(header + ' multi-objectives')
        for i in range(model.NumObj):
            model.Params.ObjNumber = i
            lines.append(f'  OBJ{i}: Priority={model.ObjNPriority} Weight={number(model.ObjNWeight)} '
                         f'AbsTol={number(model.ObjNAbsTol)} RelTol={number(model.ObjNRelTol)}')
            expr = model.getObjective(i)
            has_objective_constant |= bool(expr.getConstant())
            lines.extend(wrapped('   ', expression(
                [(expr.getVar(j).VarName, expr.getCoeff(j)) for j in range(expr.size())], expr.getConstant(), zero_var)))
    else:
        lines.append(header)
        expr = model.getObjective()
        has_objective_constant = bool(expr.getConstant())
        lines.extend(wrapped('  obj: ', expression(
            [(expr.getVar(j).VarName, expr.getCoeff(j)) for j in range(expr.size())], expr.getConstant(), zero_var)))
    model.Params.ObjNumber = 0
    lines.append('Subject To')
    matrix = model.getA().tocsr()
    variables = model.getVars()
    for i, row in enumerate(model.getConstrs()):
        start, stop = matrix.indptr[i:i+2]
        terms = [(variables[int(j)].VarName, float(a)) for j, a in
                 zip(matrix.indices[start:stop], matrix.data[start:stop])]
        operator = {'<': '<=', '>': '>=', '=': '='}[row.Sense]
        lines.extend(wrapped(f'  {row.ConstrName}: ', expression(terms, zero_var=zero_var),
                             f' {operator} {number(row.RHS)}'))
    lines.append('Bounds')
    if has_objective_constant:
        assert 'Constant' not in {v.VarName for v in variables}
        lines.append('  Constant = 1')
    for var in variables:
        if var.VType != 'B':
            assert not math.isfinite(var.LB) and not math.isfinite(var.UB)
            lines.append(f'  {var.VarName} free')
    for kind, section in [('I', 'Generals'), ('B', 'Binaries')]:
        names = [v.VarName for v in variables if v.VType == kind]
        if names:
            lines.append(section)
            lines.extend(wrapped('  ', names))
    lines.append('End')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def direct_signature(model):
    """Check lossless serialization of the already-normalized formulation."""
    check_linear(model)
    vs = model.getVars()
    matrix = model.getA().tocsr()
    rows = {}
    for i, row in enumerate(model.getConstrs()):
        a, b = matrix.indptr[i:i+2]
        rows[row.ConstrName] = (row.Sense, float(row.RHS),
            {vs[int(j)].VarName: float(x) for j, x in zip(matrix.indices[a:b], matrix.data[a:b]) if x != 0})
    f = effective_form(model)
    return dict(variables={v.VarName: (v.VType, float(v.LB), float(v.UB)) for v in vs},
                rows=rows, objectives=f['objectives'], sense=f['sense'])


def convention_tests(env, out):
    graphs, checks = [], []
    for convention in ('native', 'explicit', 'scaled_and_redundant'):
        m = gp.Model(convention, env=env)
        x = m.addVar(name='x', ub=3 if convention == 'native' else gp.GRB.INFINITY)
        y = m.addVar(name='y')
        m.addConstr(x+y <= 5, name='coupling')
        if convention == 'explicit':
            m.addConstr(x <= 3, name='bound')
        if convention == 'scaled_and_redundant':
            m.addConstr(2*x <= 6, name='scaled_upper')
            m.addConstr(x <= 5, name='redundant_upper')
            m.addConstr(-2*x <= 0, name='scaled_lower')
        m.update()
        n, _, _, v = canonicalize(m)
        p = out / 'validation_examples' / f'{convention}.lp'
        write_full_precision_lp(n, p)
        reread = gp.read(str(p), env=env)
        assert direct_signature(n) == direct_signature(reread)
        graphs.append(typed_graph(read_linear_model(p, env)))
        checks.append(dict(input_convention=convention, canonical_constraints=n.NumConstrs,
                           validation=v['passed']))
        reread.dispose(); n.dispose(); m.dispose()
    for matrix in wl_kernels(graphs, 3).values():
        assert np.allclose(matrix, 1, atol=1e-12, rtol=0)
    return dict(equivalent_bound_encodings_similarity=1.0, tests=checks)


def solve_record(model, seconds):
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = seconds
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.MIPGap = 0
    model.optimize()
    status = {gp.GRB.OPTIMAL: 'OPTIMAL', gp.GRB.INFEASIBLE: 'INFEASIBLE',
        gp.GRB.UNBOUNDED: 'UNBOUNDED', gp.GRB.INF_OR_UNBD: 'INF_OR_UNBD',
        gp.GRB.TIME_LIMIT: 'TIME_LIMIT'}.get(model.Status, str(model.Status))
    objectives = [float((model.getObjective(i) if model.NumObj > 1 else model.getObjective()).getValue()) for i in range(model.NumObj)] if model.SolCount else []
    return dict(status=status, solution_count=int(model.SolCount),
                objective_values=objectives, runtime=float(model.Runtime))


def compare_solutions(a, b):
    values = len(a['objective_values']) == len(b['objective_values']) and all(
        math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-6)
        for x, y in zip(a['objective_values'], b['objective_values']))
    return dict(both_optimal=a['status'] == b['status'] == 'OPTIMAL',
                status_match=a['status'] == b['status'],
                objective_values_match=bool(values and a['objective_values']),
                matched_optimal=a['status'] == b['status'] == 'OPTIMAL' and values)


def summaries(nearest, refs, h, dataset):
    output = []
    for cat in CATEGORIES + ['All']:
        chosen = [r for r in nearest if cat == 'All' or r['category'] == cat]
        same = [r['similarity_same'] for r in chosen if r['similarity_same'] is not None]
        all_values = [r['similarity_all'] for r in chosen]
        output.append(dict(dataset=dataset, h=h, category=cat, instances=len(chosen),
            same_category_reference_count=len(refs) if cat == 'All' else sum(r['category'] == cat for r in refs),
            median_same=float(np.median(same)) if same else None,
            mean_same=float(np.mean(same)) if same else None,
            q25_same=float(np.percentile(same, 25)) if same else None,
            q75_same=float(np.percentile(same, 75)) if same else None,
            median_all=float(np.median(all_values)) if all_values else None,
            mean_all=float(np.mean(all_values)) if all_values else None,
            equal_one_same=int(np.count_nonzero(np.isclose(same, 1, atol=1e-12, rtol=0)))))
    return output


def comparisons(records, graphs, out):
    kernels = wl_kernels(graphs, 3)
    refs = [r for r in records if r['dataset'] == 'reference']
    assert len(refs) == 15 and all(r['dataset'] == 'reference' for r in records[:15])
    all_summaries = []
    for dataset in ('large_scale_101', 'variants'):
        indices = [i for i, r in enumerate(records) if r['dataset'] == dataset]
        tests = [records[i] for i in indices]
        target = out / dataset
        for h in (1, 2, 3):
            k = kernels[h]
            assert np.allclose(k, k.T) and np.allclose(np.diag(k), 1)
            cross = k[np.ix_(indices, range(15))]
            write_csv(target / f'pairwise_h{h}.csv', [dict(instance=r['instance'], category=r['category'],
                **{ref['instance']: float(cross[i, j]) for j, ref in enumerate(refs)}) for i, r in enumerate(tests)])
            nearest = []
            for i, r in enumerate(tests):
                same = [j for j, ref in enumerate(refs) if ref['category'] == r['category']]
                best = int(np.argmax(cross[i]))
                j = max(same, key=lambda j: cross[i, j]) if same else None
                nearest.append(dict(instance=r['instance'], category=r['category'],
                    nearest_same=refs[j]['instance'] if j is not None else '',
                    similarity_same=float(cross[i, j]) if j is not None else None,
                    nearest_all=refs[best]['instance'],
                    nearest_all_category=refs[best]['category'], similarity_all=float(cross[i, best])))
            write_csv(target / f'nearest_h{h}.csv', nearest)
            all_summaries.extend(summaries(nearest, refs, h, dataset))
        write_csv(target / 'category_summary.csv', [r for r in all_summaries if r['dataset'] == dataset])
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/wl_canonical_bounds_20261006')
    parser.add_argument('--solve-time-limit', type=float, default=20)
    parser.add_argument('--skip-solves', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, csvs = inputs()
    source_hashes = {str(p): digest(p) for p in csvs + [r['path'] for r in records]}
    manifest, bounds_log, unary_log, validations, solve_log = [], [], [], [], []
    original_graphs, canonical_graphs = [], []
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        tests = convention_tests(env, out)
        for index, rec in enumerate(records, 1):
            source = gp.read(str(rec['path']), env=env)
            normalized, old_form, bound_rows, semantic = canonicalize(source)
            path = out / 'models' / rec['relative']
            write_full_precision_lp(normalized, path)
            mps = out / 'models_mps' / rec['relative'].with_suffix('.mps')
            mps.parent.mkdir(parents=True, exist_ok=True)
            normalized.write(str(mps))
            reread = gp.read(str(path), env=env)
            mps_read = gp.read(str(mps), env=env)
            roundtrip = direct_signature(normalized) == direct_signature(reread)
            mps_roundtrip = direct_signature(normalized) == direct_signature(mps_read)
            exported_semantic = same_form(old_form, effective_form(reread))
            second, _, _, _ = canonicalize(reread)
            idempotence = same_form(effective_form(reread), effective_form(second))['passed']
            assert roundtrip and mps_roundtrip and idempotence and exported_semantic['passed'], (rec, exported_semantic)
            assert source.NumVars == normalized.NumVars == reread.NumVars
            nonbinary = [v for v in reread.getVars() if v.VType != 'B']
            assert all(not math.isfinite(v.LB) and not math.isfinite(v.UB) for v in nonbinary)
            original_graphs.append(typed_graph(read_linear_model(rec['path'], env)))
            canonical_graphs.append(typed_graph(read_linear_model(path, env)))
            basic = {k: str(v) if isinstance(v, Path) else v for k, v in rec.items()}
            manifest.append({**basic, 'source_sha256': digest(rec['path']),
                'canonical_path': str(path), 'canonical_sha256': digest(path),
                'canonical_mps_path': str(mps), 'canonical_mps_sha256': digest(mps),
                'variables': source.NumVars, 'source_constraints': source.NumConstrs,
                'canonical_constraints': reread.NumConstrs,
                'source_nonzeros': source.NumNZs, 'canonical_nonzeros': reread.NumNZs,
                'removed_singleton_rows': len(old_form['singleton']),
                'explicit_bound_rows': len(bound_rows), 'objective_count': source.NumObj})
            varmap = {v.VarName: v for v in source.getVars()}
            for b in bound_rows:
                bounds_log.append(dict(dataset=rec['dataset'], instance=rec['instance'],
                    **b, native_lower=varmap[b['variable']].LB, native_upper=varmap[b['variable']].UB))
            unary_log.extend(dict(dataset=rec['dataset'], instance=rec['instance'], **r) for r in old_form['singleton'])
            validations.append(dict(dataset=rec['dataset'], instance=rec['instance'],
                feasible_set_signature_matches=exported_semantic['passed'],
                lp_roundtrip_exact=roundtrip, mps_roundtrip_exact=mps_roundtrip,
                canonicalization_idempotent=idempotence,
                maximum_absolute_errors=exported_semantic['maximum_absolute_errors']))
            if not args.skip_solves:
                original_result = solve_record(source, args.solve_time_limit)
                canonical_result = solve_record(reread, args.solve_time_limit)
                solve_log.append(dict(dataset=rec['dataset'], instance=rec['instance'],
                    original=original_result, canonical=canonical_result,
                    **compare_solutions(original_result, canonical_result)))
            source.dispose(); normalized.dispose(); reread.dispose(); mps_read.dispose(); second.dispose()
            if index % 10 == 0 or index == len(records):
                print(json.dumps(dict(progress=index, total=len(records), last=rec['instance'],
                    paired_optimal=sum(r['matched_optimal'] for r in solve_log)), ensure_ascii=False), flush=True)
    write_csv(out / 'normalization_manifest.csv', manifest)
    write_csv(out / 'effective_bound_rows.csv', [json_safe(r) for r in bounds_log])
    write_csv(out / 'absorbed_singleton_rows.csv', unary_log)
    write_csv(out / 'variant_category_mapping.csv', [{k: r[k] for k in
        ('instance', 'questions_row', 'category', 'original_category')} for r in records if r['dataset'] == 'variants'])
    baseline = comparisons(records, original_graphs, out / 'original')
    canonical = comparisons(records, canonical_graphs, out)
    prior = {(r['dataset'], r['h'], r['category']): r for r in baseline}
    changes = []
    for r in canonical:
        b = prior[(r['dataset'], r['h'], r['category'])]
        changes.append(dict(dataset=r['dataset'], h=r['h'], category=r['category'], instances=r['instances'],
            original_median_same=b['median_same'], canonical_median_same=r['median_same'],
            median_same_change=r['median_same']-b['median_same'] if r['median_same'] is not None else None,
            original_mean_same=b['mean_same'], canonical_mean_same=r['mean_same'],
            original_median_all=b['median_all'], canonical_median_all=r['median_all']))
    write_csv(out / 'category_before_after.csv', changes)
    write_json(out / 'conversion_validation.json', validations)
    write_json(out / 'solver_validation.json', solve_log)
    write_csv(out / 'solver_validation.csv', [dict(dataset=r['dataset'], instance=r['instance'],
        original_status=r['original']['status'], canonical_status=r['canonical']['status'],
        original_objectives=json.dumps(r['original']['objective_values']),
        canonical_objectives=json.dumps(r['canonical']['objective_values']),
        matched_optimal=r['matched_optimal'], original_runtime=r['original']['runtime'],
        canonical_runtime=r['canonical']['runtime']) for r in solve_log])
    unchanged = all(digest(path) == value for path, value in source_hashes.items())
    assert unchanged
    audit = dict(computed_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=dict(Counter(r['dataset'] for r in records)), source_files_unchanged=unchanged,
        source_sha256=source_hashes, gurobi_version=list(gp.gurobi.version()),
        conventions=dict(singletons='Merge all one-nonzero linear rows into native bound intersections.',
            lower_upper='Emit one explicit >= lower row and one <= upper row per effective finite bound.',
            default_bounds='Include default/native bounds, including nonnegativity and binary 0/1.',
            fixed_variables='Retain variable and emit both coincident bound rows; no variable elimination.',
            binary='Preserve intrinsic {0,1} variable domain; its numerical bounds are also exposed as rows.',
            integer_rounding='No extra ceil/floor tightening; retain the original variable domains.',
            other_rows='Retain all multi-variable and zero-nonzero rows unchanged.',
            precision='LP uses 17 significant digits; MPS also exported. Exact normalized-model round trips checked.',
            objective='Preserve all objectives, constants, sense, priorities, weights and tolerances.'),
        conversion_passed=len(validations), convention_tests=tests, wl_tests=self_test(),
        WL=dict(labels='V:B/I/C and R:E/I', primary_h=2, sensitivity_h=[1,3],
            normalization='Cosine of concatenated round 0..h label counts',
            ignored='Numerical bound values, objective, matrix coefficient magnitudes/signs, RHS, names.',
            general_presolve=False), solver_validation=dict(executed=not args.skip_solves,
            paired_cases=len(solve_log), paired_optimal_matches=sum(r['matched_optimal'] for r in solve_log),
            time_limit_per_model=args.solve_time_limit,
            unresolved=[r['instance'] for r in solve_log if not r['matched_optimal']]),
        script_sha256=digest(__file__), wl_module_sha256=digest(ROOT / 'wl_bipartite.py'))
    write_json(out / 'audit.json', audit)
    print(json.dumps(dict(audit={k: audit[k] for k in ('inputs','conversion_passed','source_files_unchanged','solver_validation')},
        results_h2=[r for r in canonical if r['h']==2]), indent=2), flush=True)


if __name__ == '__main__':
    main()
