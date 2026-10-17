"""Reproducible typed bipartite WL similarity for LP/MILP formulations.

Run from any directory using --root to identify the original project.
Source LP files are read only. No optimizer or LLM evaluation is run.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

import gurobipy as gp
import numpy as np

from wl_core import wl_kernels, self_test

CATEGORIES = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']
HERE = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_from_lp(path, env):
    """Absorb singleton rows into bounds; construct the remaining support graph.

    Effective bounds are recorded for audit, but not encoded as graph features.
    No source model or file is changed. Fixed and isolated variables are kept.
    """
    model = gp.read(str(path), env=env)
    try:
        if (model.NumQConstrs or model.NumQNZs or model.NumGenConstrs
                or model.NumSOS or model.NumPWLObjVars):
            raise ValueError(f'Only linear models without SOS/general/PWL content: {path}')
        variables, rows = model.getVars(), model.getConstrs()
        types = [v.VType for v in variables]
        if not set(types).issubset({'B', 'I', 'C'}):
            raise ValueError(f'Unsupported variable domains: {path}')
        matrix = model.getA().tocsr()
        matrix.eliminate_zeros()
        bounds = [[float(v.LB), float(v.UB)] for v in variables]
        retained, singleton_log = [], []
        for i, row in enumerate(rows):
            start, stop = matrix.indptr[i:i+2]
            if stop-start != 1:
                retained.append(i)
                continue
            j = int(matrix.indices[start])
            coefficient = float(matrix.data[start])
            value = float(row.RHS)/coefficient
            sense = row.Sense
            if coefficient < 0:
                sense = {'<': '>', '>': '<', '=': '='}[sense]
            if sense in ('>', '='):
                bounds[j][0] = max(bounds[j][0], value)
            if sense in ('<', '='):
                bounds[j][1] = min(bounds[j][1], value)
            singleton_log.append(dict(row=row.ConstrName, variable=variables[j].VarName,
                                      coefficient=coefficient, bound_sense=sense, bound_value=value))
        # Integer restrictions have the same feasible lattice after rounding.
        # Gurobi infinity is a large sentinel; isfinite alone is insufficient.
        bound_log = []
        for j, (lo, hi) in enumerate(bounds):
            lo = lo if lo > -gp.GRB.INFINITY else -math.inf
            hi = hi if hi < gp.GRB.INFINITY else math.inf
            if types[j] in ('I', 'B'):
                lo = math.ceil(lo) if math.isfinite(lo) else lo
                hi = math.floor(hi) if math.isfinite(hi) else hi
            bound_log.append(dict(variable=variables[j].VarName, domain=types[j],
                lower=lo if math.isfinite(lo) else '-inf',
                upper=hi if math.isfinite(hi) else '+inf', inconsistent=lo > hi))
        n = len(variables)
        labels = ['V:'+t for t in types]
        labels += ['R:E' if rows[i].Sense == '=' else 'R:I' for i in retained]
        adjacency = [[] for _ in labels]
        for k, i in enumerate(retained):
            start, stop = matrix.indptr[i:i+2]
            for j in matrix.indices[start:stop]:
                j = int(j)
                adjacency[j].append(n+k)
                adjacency[n+k].append(j)
        stats = dict(variables=n, source_constraints=len(rows),
            graph_constraints=len(retained), graph_edges=sum(map(len, adjacency))//2,
            absorbed_singleton_rows=len(singleton_log),
            isolated_variables=sum(not a for a in adjacency[:n]),
            inconsistent_effective_bounds=sum(r['inconsistent'] for r in bound_log))
        return (labels, adjacency), stats, singleton_log, bound_log
    finally:
        model.dispose()


def verify_toy():
    """Check the worked A/C example, independently of LP bound normalization."""
    a = (['V:C']*2+['R:I']*2, [[2,3],[2,3],[0,1],[0,1]])
    c = (['V:C']*2+['R:I']*3, [[2,3,4],[2,3],[0,1],[0,1],[0]])
    expected = [10/math.sqrt(8*13), 16/math.sqrt(16*20), 18/math.sqrt(24*27)]
    scores = wl_kernels([a,c], 2)
    for h, value in enumerate(expected):
        assert math.isclose(scores[h][0,1], value, abs_tol=1e-12)
    return dict(self_test(), toy_A_C_scores=expected)


def run(root, manifest_path, output, max_h=3, require_hash_match=True):
    records = read_csv(manifest_path)
    refs = [r for r in records if r['dataset'] == 'reference']
    tests = [r for r in records if r['dataset'] != 'reference']
    records = refs+tests
    if len(refs) != 15 or Counter(r['dataset'] for r in records) != {
            'reference':15, 'large_scale_101':101, 'variants':36}:
        raise ValueError('Expected 15 references, 101 original tests and 36 variants')
    if len({(r['dataset'],r['instance']) for r in records}) != len(records):
        raise ValueError('Duplicate dataset/instance identifiers')
    output.mkdir(parents=True, exist_ok=True)
    graphs, graph_audit, hashes = [], [], {}
    checks = verify_toy()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for r in records:
            path = root/r['lp_path']
            actual = sha256(path)
            if require_hash_match and actual != r['sha256']:
                raise ValueError(f'Input differs from reported experiment: {path}')
            hashes[str(path)] = actual
            graph, stats, singletons, bounds = graph_from_lp(path, env)
            graphs.append(graph)
            graph_audit.append(dict(dataset=r['dataset'], instance=r['instance'],
                category=r['category'], lp_path=str(path), sha256=actual,
                **stats, absorbed_rows=singletons, effective_bounds=bounds))
    kernels = wl_kernels(graphs, max_h)
    all_summary = []
    for h, matrix in kernels.items():
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix),1)
        for dataset in ('large_scale_101','variants'):
            target = output/dataset
            pairwise, nearest = [], []
            for i, r in enumerate(records):
                if r['dataset'] != dataset:
                    continue
                scores = matrix[i,:len(refs)]
                same = [j for j,q in enumerate(refs) if q['category'] == r['category']]
                other = [j for j,q in enumerate(refs) if q['category'] != r['category']]
                if not same or not other:
                    raise ValueError(f'Empty reference candidate pool: {r}')
                best = int(np.argmax(scores))
                same_best = max(same, key=lambda j:scores[j])
                loto_best = max(other, key=lambda j:scores[j])
                assert scores[loto_best] <= scores[best]+1e-12
                pairwise.append(dict(instance=r['instance'], category=r['category'],
                    **{q['instance']:float(scores[j]) for j,q in enumerate(refs)}))
                nearest.append(dict(instance=r['instance'], category=r['category'],
                    nearest_same=refs[same_best]['instance'], similarity_same=float(scores[same_best]),
                    nearest_all=refs[best]['instance'], nearest_all_category=refs[best]['category'],
                    similarity_all=float(scores[best]),
                    nearest_loto=refs[loto_best]['instance'], nearest_loto_category=refs[loto_best]['category'],
                    similarity_loto=float(scores[loto_best]),
                    excluded_references=';'.join(refs[j]['instance'] for j in same),
                    remaining_reference_count=len(other)))
            write_csv(target/f'pairwise_h{h}.csv', pairwise)
            write_csv(target/f'nearest_h{h}.csv', nearest)
            for category in CATEGORIES+['All']:
                group = nearest if category == 'All' else [r for r in nearest if r['category']==category]
                if group:
                    all_summary.append(dict(dataset=dataset, h=h, category=category, instances=len(group),
                        median_same=median(r['similarity_same'] for r in group),
                        mean_same=mean(r['similarity_same'] for r in group),
                        median_all=median(r['similarity_all'] for r in group),
                        median_loto=median(r['similarity_loto'] for r in group),
                        mean_loto=mean(r['similarity_loto'] for r in group)))
    write_csv(output/'category_summary_all_depths.csv', all_summary)
    primary = [r for r in all_summary if r['h']==2]
    if primary:
        write_csv(output/'category_summary_h2.csv', primary)
    audit = dict(method='Typed bipartite cumulative WL subtree kernel; cosine normalization',
        bounds='Singleton rows absorbed into variable bounds; bounds not encoded',
        labels='Variable B/I/C; row equality/inequality; <= and >= share R:I',
        ignored='Objective, numerical coefficients/signs, RHS, bound values, names',
        graph_audit=graph_audit, checks=checks, max_h=max_h,
        gurobi_version=list(gp.gurobi.version()), numpy_version=np.__version__,
        manifest_sha256=sha256(manifest_path), core_sha256=sha256(HERE/'wl_core.py'),
        runner_sha256=sha256(__file__), input_hashes_verified=require_hash_match,
        input_files_unchanged=all(sha256(p)==v for p,v in hashes.items()))
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(primary, indent=2))
    return all_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd(),help='Original lean-llm-opt project root')
    parser.add_argument('--manifest',type=Path,default=HERE/'input_manifest.csv')
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--max-h',type=int,default=3)
    parser.add_argument('--allow-changed-inputs',action='store_true',help='Explicitly allow updated LPs')
    args = parser.parse_args()
    if args.max_h < 0:
        parser.error('--max-h must be nonnegative')
    run(args.root.resolve(),args.manifest,args.out,args.max_h,not args.allow_changed_inputs)
