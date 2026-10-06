"""Compare Variants1--36 to the current 15-reference pool without presolve.
Run: /usr/bin/python3 outputs/wl_variants_20261004/recompute_variants_wl.py
Unknown variant categories are not inferred from the nearest reference.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gurobipy as gp
import numpy as np
from wl_bipartite import read_linear_model, typed_graph, wl_kernels, self_test


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    source = ROOT / 'Large_Scale_Or_Files/RAG_Examples_All.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    refs = [dict(instance=f'Ref{i:02d}', dataset='reference', category=r['Type'].strip(),
                 path=str(ROOT / 'Ref_Data_Large_Scale_LP' / f'{i}.lp'))
            for i, r in enumerate(rows, 1)]
    tests = [dict(instance=f'Variants{i}', dataset='variant', category='',
                  path=str(ROOT / 'Variants_1_36_lp' / f'Variants{i}.lp')) for i in range(1, 37)]
    assert len(refs) == 15
    assert {Path(r['path']).name for r in tests} == {p.name for p in (ROOT / 'Variants_1_36_lp').glob('*.lp')}
    for r in refs + tests:
        r['sha256'] = digest(r['path'])
    validation = self_test()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        models = [read_linear_model(r['path'], env) for r in refs + tests]
    for r, model in zip(refs + tests, models):
        r.update(variables=model['n'], constraints=model['m'], nonzeros=int(model['nnz']),
                 variable_domains=json.dumps(dict(Counter(model['types']))),
                 row_senses=json.dumps(dict(Counter(model['senses']))))
    write_csv('manifest.csv', refs + tests)
    kernels = wl_kernels([typed_graph(m) for m in models], 3)
    summaries = []
    categories = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']
    for h in (1, 2, 3):
        k = kernels[h]
        assert np.allclose(k, k.T) and np.allclose(np.diag(k), 1)
        cross = k[len(refs):, :len(refs)]
        assert cross.shape == (36, 15)
        write_csv(f'pairwise_h{h}.csv', [dict(instance=t['instance'], **{
            r['instance']: float(cross[i, j]) for j, r in enumerate(refs)}) for i, t in enumerate(tests)])
        nearest = []
        for i, t in enumerate(tests):
            best = int(np.argmax(cross[i]))
            ties = np.flatnonzero(np.isclose(cross[i], cross[i, best], atol=1e-12, rtol=0))
            rec = dict(instance=t['instance'], nearest_reference=refs[best]['instance'],
                       nearest_reference_category=refs[best]['category'],
                       similarity_all=float(cross[i, best]),
                       tied_references=';'.join(refs[j]['instance'] for j in ties))
            for category in categories:
                js = [j for j, r in enumerate(refs) if r['category'] == category]
                j = max(js, key=lambda j: cross[i, j])
                rec[f'best_{category}'] = refs[j]['instance']
                rec[f'similarity_to_{category}'] = float(cross[i, j])
            nearest.append(rec)
        write_csv(f'nearest_h{h}.csv', nearest)
        for pool in ['All'] + categories:
            key = 'similarity_all' if pool == 'All' else f'similarity_to_{pool}'
            v = np.asarray([r[key] for r in nearest])
            summaries.append(dict(h=h, reference_pool=pool, variant_count=len(tests),
                reference_count=len(refs) if pool == 'All' else sum(r['category'] == pool for r in refs),
                median=float(np.median(v)), mean=float(v.mean()),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                minimum=float(v.min()), maximum=float(v.max()),
                equal_one=int(np.count_nonzero(np.isclose(v, 1, atol=1e-12, rtol=0)))))
    write_csv('reference_pool_summary.csv', summaries)
    old = json.loads((ROOT / 'outputs/wl_latest_20261003/audit.json').read_text())
    with (ROOT / 'outputs/wl_latest_20261003/manifest.csv').open(encoding='utf-8-sig') as f:
        old_refs = {r['instance']: r['sha256'] for r in csv.DictReader(f) if r['dataset'] == 'reference'}
    audit = dict(primary_h=2, variants=36, references=15, presolve=False,
        representation='typed variable-constraint bipartite graph: V:B/I/C and R:E/I',
        normalization='cosine on concatenated WL label counts from rounds 0..h',
        ignored=['objective', 'nonzero coefficient magnitudes and signs', 'RHS', 'bounds', 'names'],
        source_csv=str(source), source_csv_sha256=digest(source),
        reference_mapping='CSV data row i -> Ref_Data_Large_Scale_LP/i.lp',
        reference_csv_unchanged_since_20261003=digest(source) == old['source_csv_sha256'],
        reference_lps_unchanged_since_20261003=all(r['sha256'] == old_refs[r['instance']] for r in refs),
        variant_categories='unknown; same-category results are not computed',
        pool_summary_interpretation='All 36 variants matched against each reference-category pool; these are NOT summaries of variants in that category.',
        implementation_sha256=digest(ROOT / 'wl_bipartite.py'), runner_sha256=digest(__file__),
        gurobi_version=list(gp.gurobi.version()), validation=validation)
    assert all(digest(r['path']) == r['sha256'] for r in refs + tests)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps([r for r in summaries if r['h'] == 2], indent=2))


if __name__ == '__main__':
    main()
