"""Current 101 test LPs vs current 15 reference LPs: typed bipartite WL.
Run with /usr/bin/python3 recompute_wl_101.py. No optimization or API calls.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import gurobipy as gp
import numpy as np
from wl_bipartite import read_linear_model, self_test, typed_graph, wl_kernels

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/wl_101_current_bipartite_20260926'
CATEGORIES=['NRM','RA','TP','AP','UFLP','Mixture','Others']


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    refs=[]
    for i,r in enumerate(csv_rows(ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv'),1):
        refs.append(dict(dataset='reference',instance=f'Ref{i:02d}',category=r['Type'].strip(),
                         path=str(ROOT/'Ref Data_Large_Scale_LP'/f'{i}.lp')))
    assert len(refs)==15
    tests=[]
    for i,r in enumerate(csv_rows(ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101.csv'),1):
        found=set(re.findall(r'/([A-Za-z]+)_(?:testing|example)/([^/\s]+)/',r['Dataset_address']))
        assert len(found)==1,(i,found)
        category,name=next(iter(found))
        assert category==r['Problem Type'].strip().split()[0]
        if category=='Others':name=name.replace('Others','Other')
        tests.append(dict(dataset='test',instance=f'{category}/{name}',category=category,
                          problem_id=f'OR-{i:03d}',
                          path=str(ROOT/'generated_label_models'/category/name/f'{name}.lp')))
    assert len(tests)==101 and len({x['instance'] for x in tests})==101
    for rec in refs+tests:
        p=Path(rec['path']);assert p.is_file(),p
        rec['sha256']=sha256(p)
    validation = self_test()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        models=[read_linear_model(Path(r['path']),env) for r in refs+tests]
    graphs=[typed_graph(m) for m in models]
    kernels=wl_kernels(graphs,3)
    manifest=[]
    for r,m in zip(refs+tests,models):
        manifest.append({**r,'variables':m['n'],'constraints':m['m'],'nonzeros':m['nnz']})
    fields=list(dict.fromkeys(k for r in manifest for k in r))
    with (OUT/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)
    summaries=[]
    for h in (1,2,3):
        k=kernels[h];assert np.allclose(k,k.T) and np.allclose(np.diag(k),1)
        cross=k[15:,:15]
        with (OUT/f'pairwise_h{h}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['instance']+[r['instance'] for r in refs])
            w.writerows([[r['instance']]+cross[i].tolist() for i,r in enumerate(tests)])
        nearest=[]
        for i,r in enumerate(tests):
            same=[j for j,s in enumerate(refs) if s['category']==r['category']]
            best=int(np.argmax(cross[i]))
            j=max(same,key=lambda j:cross[i,j]) if same else None
            nearest.append(dict(instance=r['instance'],problem_id=r['problem_id'],category=r['category'],
                                nearest_same=refs[j]['instance'] if j is not None else '',
                                similarity_same=float(cross[i,j]) if j is not None else None,
                                nearest_all=refs[best]['instance'],similarity_all=float(cross[i,best])))
        write_csv(OUT/f'nearest_h{h}.csv',nearest)
        for category in CATEGORIES+['All']:
            selected=[r for r in nearest if category=='All' or r['category']==category]
            same=[r['similarity_same'] for r in selected if r['similarity_same'] is not None]
            all_values=[r['similarity_all'] for r in selected]
            summaries.append(dict(h=h,category=category,instances=len(selected),
                same_category_reference_count=len(refs) if category=='All' else sum(r['category']==category for r in refs),
                same_available=len(same),median_same=float(np.median(same)) if same else None,
                mean_same=float(np.mean(same)) if same else None,
                median_all=float(np.median(all_values)),mean_all=float(np.mean(all_values))))
    write_csv(OUT/'category_summary.csv',summaries)
    audit=dict(test_count=101,reference_count=15,primary_h=2,
               construction='variable/constraint bipartite support graph, V:B/I/C and R:E/I',
               normalization='cosine of concatenated label counts from rounds 0..h',presolve=False,
               ignored=['objective','coefficient magnitudes/signs','RHS','bounds','names'],
               reference_mapping='current RAG_Examples_All.csv row i (1-based) -> Ref Data_Large_Scale_LP/i.lp',
               source_csv_sha256=sha256(ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv'),
               implementation_sha256=sha256(ROOT/'wl_bipartite.py'),
               test_lp_files_present=True,
               implementation_module='wl_bipartite.py',
               validation=validation,
               category_counts=dict(Counter(r['category'] for r in tests)),
               note='Current reference categories are taken from the current CSV. Others is heterogeneous; Mixture has 8 references.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps([r for r in summaries if r['h']==2],indent=2))


if __name__=='__main__':main()
