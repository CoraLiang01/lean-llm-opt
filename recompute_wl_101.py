"""Current 101 test LPs vs current 15 reference LPs: typed bipartite WL.
Run with /usr/bin/python3 recompute_wl_101.py. No optimization or API calls.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gurobipy as gp
import numpy as np
from wl_bipartite import read_linear_model, self_test, typed_graph, wl_kernels

ROOT=Path(__file__).resolve().parent
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=ROOT/'Ref_Data_Large_Scale_LP')
    parser.add_argument('--reference-csv',type=Path,default=ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv')
    parser.add_argument('--test-dir',type=Path,default=ROOT/'generated_label_models')
    parser.add_argument('--test-csv',type=Path,default=ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101.csv')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/wl_latest_20261003')
    args=parser.parse_args()
    OUT=args.out.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    refs=[]
    for i,r in enumerate(csv_rows(args.reference_csv),1):
        refs.append(dict(dataset='reference',instance=f'Ref{i:02d}',category=r['Type'].strip(),
                         path=str(args.reference_dir.resolve()/f'{i}.lp')))
    assert len(refs)==15
    tests=[]
    for i,r in enumerate(csv_rows(args.test_csv),1):
        found=set(re.findall(r'/([A-Za-z]+)_(?:testing|example)/([^/\s]+)/',r['Dataset_address']))
        assert len(found)==1,(i,found)
        category,name=next(iter(found))
        assert category==r['Problem Type'].strip().split()[0]
        if category=='Others':name=name.replace('Others','Other')
        tests.append(dict(dataset='test',instance=f'{category}/{name}',category=category,
                          problem_id=f'OR-{i:03d}',
                          path=str(args.test_dir.resolve()/category/name/f'{name}.lp')))
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
        manifest.append({**r,'variables':m['n'],'constraints':m['m'],'nonzeros':m['nnz'],
                         'variable_domains':json.dumps(dict(Counter(m['types'])),sort_keys=True),
                         'row_senses':json.dumps(dict(Counter(m['senses'])),sort_keys=True)})
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
                q25_same=float(np.percentile(same,25)) if same else None,
                q75_same=float(np.percentile(same,75)) if same else None,
                min_same=float(min(same)) if same else None,max_same=float(max(same)) if same else None,
                equal_one_same=int(np.count_nonzero(np.isclose(same,1,atol=1e-12,rtol=0))),
                equal_one_all=int(np.count_nonzero(np.isclose(all_values,1,atol=1e-12,rtol=0))),
                median_all=float(np.median(all_values)),mean_all=float(np.mean(all_values))))
    write_csv(OUT/'category_summary.csv',summaries)
    # Sensitivity analysis: change only variable-domain labels, keeping both
    # graph partitions, row labels and every edge unchanged. Not the primary metric.
    domain_agnostic=[(['V' if label.startswith('V:') else label for label in labels],adj)
                     for labels,adj in graphs]
    neutral_cross=wl_kernels(domain_agnostic,2)[2][15:,:15]
    neutral_rows=[]
    for i,r in enumerate(tests):
        same=[j for j,s in enumerate(refs) if s['category']==r['category']]
        j=max(same,key=lambda j:neutral_cross[i,j]) if same else None
        best=int(np.argmax(neutral_cross[i]))
        neutral_rows.append(dict(instance=r['instance'],category=r['category'],
             nearest_same=refs[j]['instance'] if j is not None else '',
             similarity_same=float(neutral_cross[i,j]) if j is not None else None,
             nearest_all=refs[best]['instance'],similarity_all=float(neutral_cross[i,best])))
    write_csv(OUT/'domain_agnostic_nearest_h2.csv',neutral_rows)
    neutral_summary=[]
    for category in CATEGORIES+['All']:
        selected=[r for r in neutral_rows if category=='All' or r['category']==category]
        same=[r['similarity_same'] for r in selected if r['similarity_same'] is not None]
        neutral_summary.append(dict(category=category,instances=len(selected),
            median_same=float(np.median(same)) if same else None,
            mean_same=float(np.mean(same)) if same else None,
            median_all=float(np.median([r['similarity_all'] for r in selected]))))
    write_csv(OUT/'domain_agnostic_summary_h2.csv',neutral_summary)
    # A reference-side ablation explains the current NRM score. It does not
    # change any LP/CSV on disk and is not a test of downstream performance.
    nrm_diagnostic={'status':'not_applicable'}
    nrm_refs=[i for i,r in enumerate(refs) if r['category']=='NRM']
    nrm_tests=[i for i,r in enumerate(tests) if r['category']=='NRM']
    if len(nrm_refs)==1 and nrm_tests:
        ref_idx=nrm_refs[0]; model=models[ref_idx]
        capacity_rows=[i for i,name in enumerate(model['row_names']) if name=='dispatch_capacity']
        if len(capacity_rows)==1:
            keep=np.arange(model['m'])!=capacity_rows[0]
            reduced_matrix=model['A'][keep,:].tocsr()
            reduced={**model,'A':reduced_matrix,'senses':model['senses'][keep],
                     'm':int(keep.sum()),'nnz':reduced_matrix.nnz}
            nrm_kernel=wl_kernels([typed_graph(reduced)]+[graphs[15+i] for i in nrm_tests],2)[2]
            pairs=nrm_kernel[1:,1:][np.triu_indices(len(nrm_tests),k=1)]
            nrm_diagnostic=dict(status='computed',reference=refs[ref_idx]['instance'],
                removed_row='dispatch_capacity',all_disk_inputs_unchanged=True,
                current_reference_median=float(np.median(kernels[2][[15+i for i in nrm_tests],ref_idx])),
                without_capacity_median=float(np.median(nrm_kernel[1:,0])),
                within_test_pairwise_min=float(pairs.min()),within_test_pairwise_max=float(pairs.max()),
                note='Reference-side in-memory ablation, not evidence of robustness to new test-side constraints.')
            write_csv(OUT/'nrm_reference_ablation_h2.csv',[
                dict(instance=tests[i]['instance'],current_similarity=float(kernels[2][15+i,ref_idx]),
                     reference_without_capacity_similarity=float(nrm_kernel[t+1,0]))
                for t,i in enumerate(nrm_tests)])
    (OUT/'nrm_diagnostic.json').write_text(json.dumps(nrm_diagnostic,indent=2))
    audit=dict(test_count=101,reference_count=15,primary_h=2,
               computed_at_utc=datetime.now(timezone.utc).isoformat(),
               software=dict(gurobi='.'.join(map(str,gp.gurobi.version())),numpy=np.__version__),
               construction='variable/constraint bipartite support graph, V:B/I/C and R:E/I',
               normalization='cosine of concatenated label counts from rounds 0..h',presolve=False,
               ignored=['objective','coefficient magnitudes/signs','RHS','bounds','names'],
               reference_mapping=f'CSV row i (1-based, header excluded) -> {args.reference_dir.resolve()}/i.lp',
               reference_csv_path=str(args.reference_csv.resolve()),
               test_csv_path=str(args.test_csv.resolve()),
               source_csv_sha256=sha256(args.reference_csv),
               test_csv_sha256=sha256(args.test_csv),
               implementation_sha256=sha256(ROOT/'wl_bipartite.py'),
               runner_sha256=sha256(Path(__file__)),
               supplemental_diagnostic='domain-agnostic h=2; only replace V:B/I/C by V',
               nrm_diagnostic=nrm_diagnostic,
               test_lp_files_present=True,
               implementation_module='wl_bipartite.py',
               validation=validation,
               category_counts=dict(Counter(r['category'] for r in tests)),
               note='Current reference categories are taken from the current CSV. Others is heterogeneous; Mixture has 8 references.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps([r for r in summaries if r['h']==2],indent=2))


if __name__=='__main__':main()
