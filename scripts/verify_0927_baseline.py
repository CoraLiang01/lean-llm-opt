"""Verify the given full notebook's code/data against saved historical case fingerprints."""
import hashlib
import json
from pathlib import Path
from check_0927_experiments import ROOT,namespace

# Once the user-authorized revision is delivered, verify the exact archived original.
archived=ROOT/'outputs/optimization_0927_20261006/source_snapshots/before/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'
p=archived if archived.exists() else ROOT/'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'
book=json.loads(p.read_text());ns=namespace(p)
source='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'experiment' not in c.get('metadata',{}).get('tags',[]))
historical_path=ROOT/'LEAN_LLM_OPT_4.1_Large-scale.ipynb'
digest=hashlib.sha256((ns['CACHE_SCHEMA_VERSION']+'\n'+str(historical_path)+'\n'+source).encode())
for path in sorted((ROOT/'Large_Scale_Or_Files').rglob('*.csv')):
    digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
source_fp=digest.hexdigest()
results=ROOT/'outputs/final_101_V2/automatic/results.csv'
records={r['problem_id']:r for r in ns['load_records'](results)}
checks=[]
for case in ns['load_benchmark']().to_dict('records'):
    reconstructed=ns['case_fingerprint'](case,source_fingerprint=source_fp)
    saved=records[case['problem_id']]['cache_fingerprint']
    gold_match=(case['true_label']==records[case['problem_id']]['true_label'] and
                case['true_route']==records[case['problem_id']]['true_route'] and
                case['label_objective']==records[case['problem_id']]['label_objective'])
    assert gold_match,case['problem_id']
    checks.append({'problem_id':case['problem_id'],'matches':reconstructed==saved,'gold_labels_and_objective_match':gold_match,
                   'reconstructed':reconstructed,'saved':saved})
result={'provided_full_notebook':str(p),'provided_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'historical_filename':str(historical_path),'historical_source_fingerprint':source_fp,
        'baseline_results':str(results),'baseline_results_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
        'matched_cases':sum(r['matches'] for r in checks),'matched_gold_labels_objectives':sum(r['gold_labels_and_objective_match'] for r in checks),'total_cases':len(checks),
        'explanation':'Historical fingerprint excludes notebook outputs and experiment switches. It includes definition-cell code, model/embedding settings, all reference CSVs, query and current-case source CSV bytes.',
        'checks':checks}
assert result['matched_cases']==result['total_cases']==101
out=ROOT/'outputs/ablation_loto_0927/baseline_provenance.json';out.write_text(json.dumps(result,indent=2))
print(f"Baseline code/data fingerprint matches: {result['matched_cases']}/{result['total_cases']}")
