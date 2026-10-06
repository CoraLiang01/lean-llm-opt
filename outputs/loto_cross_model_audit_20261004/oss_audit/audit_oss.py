from pathlib import Path
import csv, json, math, hashlib
from collections import Counter

SRC=Path('/Users/cora/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_yj16fcohm45a22_43e6/msg/file/2026-10/LOTO_outputs')
OUT=Path(__file__).resolve().parent
TYPES=['TP','NRM','RA','FLP','AP','Mixture','Others']
VARIANTS=['examples_only','examples_and_route']
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def yes(x):return str(x).lower()=='true'
def num(x):
    try:return float(x)
    except:return None
def route(x):return 'Others' if x=='Mixture' else x
def normalize(x):return 'FLP' if x=='UFLP' else x
def write_csv(name,rs,fields=None):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rs[0]));w.writeheader();w.writerows(rs)
allrows={}; audit={}; manifests={}; hashes={}
for variant in VARIANTS:
    p=SRC/variant;fold=[];checks=[];mfs={};blocked=[];missing=[]
    report={r['problem_id']:r for r in rows(p/'reports/case_details.csv')}
    for typ in TYPES:
        mf=json.loads((p/typ/'fold_manifest.json').read_text());mfs[typ]=mf
        rs=rows(p/typ/'results.csv');ids=[r['problem_id'] for r in rs]
        fs=rows(p/'fold_reports'/typ/'case_summary.csv')
        fr={r['problem_id']:r for r in fs}
        if sorted(fr)!=sorted(ids):checks.append(['fold_report_ids',typ])
        for raw in rs:
            rr=fr.get(raw['problem_id'],{})
            for key,val in rr.items():
                if val!=raw.get(key,''):checks.append(['fold_report_field',raw['problem_id'],key])
        if sorted(ids)!=sorted(mf['case_ids']):checks.append(['manifest_ids',typ])
        if mf['held_out_type']!=typ:checks.append(['held_out_type',typ])
        if any(normalize(e['type'])!=typ for e in mf['removed_examples']):checks.append(['bad_removal',typ])
        if typ in mf['remaining_counts']:checks.append(['heldout_remaining',typ])
        for r in rs:
            pid=r['problem_id'];ref=num(r['label_objective']);obj=num(r['final_objective'])
            solved=yes(r['final_ok']);matched=bool(solved and obj is not None and ref is not None and math.isclose(obj,ref,rel_tol=1e-4,abs_tol=1e-4))
            d={k:r.get(k,'') for k in ['problem_id','true_label','predicted_label','assigned_route','label_objective','final_objective','pipeline_stage','execution_error_type','execution_error','cache_source','cache_fingerprint','csvqa_mode','csvqa_status','code_repair_count','route_allowed','reference_sha256','reference_count_after']}
            d.update(variant=variant,solved=solved,matched=matched,classification_correct=r['true_label']==r['predicted_label'] if variant=='examples_only' else None,label_equals_true_diagnostic=r['true_label']==r['predicted_label'],disabled_route=mf['disabled_workflow_route'],artifact_dir=str(p/typ/'results_cases/AUTO'/pid))
            sd=p/typ/'results_cases/AUTO'/pid/'solution.json';sj=json.loads(sd.read_text()) if sd.exists() else None
            d.update(solution_status=sj.get('status') if sj else None,solution_error=sj.get('error') if sj else None)
            for fn in ['model.md','solve.py','data_overview.md','solution.json']:
                fp=sd.parent/fn
                d[fn+'_sha256']=hashlib.sha256(fp.read_bytes()).hexdigest() if fp.exists() else None
            fold.append(d)
            if matched!=yes(r['solution_correct']):checks.append(['stored_match',pid])
            if solved!=(sj is not None and sj.get('status')=='OPTIMAL'):checks.append(['solution_status',pid])
            if solved and (num(sj.get('objective'))!=obj):checks.append(['solution_objective',pid])
            if r['true_label']!=typ or r['held_out_type']!=typ:checks.append(['fold_label',pid])
            if r['reference_sha256']!=mf['reference_sha256']:checks.append(['reference_sha',pid])
            if int(r['reference_count_after'])!=mf['reference_count_after']:checks.append(['reference_count',pid])
            if r['assigned_route']!=route(r['predicted_label']):checks.append(['mapping',pid])
            if not r['predicted_label']:missing.append(pid)
            if variant=='examples_and_route' and r['predicted_label'] and (r['assigned_route']==mf['disabled_workflow_route'] or r['predicted_label'] not in mf['allowed_labels']):
                blocked.append(pid)
                if r['pipeline_stage']!='route_selection' or solved or (sd.parent/'solve.py').exists():checks.append(['forbidden_executed_or_unblocked',pid])
            rr=report.get(pid)
            if rr is None:checks.append(['missing_in_report',pid])
            else:
                for key in ['true_label','predicted_label','assigned_route','pipeline_stage']:
                    if rr[key]!=r[key]:checks.append(['report_mismatch',pid,key,rr[key],r[key]])
                if yes(rr['final_ok'])!=solved or yes(rr['solution_correct'])!=matched:checks.append(['report_score',pid])
    manifests[variant]=mfs;allrows[variant]={r['problem_id']:r for r in fold}
    if len(fold)!=101 or len(allrows[variant])!=101 or len(report)!=101:checks.append(['counts',len(fold),len(allrows[variant]),len(report)])
    bytype=[]
    for typ in TYPES:
        rs=[r for r in fold if r['true_label']==typ];n=len(rs);s=sum(r['solved'] for r in rs);m=sum(r['matched'] for r in rs)
        bytype.append(dict(type=typ,cases=n,solved=s,matched=m,failed=n-s,solved_mismatch=s-m,classification_correct=sum(r['classification_correct'] for r in rs) if variant=='examples_only' else None,solve_rate=s/n,objective_accuracy=m/n))
    matrices={}
    for k in ['predicted_label','assigned_route']:
        cols=(TYPES if k=='predicted_label' else [x for x in TYPES if x!='Mixture'])+['Missing']
        matrix=[dict(true_label=t,**{c:sum(r['true_label']==t and (r[k] or 'Missing')==c for r in fold) for c in cols}) for t in TYPES]
        write_csv(f'{variant}_{k}_matrix.csv',matrix);matrices[k]=matrix
    audit[variant]=dict(cases=len(fold),solved=sum(r['solved'] for r in fold),matched=sum(r['matched'] for r in fold),classification_correct=sum(r['classification_correct'] for r in fold) if variant=='examples_only' else None,label_equals_true_diagnostic=sum(r['label_equals_true_diagnostic'] for r in fold),cache_source=dict(Counter(r['cache_source'] for r in fold)),models=list(set(m['model'] for m in mfs.values())),base_sha=list(set(m['base_notebook_sha256'] for m in mfs.values())),reference_sha=list(set(m['reference_sha256'] for m in mfs.values())),integrity_issues=checks,blocked_forbidden_attempts=blocked,missing_classifications=missing,by_type=bytype,matrices=matrices)
    overall=rows(p/'reports/overall.csv')[0]
    for key,actual in [('recorded_cases',len(fold)),('solved',sum(r['solved'] for r in fold)),('objective_correct',sum(r['matched'] for r in fold)),('forbidden_route_selections',len(blocked))]:
        if int(overall[key])!=actual:checks.append(['overall_report',key,overall[key],actual])
    for b in bytype:
        summ={r['metric']:r for r in rows(p/'fold_reports'/b['type']/'summary.csv')}
        for key,actual in [('Solved',b['solved']),('Objective match',b['matched'])]:
            if num(summ[key]['correct'])!=actual or num(summ[key]['total'])!=b['cases']:checks.append(['fold_summary',b['type'],key])
    write_csv(f'{variant}_case_details.csv',fold)
    write_csv(f'{variant}_by_type.csv',bytype)
paired=[]
for pid,a in allrows[VARIANTS[0]].items():
    b=allrows[VARIANTS[1]][pid]
    if a['matched'] and b['matched']:change='both_matched'
    elif a['matched']:change='lost_to_mismatch' if b['solved'] else 'lost_to_failure'
    elif b['matched']:change='gained_from_mismatch' if a['solved'] else 'gained_from_failure'
    else:change='neither_matched'
    paired.append(dict(problem_id=pid,true_label=a['true_label'],change=change,a_route=a['assigned_route'],b_route=b['assigned_route'],a_label=a['predicted_label'],b_label=b['predicted_label'],a_solved=a['solved'],b_solved=b['solved'],a_matched=a['matched'],b_matched=b['matched'],reference=a['label_objective'],a_objective=a['final_objective'],b_objective=b['final_objective'],a_error=a['solution_error'] or a['execution_error'],b_error=b['solution_error'] or b['execution_error'],same_code=a['solve.py_sha256']==b['solve.py_sha256'],same_model=a['model.md_sha256']==b['model.md_sha256'],same_observation=a['data_overview.md_sha256']==b['data_overview.md_sha256'],same_solution=a['solution.json_sha256']==b['solution.json_sha256'],same_cache_fingerprint=a['cache_fingerprint']==b['cache_fingerprint']))
write_csv('paired_cases.csv',paired)
both_solved=[r for r in paired if r['a_solved'] and r['b_solved']]
audit['paired']=dict(change_counts=dict(Counter(r['change'] for r in paired)),common_solved=dict(cases=len(both_solved),a_matched=sum(r['a_matched'] for r in both_solved),b_matched=sum(r['b_matched'] for r in both_solved)),same_artifacts={k:sum(r[k] for r in paired) for k in ['same_code','same_model','same_observation','same_solution','same_cache_fingerprint']},changes=[r for r in paired if r['change'] not in ['both_matched','neither_matched']],failures=[r for r in paired if not r['a_solved'] or not r['b_solved']])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
(OUT/'manifest_summary.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False))
print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='matrices'} if k!='paired' else v) for k,v in audit.items()},indent=2,ensure_ascii=False))
