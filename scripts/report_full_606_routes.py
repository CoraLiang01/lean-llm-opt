"""Preserve raw attempts and normalize route metadata for forced-route reporting."""
import argparse
import collections
import csv
import json
from pathlib import Path
import evaluate_0927_optimization as engine
ROOT=engine.ROOT
BASE=ROOT/'outputs/optimization_0927_20261006/full_review_v6_completed'
OUT=ROOT/'outputs/optimization_0927_20261006/full_606_routes_v6_r1'
ROUTES=['TP','NRM','RA','FLP','AP','Others']
CLASSES=['TP','NRM','RA','FLP','AP','Mixture','Others']

def inventory():
    rows=[]
    with (BASE/'automatic/results.csv').open(encoding='utf-8-sig') as f:automatic=[r for r in csv.DictReader(f) if r['benchmark']=='main']
    for a in automatic:
        origin=BASE/'attempts'/a['problem_id']/'result.json';r=json.loads(origin.read_text())
        r.update(forced_route=r['assigned_route'],experiment_mode='forced',route_result_source='reused_full_automatic',reused=True,
                 reuse_source_result=str(origin),reuse_source_seconds=r.get('seconds'),seconds=0,cache_source='reused_same_full_route')
        rows.append(r)
    for p in OUT.glob('attempts/*/*/result.json'):
        r=json.loads(p.read_text());route=p.parent.parent.name
        if route not in ROUTES:raise ValueError('Unknown scheduled route')
        if r.get('forced_route')!=route:r['original_recorded_forced_route']=r.get('forced_route')
        r.update(forced_route=route,assigned_route=route,experiment_mode='forced',route_result_source='new_forced_execution',reused=False,raw_attempt_result=str(p))
        rows.append(r)
    for r in rows:
        if r['reused']:
            r['reuse_original_predicted_label'] = r.get('predicted_label')
            r['reuse_original_classification_correct'] = r.get('classification_correct')
        r['predicted_label'] = None
        r['classification_correct'] = None
        r['classification_metric_applicable'] = False
    if len({(r['problem_id'],r['forced_route']) for r in rows})!=len(rows):raise ValueError('Duplicate route/case')
    return rows

def summary(rows):
    return {'expected':606,'recorded':len(rows),'reused':sum(r['reused'] for r in rows),'newly_executed':sum(not r['reused'] for r in rows),
            'objective_match':sum(r.get('solution_correct') is True for r in rows),'solved':sum(r.get('final_ok') is True for r in rows),
            'quota_errors':sum(engine.credit_exhausted(r) for r in rows),
            'by_route':{route:{'expected':101,'recorded':len(g),'objective_match':sum(r.get('solution_correct') is True for r in g),'solved':sum(r.get('final_ok') is True for r in g),'errors':sum(r.get('final_ok') is not True for r in g)}
                        for route in ROUTES for g in [[r for r in rows if r['forced_route']==route]]}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');a=p.parse_args()
    rows=inventory();data=summary(rows)
    (OUT/'progress_normalized.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
    if not a.finalize:return
    if len(rows)!=606 or data['quota_errors']:raise ValueError('No valid complete 606-case round')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    if engine.sha(OUT/'frozen_notebook.ipynb')!=manifest['notebook_sha256']:raise ValueError('Frozen source changed')
    for path,value in manifest['input_sha256'].items():
        if engine.sha(path)!=value:raise ValueError('Inputs changed during run: '+path)
    base_ids={r['problem_id'] for r in rows if r['reused']}
    if {(r['problem_id'],r['forced_route']) for r in rows}!={(pid,route) for pid in base_ids for route in ROUTES}:raise ValueError('Incomplete Cartesian product')
    if any(r.get('notebook_source_sha256')!=manifest['notebook_sha256'] for r in rows):raise ValueError('Record source mismatch')
    ns=engine.namespace(OUT/'frozen_notebook.ipynb');destination=OUT/'forced/results.csv'
    archive=OUT/'forced/results_before_route_metadata_normalization.csv'
    if archive.exists():raise ValueError('Normalization already finalized')
    if destination.exists():destination.rename(archive)
    for r in rows:ns['save_record'](destination,r)
    report=OUT/'report';report.mkdir(exist_ok=True)
    group_rows=[]
    for route in ROUTES:
        for c in CLASSES:
            g=[r for r in rows if r['forced_route']==route and r['true_label']==c]
            group_rows.append({'route':route,'true_label':c,'total':len(g),'objective_match':sum(r.get('solution_correct') is True for r in g),'solved':sum(r.get('final_ok') is True for r in g)})
    def write(path,records,fields):
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    write(report/'per_route_per_class.csv',group_rows,list(group_rows[0]))
    route_summary = [{'route':r,**s} for r,s in data['by_route'].items()]
    write(report/'per_route_summary.csv',route_summary,['route','expected','recorded','objective_match','solved','errors'])
    old_summary=OUT/'forced/summary.csv'
    if old_summary.exists():old_summary.rename(OUT/'forced/summary_before_route_metadata_normalization.csv')
    write(old_summary,route_summary,['route','expected','recorded','objective_match','solved','errors'])
    fields=['problem_id','true_label','forced_route','final_ok','solution_correct','final_objective','label_objective','execution_error_type','execution_error','pipeline_stage','seconds','route_result_source','reuse_source_result','raw_attempt_result']
    write(report/'all_cases.csv',rows,fields);write(report/'unsuccessful_cases.csv',[r for r in rows if r.get('solution_correct') is not True],fields)
    for flag,name in [('solution_correct','objective_match_matrix.csv'),('final_ok','solved_matrix.csv')]:
        matrix=[]
        for pid in sorted({r['problem_id'] for r in rows}):
            g={r['forced_route']:r for r in rows if r['problem_id']==pid}
            matrix.append({'problem_id':pid,'true_label':g[ROUTES[0]]['true_label'],**{route:g[route].get(flag) is True for route in ROUTES}})
        write(report/name,matrix,['problem_id','true_label']+ROUTES)
    lines=['# 全量 606 route 结果','', '101 题 × 六个 route。复用同一全量代码自动分类所选 route 的 101 个结果，包括四个目标不匹配；新增运行 505 个组合。没有按结果选择重跑。','', '| Route | Objective match | Solved | 失败 |','|---|---:|---:|---:|']
    for route,s in data['by_route'].items():lines.append(f'| {route} | {s["objective_match"]}/101 | {s["solved"]}/101 | {s["errors"]} |')
    for flag,title in [('objective_match','Objective match'),('solved','Solved')]:
        lines+=['',f'## 按真实类别：{title}','', '| 类别 | 题数 | '+' | '.join(ROUTES)+' |','|---|---:|'+('---:|'*6)]
        for c in CLASSES:
            groups=[next(r for r in group_rows if r['route']==route and r['true_label']==c) for route in ROUTES];n=groups[0]['total']
            lines.append(f'| {c} | {n} | '+' | '.join(f'{r[flag]}/{n}' for r in groups)+' |')
    lines+=['','## 口径与路径','', '模型、示例、数据访问、Formulation、代码生成、求解与匹配容差均使用冻结全量代码；强制 route 组合跳过分类。真实类别只用于事后分组，参考答案只用于评分。','', '错误记录中的 route 元数据依据任务目录修正，原始 result.json 未改；results_before_route_metadata_normalization.csv 保留原汇总。分类准确率不适用于强制 route 实验。','',f'完整逐题记录：{destination}',f'日志/请求/输出：{OUT}/attempts/<route>/<题号>/',f'复用来源：{BASE}/attempts/<题号>/result.json',f'配置与来源：{OUT}/run_manifest.json',f'类别统计：{report}/per_route_per_class.csv',f'101×6 匹配矩阵：{report}/objective_match_matrix.csv',f'全部失败和不匹配：{report}/unsuccessful_cases.csv','']
    (report/'report.md').write_text('\n'.join(lines))
    (OUT/'run_status.json').write_text(json.dumps({'state':'completed','normalized_route_metadata':True,'progress':data},ensure_ascii=False,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
