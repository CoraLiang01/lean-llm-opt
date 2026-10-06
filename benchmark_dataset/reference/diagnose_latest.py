"""Replay saved agent code locally. Diagnostic copies only; no API calls."""
import ast
import contextlib
import io
import json
import traceback
import warnings
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmark_dataset'
SOURCE=Path('/Users/xiaomao/Desktop/LEAN/lean-llm-opt-老版本/outputs/Large-scale-or-101_冗余列35个 copy.csv/automatic/results_cases/AUTO')
OLDROOT=str(SOURCE).split('/outputs/')[0]

def corrected_helper(scope,value,day,name):
    # All tables share precisely the same record-selection rule.
    return f'''def {name}(df, *args, **kwargs):
    df = df[(df[{scope!r}] == {value!r}) & (df['effective_date'] <= {day!r})].copy()
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values('revision', ascending=False).drop_duplicates(['table', 'record_id'])
    return df[df['action'].str.upper() != 'DELETE'].copy()
'''

def run(source, filename):
    ns={'__name__':'__diagnostic__'}
    with contextlib.redirect_stdout(io.StringIO()) as output, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            exec(compile(source,filename,'exec'),ns)
            m=ns['m']
            result={'status':m.Status,'objective':round(m.ObjVal) if m.SolCount else None}
            if m.Status == 3:
                m.computeIIS()
                result['iis_constraints']=[c.ConstrName for c in m.getConstrs() if c.IISConstr]
        except Exception as e:
            result={'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()}
    return result,output.getvalue()

def main():
    out=BASE/'reference/latest_agent_diagnostics'
    out.mkdir(exist_ok=True)
    manifest={r['case']:r for r in json.loads((BASE/'case_manifest.json').read_text())}
    report=[]
    for num,name in [(4,'RA12'),(7,'RA3'),(9,'RA7'),(10,'RA10')]:
        src=(SOURCE/f'OR-{num:03d}'/'solve.py').read_text().replace(OLDROOT,str(ROOT))
        historical=ROOT/'benchmark_archive/before_targeted_agent_tuning/RA7'
        if name=='RA7' and historical.exists():
            src=src.replace(str(BASE/'RA7/inputs'),str(historical/'inputs'))
        before,log=run(src,f'{name}_original.py')
        (out/f'{name}_original.log').write_text(log)
        tree=ast.parse(src)
        cfg=manifest[name]
        helper=lambda n:ast.parse(corrected_helper(cfg['scope_column'],cfg['scope_value'],cfg['asof'],n)).body[0]
        if name=='RA12':
            tree.body=[helper(n.name) if isinstance(n,ast.FunctionDef) and n.name in ['select_latest_records','select_store_records'] else n for n in tree.body]
        elif name=='RA3':
            class RemoveInvalid(ast.NodeTransformer):
                def visit_Expr(self,node):
                    if 'y_x_link_lb_' in ast.unparse(node): return None
                    return self.generic_visit(node)
            tree=RemoveInvalid().visit(tree)
        elif name=='RA7':
            new=[helper('canonical_records')]
            for node in tree.body:
                new.append(node)
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='pd.concat':
                    target=ast.unparse(node.targets[0])
                    if target in ['usage_df','cap_df','benefit_df','fx_df']:
                        new.extend(ast.parse(f'{target} = canonical_records({target})').body)
            tree.body=new
        else:
            tree.body=[helper(n.name) if isinstance(n,ast.FunctionDef) and n.name=='select_latest_valid' else n for n in tree.body]
        ast.fix_missing_locations(tree)
        corrected=ast.unparse(tree)
        (out/f'{name}_diagnostic_fix.py').write_text(corrected)
        after,log=run(corrected,f'{name}_diagnostic_fix.py')
        (out/f'{name}_corrected.log').write_text(log)
        row={'case':name,'before':before,'after':after}
        if name=='RA10':
            # A bundle involving an unauthorized option must not earn a bonus.
            fixed=corrected.replace("obj_bundle_bonus = gp.LinExpr()", "for (a, b), var in bvar.items():\n    if not itemref_to_keys.get(a) or not itemref_to_keys.get(b):\n        m.addConstr(var == 0)\nobj_bundle_bonus = gp.LinExpr()")
            fixed=fixed.replace('>= minq * z[c]', '>= minq')
            final,final_log=run(fixed,'RA10_model_fix.py')
            row['after_bundle_and_category_fix']=final
            (out/'RA10_model_fix.py').write_text(fixed)
            (out/'RA10_model_fix.log').write_text(final_log)
        report.append(row)
        print(json.dumps(row),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
