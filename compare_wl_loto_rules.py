"""Combine ordinary and Mixture/Others-grouped LOTO category summaries."""
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parent
OLD = ROOT/'outputs/wl_leave_one_type_out_20261015/category_summary_with_loto_h2.csv'
NEW = ROOT/'outputs/wl_group_loto_20261016/category_summary_group_loto_h2.csv'
OUT = ROOT/'outputs/wl_group_loto_20261016'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    old={(r['dataset'],r['category']):r for r in read(OLD)}
    new={(r['dataset'],r['category']):r for r in read(NEW)}
    rows=[]
    for key, r in old.items():
        q=new[key]
        rows.append(dict(dataset=r['dataset'],h=2,category=r['category'],instances=r['instances'],
            median_same=r['median_same'],mean_same=r['mean_same'],median_all=r['median_all'],
            median_loto_same_type=r['median_loto'],
            median_loto_grouped_mixture_others=q['median_loto_group'],
            change=float(q['median_loto_group'])-float(r['median_loto']),
            grouped_rule=q['exclusion_rule']))
    write(OUT/'category_summary_loto_comparison_h2.csv',rows)
    lines=['# Comparison of LOTO definitions','',
      'For NRM, RA, TP, AP and UFLP, both columns use same-category exclusion. For Mixture and Others, the grouped column jointly removes all Mixture and Others references.', '',
      '| Dataset | Category | Tests | Median same | Mean same | Median all | LOTO same type | LOTO Mixture+Others grouped | Change |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['category']} | {r['instances']} | {float(r['median_same']):.4f} | {float(r['mean_same']):.4f} | {float(r['median_all']):.4f} | {float(r['median_loto_same_type']):.4f} | {float(r['median_loto_grouped_mixture_others']):.4f} | {float(r['change']):+.4f} |")
    (OUT/'LOTO_RULE_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([r for r in rows if r['category'] in ('Mixture','Others')],indent=2))

if __name__=='__main__':main()
