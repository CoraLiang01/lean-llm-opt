"""Compare this package's outputs against the saved, previously reported run."""
import argparse
import json
from pathlib import Path

from run_wl import read_csv


def verify(root, results):
    original = root/'outputs/wl_bounds_excluded_20261006'
    largest_error, pair_count = 0.0, 0
    for dataset in ('large_scale_101', 'variants'):
        for h in (1,2,3):
            old = {r['instance']:r for r in read_csv(original/dataset/f'pairwise_h{h}.csv')}
            new = {r['instance']:r for r in read_csv(results/dataset/f'pairwise_h{h}.csv')}
            assert old.keys() == new.keys()
            for name, row in old.items():
                assert row['category'] == new[name]['category']
                for key in row.keys()-{'instance','category'}:
                    error = abs(float(row[key])-float(new[name][key]))
                    largest_error = max(largest_error,error)
                    assert error <= 1e-12, (dataset,h,name,key,error)
                    pair_count += 1
    old_stats = {(r['dataset'],r['instance']):r for r in read_csv(original/'normalization_manifest.csv')}
    audit = json.loads((results/'audit.json').read_text())
    for row in audit['graph_audit']:
        previous = old_stats[(row['dataset'],row['instance'])]
        for key in ['variables','source_constraints','graph_constraints','graph_edges',
                    'absorbed_singleton_rows']:
            assert row[key] == int(previous[key]), (row['instance'],key)
        assert row['isolated_variables'] == int(previous['isolated_variable_nodes'])
    old_summary = {(r['dataset'],r['category']):r for r in read_csv(
        root/'outputs/wl_leave_one_type_out_20261015/category_summary_with_loto_h2.csv')}
    summary_error = 0.0
    summary_count = 0
    for row in read_csv(results/'category_summary_h2.csv'):
        previous = old_summary[(row['dataset'],row['category'])]
        assert row['instances'] == previous['instances']
        for key in ['median_same','mean_same','median_all','median_loto','mean_loto']:
            error = abs(float(row[key])-float(previous[key]))
            assert error <= 1e-12
            summary_error = max(summary_error,error)
            summary_count += 1
    report = dict(passed=True, graph_records_checked=len(old_stats),
        pairwise_values_checked=pair_count, max_pairwise_error=largest_error,
        h2_summary_values_checked=summary_count, max_summary_error=summary_error,
        input_files_unchanged=audit['input_files_unchanged'])
    (results/'reproduction_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    verify(args.root,args.results)
