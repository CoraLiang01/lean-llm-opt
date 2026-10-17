"""Add nearest-reference WL scores after excluding the test's entire category.

Uses existing WL pairwise matrices, with no graph, depth or LP changes.
Only Python's standard library is required.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'outputs/wl_bounds_excluded_20261006')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/wl_leave_one_type_out_20261015')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.input/'normalization_manifest.csv'
    manifest = read_csv(manifest_path)
    refs = {r['instance']: r['category'] for r in manifest if r['dataset'] == 'reference'}
    assert len(refs) == 15
    source_audit = json.loads((args.input/'audit.json').read_text())
    current_inputs = []
    for name, expected in source_audit['source_sha256'].items():
        path = Path(name)
        status = 'unchanged' if path.exists() and digest(path) == expected else 'changed_or_missing'
        current_inputs.append(dict(path=name, status=status))
    summaries, details, inputs = [], [], {str(manifest_path): digest(manifest_path)}
    md = ['# WL similarity with leave-one-type-out reference matching', '',
          'Setting: cumulative rounds 0–2; effective bounds retained as variable restrictions and excluded from the graph.', '',
          'For each test P, exclude every reference with the same category as P, then take the maximum remaining similarity. The reported LOTO column is the median of these per-test maxima, not an accuracy measure.', '']
    latex = []
    for dataset, count in [('large_scale_101', 101), ('variants', 36)]:
        pair_path = args.input/dataset/'pairwise_h2.csv'
        nearest_path = args.input/dataset/'nearest_h2.csv'
        inputs.update({str(p): digest(p) for p in (pair_path, nearest_path)})
        pairs = read_csv(pair_path)
        previous = {r['instance']: r for r in read_csv(nearest_path)}
        assert len(pairs) == len(previous) == count
        rows = []
        for row in pairs:
            assert set(row)-{'instance', 'category'} == set(refs)
            category = row['category']
            scores = {r: float(row[r]) for r in refs}
            assert all(math.isfinite(v) and 0 <= v <= 1+1e-12 for v in scores.values())
            same = [r for r in refs if refs[r] == category]
            other = [r for r in refs if refs[r] != category]
            assert same and other and set(same).isdisjoint(other)
            same_best = max(same, key=scores.get)
            all_best = max(refs, key=scores.get)
            loto_best = max(other, key=scores.get)
            old = previous[row['instance']]
            assert old['category'] == category
            assert math.isclose(scores[same_best], float(old['similarity_same']), abs_tol=1e-12)
            assert math.isclose(scores[all_best], float(old['similarity_all']), abs_tol=1e-12)
            assert scores[loto_best] <= scores[all_best]+1e-12
            rows.append(dict(dataset=dataset, instance=row['instance'], category=category,
                nearest_same=same_best, similarity_same=scores[same_best],
                nearest_all=all_best, similarity_all=scores[all_best],
                excluded_references=';'.join(same), remaining_reference_count=len(other),
                nearest_loto=loto_best, nearest_loto_category=refs[loto_best],
                similarity_loto=scores[loto_best],
                tied_nearest_loto=';'.join(r for r in other if math.isclose(scores[r], scores[loto_best], rel_tol=0, abs_tol=1e-12))))
        details.extend(rows)
        md += [f'## {dataset}', '', '| Category | Tests | Median same | Mean same | Median all | Median LOTO |',
               '|---|---:|---:|---:|---:|---:|']
        latex += [r'\begin{tabular}{lrrrrr}', r'\toprule',
                  r'Category & Tests & Median $S_{\rm same}$ & Mean $S_{\rm same}$ & Median $S_{\rm all}$ & Median $S_{\rm LOTO}$ \\', r'\midrule']
        for category in CATEGORIES+['All']:
            group = rows if category == 'All' else [r for r in rows if r['category'] == category]
            if not group:
                continue
            summary = dict(dataset=dataset, h=2, category=category, instances=len(group),
                median_same=median(r['similarity_same'] for r in group),
                mean_same=mean(r['similarity_same'] for r in group),
                median_all=median(r['similarity_all'] for r in group),
                median_loto=median(r['similarity_loto'] for r in group),
                mean_loto=mean(r['similarity_loto'] for r in group))
            summaries.append(summary)
            cells = [category, str(len(group))]+[f'{summary[k]:.4f}' for k in ['median_same','mean_same','median_all','median_loto']]
            md.append('| '+' | '.join(cells)+' |')
            latex.append(' & '.join(cells)+r' \\')
        md.append('')
        latex += [r'\bottomrule', r'\end{tabular}', '']
    assert len(details) == 137
    write_csv(args.output/'nearest_with_loto_h2.csv', details)
    write_csv(args.output/'category_summary_with_loto_h2.csv', summaries)
    (args.output/'tables.tex').write_text('\n'.join(latex)+'\n')
    statuses = dict(Counter(r['status'] for r in current_inputs))
    md += ['## Input verification', '', str(statuses), '',
           'NRM uses the previously revised reference. Its LOTO score excludes Ref03 entirely. AP variable-domain differences and omitted numerical attributes remain limitations. No high/low similarity threshold is introduced.']
    (args.output/'RESULTS.md').write_text('\n'.join(md)+'\n')
    audit = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_files_sha256=inputs, reference_categories=refs,
        original_input_status=statuses, original_input_checks=current_inputs,
        script_sha256=digest(Path(__file__)), h=2,
        operation='Maximum over references whose category differs from test category; then median across tests.',
        original_results_reproduced=True, loto_never_exceeds_all=True,
        input_files_unchanged=all(digest(Path(p)) == v for p,v in inputs.items()))
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(results=summaries, input_status=statuses, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
