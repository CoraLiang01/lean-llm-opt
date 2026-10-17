"""Compute a grouped leave-one-type-out score for Mixture and Others.

For Mixture and Others tests, all Mixture and Others references are removed
together. For every other category, the ordinary same-category exclusion is
retained. Existing pairwise WL matrices are reused; LP files are unchanged.
"""
from pathlib import Path
import csv, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'outputs/wl_bounds_excluded_20261006'
OUTPUT = ROOT / 'outputs/wl_group_loto_20261016'
GROUP = {'Mixture', 'Others'}
CATEGORIES = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(INPUT/'normalization_manifest.csv')
    refs = {r['instance']: r['category'] for r in manifest if r['dataset'] == 'reference'}
    assert len(refs) == 15
    details, summaries = [], []
    input_hashes = {}
    for dataset, expected in [('large_scale_101', 101), ('variants', 36)]:
        path = INPUT/dataset/'pairwise_h2.csv'; input_hashes[str(path)] = digest(path)
        pairs = read_csv(path); assert len(pairs) == expected
        rows = []
        for row in pairs:
            category = row['category']; scores = {r: float(row[r]) for r in refs}
            if category in GROUP:
                excluded = [r for r in refs if refs[r] in GROUP]
                rule = 'Mixture+Others jointly excluded'
            else:
                excluded = [r for r in refs if refs[r] == category]
                rule = 'same category excluded'
            candidates = [r for r in refs if r not in excluded]
            assert candidates
            best = max(candidates, key=scores.get)
            rows.append(dict(dataset=dataset, instance=row['instance'], category=category,
                exclusion_rule=rule, excluded_references=';'.join(excluded),
                remaining_reference_count=len(candidates), nearest_group_loto=best,
                nearest_group_loto_category=refs[best], similarity_group_loto=scores[best]))
        details.extend(rows)
        for category in CATEGORIES + ['All']:
            group = rows if category == 'All' else [r for r in rows if r['category'] == category]
            if not group: continue
            summaries.append(dict(dataset=dataset, h=2, category=category, instances=len(group),
                exclusion_rule=('Mixture+Others jointly excluded for Mixture/Others; same category otherwise' if category == 'All' else
                                'Mixture+Others jointly excluded' if category in GROUP else 'same category excluded'),
                median_loto_group=median(r['similarity_group_loto'] for r in group),
                mean_loto_group=mean(r['similarity_group_loto'] for r in group)))
    write_csv(OUTPUT/'nearest_group_loto_h2.csv', details)
    write_csv(OUTPUT/'category_summary_group_loto_h2.csv', summaries)
    lines = ['# Grouped LOTO WL results', '',
        'Primary setting: h=2, effective bounds retained as variable restrictions and excluded from the graph.',
        'For Mixture and Others tests, all 8 Mixture and 2 Others references are removed jointly. For NRM, RA, TP, AP and UFLP, only the test category references are removed.', '']
    for dataset in ['large_scale_101','variants']:
        lines += [f'## {dataset}', '', '| Category | Tests | Exclusion rule | Median grouped LOTO | Mean grouped LOTO |', '|---|---:|---|---:|---:|']
        for r in summaries:
            if r['dataset'] == dataset:
                lines.append(f"| {r['category']} | {r['instances']} | {r['exclusion_rule']} | {r['median_loto_group']:.4f} | {r['mean_loto_group']:.4f} |")
        lines.append('')
    (OUTPUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), h=2,
        group=sorted(GROUP), reference_categories=refs, input_hashes=input_hashes,
        input_files_unchanged=all(digest(p)==h for p,h in input_hashes.items()),
        operation='For Mixture/Others jointly exclude all references in the union; otherwise exclude same category.',
        source='Existing pairwise_h2.csv; no LP files modified')
    (OUTPUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__': main()
