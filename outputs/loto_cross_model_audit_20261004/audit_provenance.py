"""Read-only source/manifest audit; never invokes a model or generated solver code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OSS = Path('/Users/cora/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_yj16fcohm45a22_43e6/msg/file/2026-10/LOTO_outputs')
TYPES = ['TP', 'NRM', 'RA', 'FLP', 'AP', 'Mixture', 'Others']
SPECS = [
    ('gpt41', 'examples_only', ROOT / 'outputs/leave_one_type_out/gpt41/examples_only_V1', 'LOTO_Examples_Only_GPT4.1_Large-scale.ipynb'),
    ('gpt41', 'examples_and_route', ROOT / 'outputs/leave_one_type_out/gpt41/examples_and_route_V1', 'LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb'),
    ('oss20b', 'examples_only', OSS / 'examples_only', 'LOTO_Examples_Only_gpt_oss_20b_Large-scale.ipynb'),
    ('oss20b', 'examples_and_route', OSS / 'examples_and_route', 'LOTO_Examples_And_Route_gpt_oss_20b_Large-scale.ipynb'),
]


def main():
    experiments = {}
    common_reference = {}
    shared_cells = {37: [], 41: []}
    for model, variant, directory, notebook_name in SPECS:
        notebook_path = ROOT / notebook_name
        notebook = json.loads(notebook_path.read_text())
        code_cells = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, f'{notebook_name}:cell{i}', 'exec')
                code_cells.append(source)
        for i in shared_cells:
            shared_cells[i].append(''.join(notebook['cells'][i]['source']))
        manifests = []
        all_ids = []
        for kind in TYPES:
            path = directory / kind / 'fold_manifest.json'
            manifest = json.loads(path.read_text())
            assert manifest['held_out_type'] == kind
            assert manifest['experiment'] == variant
            assert manifest['rel_tol'] == manifest['abs_tol'] == 1e-4
            route = 'Others' if kind in ('Mixture', 'Others') else kind
            disabled = route if variant == 'examples_and_route' else None
            assert manifest['disabled_workflow_route'] == disabled
            allowed = [label for label in TYPES if ('Others' if label in ('Mixture', 'Others') else label) != disabled]
            assert manifest['allowed_labels'] == allowed
            comparison = {key: manifest[key] for key in ('case_ids', 'removed_examples', 'remaining_example_indices', 'remaining_counts', 'reference_count_before', 'reference_count_after', 'reference_sha256')}
            if kind in common_reference:
                assert comparison == common_reference[kind], (model, variant, kind)
            else:
                common_reference[kind] = comparison
            all_ids.extend(manifest['case_ids'])
            manifests.append({key: manifest[key] for key in ('held_out_type', 'model', 'case_ids', 'reference_sha256', 'source_fingerprint', 'base_notebook_sha256', 'disabled_workflow_route', 'allowed_labels')})
        assert len(all_ids) == len(set(all_ids)) == 101
        assert set(all_ids) == {f'OR-{i:03d}' for i in range(1, 102)}
        assert len({m['source_fingerprint'] for m in manifests}) == 1
        experiments[f'{model}/{variant}'] = {
            'notebook_path': str(notebook_path), 'results_path': str(directory),
            'notebook_file_sha256_current': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
            'code_cells_sha256_current': hashlib.sha256('\n'.join(code_cells).encode()).hexdigest(),
            'cell_count': len(notebook['cells']), 'compiled_code_cell_count': len(code_cells),
            'unique_notebook_cell_ids': len({c.get('id') for c in notebook['cells']}) == len(notebook['cells']),
            'manifest_case_count': len(all_ids), 'manifests': manifests,
        }
    assert all(len(set(sources)) == 1 for sources in shared_cells.values())
    result = {
        'experiments': experiments,
        'checks': {
            'all_code_cells_compile': True,
            'each_manifest_set_has_101_unique_ids': True,
            'same_fold_cases_and_reference_exclusions_across_all_four': True,
            'same_reference_sha256_across_all_four': True,
            'same_score_tolerances': True,
            'expected_allowed_labels_and_disabled_routes': True,
            'common_control_and_runner_cells_37_41_identical': True,
        },
        'limitations': [
            'Current notebook syntax and recorded manifests were checked; no inference or generated solver code was executed.',
            'Recorded runtime source fingerprints are not independently reproduced: original GPT CSV files are currently protected, and OSS run-site files/environment were not supplied.',
            'A source fingerprint is not a source snapshot or model-weight digest; matching model names alone does not establish identical deployments.',
        ],
    }
    target = OUT / 'provenance_audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['checks'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
