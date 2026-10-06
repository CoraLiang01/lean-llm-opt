"""Check saved observations against physical CSV rows and frozen source hashes."""
import contextlib
import io
import json
from pathlib import Path
from sample_gpt41_extraction_v6 import namespace, BASE, ROOT, OUT, sha

manifest = json.loads((OUT / 'sample_manifest.json').read_text())
with contextlib.redirect_stdout(io.StringIO()): ns = namespace(BASE)
tables = {}
for sample in manifest['samples']:
    tables[(sample['dataset'], sample['case']['problem_id'])] = ns['_load_tables'](sample['case']['dataset_address'])
cells = runs = 0
for path in (OUT / 'runs').glob('*/*/*/*/result.json'):
    row = json.loads(path.read_text()); runs += 1
    observation = (path.parent / 'data_overview.json').read_text()
    data = json.loads(observation); sources = tables[(row['dataset'], row['problem_id'])]
    trace = json.loads((path.parent / 'csvqa_trace.json').read_text())
    import hashlib
    assert hashlib.sha256(observation.encode()).hexdigest() == trace['payload_hash']
    assert trace['planner_repair_enabled'] is False
    assert trace['planner_repair_attempted'] is False
    for table in data['tables']:
        frame = sources[table['file_index']]['frame']
        assert table['original_rows'] == len(frame)
        assert table['returned_rows'] == len(table['records'])
        source_rows = [record['source_row'] for record in table['records']]
        assert source_rows == sorted(source_rows), 'Source rows reordered'
        for record in table['records']:
            for column, value in record['values'].items():
                assert value == str(frame.loc[record['source_row'], column])
                cells += 1
    assert row['source_sha256'] == manifest[f"{'baseline' if row['version']=='baseline' else 'improved'}_sha256"]
assert runs == 36
assert all(sha(Path(p)) == digest for p, digest in manifest['inputs'].items())
assert sha(ROOT / 'LEAN_LLM_OPT_4.1_Large-scale.ipynb') == manifest['baseline_sha256']
assert sha(ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_Extraction_V6.ipynb') == manifest['improved_sha256']
summary = {'status': 'PASS', 'runs': runs, 'checked_source_cells': cells,
           'source_order_preserved': True, 'plan_repair_off': True,
           'input_hashes_unchanged': True, 'main_unchanged': True, 'api_calls': 0}
(OUT / 'artifact_audit.json').write_text(json.dumps(summary, indent=2))
print(summary)
