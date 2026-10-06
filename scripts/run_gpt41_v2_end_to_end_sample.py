"""Preflight/run six fixed benchmark cases in all three V2 copies.

Use the project's lean_llm_opt_4_1 Python environment. --run explicitly enables
real model calls. An unreadable input stops preflight before any model request.
"""
import argparse
import contextlib
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpt41_model_interface_v2_validation'
CASE_IDS = ['OR-002', 'OR-004', 'OR-018', 'OR-036', 'OR-063', 'OR-069']


def main(run):
    report, prepared = [], []
    batch_id = 'sample_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in sorted(ROOT.glob('*Large-scale_Model_Interface_V2.ipynb')):
        entry = {'notebook': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'requested_ids': CASE_IDS, 'end_to_end_run': False}
        ns = {'__name__': '__sample_runner__'}
        loto = path.name.startswith('LOTO_')
        try:
            notebook = json.loads(path.read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                for i, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code' and i <= (41 if loto else 35):
                        exec(compile(''.join(cell['source']), f'{path}:cell{i}', 'exec'), ns)
            frame = ns['load_benchmark']()
            ref = ROOT / ns['RAG_EXAMPLES_ALL_PATH']
            content = ref.read_bytes()
            if content.startswith(b'%TSD'):
                raise PermissionError('Reference CSV is protected, not readable CSV text')
            for address in frame['dataset_address']:
                for name in str(address).splitlines():
                    raw = Path(name).read_bytes()
                    if raw.startswith(b'%TSD'):
                        raise PermissionError('A benchmark data CSV is protected')
            ns['_source_fingerprint']()  # Preflight every reference CSV used for cache provenance.
            if loto:
                ns['loto_preflight']()
            indices = frame.index[frame['problem_id'].isin(CASE_IDS)].tolist()
            assert len(indices) == len(CASE_IDS)
            ns['RESULTS_DIR'] = ns['RESULTS_DIR'] / batch_id
            entry.update(preflight='PASS', output_dir=str(ns['RESULTS_DIR']))
            prepared.append((entry, ns, frame, indices, loto))
        except Exception as exc:
            entry.update(preflight='BLOCKED', error=f'{type(exc).__name__}: {exc}')
        report.append(entry)
    if run and len(prepared) == 3:
        for entry, ns, frame, indices, loto in prepared:
            ns['RESULTS_DIR'].mkdir(parents=True, exist_ok=False)
            log_path = ns['RESULTS_DIR'] / 'run.log'
            entry['run_log'] = str(log_path)
            try:
                with log_path.open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    results = ns['run_loto'](rows=indices, reuse_completed=False) if loto else ns['run_test'](
                        frame.loc[indices], output_csv=ns['RESULTS_DIR'] / 'results.csv', reuse_completed=False)
                entry.update(end_to_end_run=True, evaluated=len(results),
                             solved=sum(r['final_ok'] for r in results),
                             matched=sum(r.get('solution_correct') is True for r in results))
            except Exception as exc:
                entry['run_error'] = f'{type(exc).__name__}: {exc}'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'end_to_end_sample_status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if run:
        from compare_gpt41_v2_end_to_end_sample import compare
        compare(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    main(parser.parse_args().run)
