#!/usr/bin/env python3
"""CA package offline verification; optional reference solving is NOT LLM accuracy."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Test_Dataset/Air_NRM/large_scale"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solve-reference', action='store_true', help='requires Gurobi licence; not model inference')
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = json.loads((ROOT / 'package_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'package missing/changed file: {name}; extract a fresh package')
    print(f"OK: {len(manifest['sha256'])} packaged file hashes")
    pins = dict(re.findall(r'^([\w-]+)==([^\s#]+)', (ROOT / 'requirements-server.txt').read_text(), re.M))
    for package in ('langchain', 'langchain-core', 'langchain-classic', 'langchain-community',
                    'langchain-openai', 'langchain-ollama', 'ollama'):
        if importlib.metadata.version(package) != pins[package]:
            raise ValueError(f'{package} does not match requirements-server.txt')

    import pandas as pd
    import leanopt_exp as lx
    from langchain_core.embeddings import Embeddings
    from run_notebook import split_at_first_call
    from ca_lean_workflow import AirNrmCAData, parse_ca_query, example_answer, validate_complete_generated_answer
    queries = pd.read_csv(DATA / 'query_CA.csv')
    if len(queries) != 15 or list(queries) != ['n_variables', 'Query', 'Label-objective']:
        raise ValueError('wrong CA query dataset')
    data = AirNrmCAData.from_path(DATA)
    for _, row in queries.iterrows():
        parameters = data.assemble(row['Query'])
        parsed = parse_ca_query(row['Query'])
        if int(row['n_variables']) != 4 * len(parsed.flights) + len(parameters['avg_pax']):
            raise ValueError('CA variable count mismatch')
        validate_complete_generated_answer(example_answer(parameters), parameters)
    print('OK: 15 CA queries, five-file assembly, four consumptions and 112–421 variables')

    class OfflineEmbeddings(Embeddings):
        def embed_query(self, text):
            return [(b - 128) / 128 for b in hashlib.sha256(text.encode()).digest()]
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

    nb = json.loads((DATA / 'LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_CA_LargeScale.ipynb').read_text())
    setup, stop = split_at_first_call(nb)
    if stop is None:
        raise ValueError('CA batch execution cell not found')
    ns = {'__name__': '__main__'}
    with tempfile.TemporaryDirectory(prefix='ca_package_verify_') as temporary:
        with patch.dict(os.environ, {'LEANOPT_OUT_DIR': temporary}), \
             patch.object(lx, 'build_embeddings', return_value=OfflineEmbeddings()), \
             patch('socket.socket.connect', side_effect=RuntimeError('offline check attempted network')):
            for idx, source in setup:
                if not source.strip().startswith(('!', '%')):
                    exec(compile(source, f'<CA offline cell {idx}>', 'exec'), ns)
            cfg = ns['CFG']
            if cfg.dataset != 'Air-NRM-CA-LS' or cfg.model_profile != 'gpt-oss-20b':
                raise ValueError('wrong CA notebook defaults')
            for role, spec in cfg.models.items():
                if spec.provider != 'ollama' or spec.model != 'gpt-oss:20b':
                    raise ValueError('hosted model found in CA roles')
                lx.build_llm(cfg, role)
            ns['Classification_Agent']()
            ns['CA_Agent'](queries.iloc[0]['Query'])
            ns['Problemtype'] = lambda query: 'Sales‑Based Linear Programming (SBLP).'
            ns['ProcessCA'] = lambda query: example_answer(ns['CA_DATA'].assemble(query))
            ns['Batch_Process_Queries'](queries)
            records = ns['LOG'].records
            if len(records) != 15 or any(r.status != 'ok' for r in records):
                raise ValueError('simulated CA batch failed')
            bad = 'Agent stopped due to iteration limit or time limit.'
            ns['ProcessCA'] = lambda query: bad
            bad_row = queries.head(1).copy()
            bad_row.index = [99]
            ns['Batch_Process_Queries'](bad_row)
            record = records[-1]
            if record.status != 'failed' or record.extra.get('model_output') != bad:
                raise ValueError('failed raw model output was not retained')
            ns['LOG'].close()
    print('OK: actual chain/Agent construction, 15 simulated batches, failed raw output retained')
    print('All simulated records were removed; no real inference was invoked.')
    if args.solve_reference:
        from ca_lean_workflow import EXAMPLE_MODEL_CODE, format_parameters
        for idx, row in queries.iterrows():
            ns = {}
            with contextlib.redirect_stdout(io.StringIO()):
                code = format_parameters(data.assemble(row['Query'])) + EXAMPLE_MODEL_CODE
                exec(compile(code, '<CA reference only>', 'exec'), ns)
            model = ns['model']
            gold = float(row['Label-objective'])
            if model.Status != 2 or abs(model.ObjVal - gold) > 1e-4 * max(1, abs(gold)):
                raise ValueError(f'CA reference query {idx + 1} differs from label: {model.Status}')
            print(f'Reference {idx + 1}: {model.ObjVal:.6f}; label {gold:.6f}; OK')
        print('PASS: reference LP objectives match 15/15 labels. This is not an OSS accuracy measurement.')
    print('PASS: CA package offline verification; actual model inference remains the server smoke test.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
