#!/usr/bin/env python3
"""Build a new portable CA-only server package without changing old CA/NP artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "oss20b_ca_pkg_src"
DATA = Path("Test_Dataset/Air_NRM/large_scale")
NOTEBOOK = DATA / "LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_CA_LargeScale.ipynb"


def build_notebook():
    nb = copy.deepcopy(json.loads((ROOT / DATA / "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb").read_text()))
    def cell(idx, text):
        nb["cells"][idx]["source"] = text.splitlines(keepends=True)
    cell(0, '# Independent CA direct-full v1 notebook; terminal entry: run_airnrm_ca.py\n'
         + 'import os, sys' + ''.join(nb['cells'][0]['source']).split('import os, sys', 1)[1])
    config = ''.join(nb['cells'][1]['source']).replace("model_profile='gpt-4.1'", "model_profile='gpt-oss-20b'")
    cell(1, config.replace("dataset='Air-NRM-NP-LS'", "dataset='Air-NRM-CA-LS'"))
    cell(9, 'from ca_lean_workflow import install_ca_workflow, validate_complete_generated_answer, PROTOCOL\n'
         'globals().update(install_ca_workflow(globals()))\n')
    cell(11, '# Old fixed template/data-transcription code is not used by this CA protocol.\n')
    routing = ''.join(nb['cells'][13]['source']).split('def Process_Input', 1)[0]
    cell(13, routing + '''def Process_Input(query):
    with lx.stage("classification"):
        original = Problemtype(query)
    print("Problem classification:", original)
    if not _is_airnrm_category(original):
        raise ValueError(f"unroutable CA classification: {original!r}")
    with lx.stage("modeling"):
        answer = ProcessCA(query)
    return "CA", answer
''')
    cell(15, '''from tqdm import tqdm

def Batch_Process_Queries(df, query_column="Query", log=LOG, cfg=CFG,
                          gold_column="Label-objective"):
    results = []
    for idx, row in tqdm(list(df.iterrows()), desc="Processing CA Queries"):
        query = row[query_column]
        category, raw = None, None
        with log.instance(instance_id=int(idx), query=query,
                          gold_objective=row.get(gold_column)) as rec:
            try:
                CA_SET_CURRENT_QUERY(query)
                category, raw = Process_Input(query)
                rec.set(pred_type=category, model_output=raw, output_protocol=PROTOCOL)
                code = validate_complete_generated_answer(raw, CA_DATA.assemble(query))
                rec.set(code_output=code, code_assembly="direct_LLM_code_extraction")
            except Exception as exc:
                rec.fail(exc)
                print(f"[{idx}] CA failed: {type(exc).__name__}: {exc}")
        results.append({"Category": category, "Original_Query": query, "Output": raw})
    return pd.DataFrame(results)
''')
    cell(16, '# Terminal scoring is optional and separate from generation.\n')
    cell(18, 'df = pd.read_csv("Test_Dataset/Air_NRM/large_scale/query_CA.csv")\n'
         'result_df = Batch_Process_Queries(df)\n'
         'result_df.to_csv(LOG.dir / "ca_results.csv", index=False)\n')
    for idx in range(19, len(nb['cells'])):
        if nb['cells'][idx]['cell_type'] == 'code':
            cell(idx, '# CA generation only; use score_runs.py for objective scoring.\n')
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    # Markdown from the source notebook must not advertise the NP protocol.
    for c in nb['cells']:
        if c['cell_type'] == 'markdown':
            c['source'] = ['Air-NRM large-scale CA · gpt-oss:20b · ca-direct-full-v1\n']
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / 'deliverables' /
              ('airnrm_ca_oss20b_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if output.exists():
        parser.error(f'output exists; choose a new path: {output}')
    files = ['leanopt_exp.py', 'run_notebook.py', 'score_runs.py', 'aggregate_runs.py',
             'Large_Scale_Or_Files/RefData.csv', 'Large_Scale_Or_Files/RAG_Example_SBLP_CA_LargeScale.csv']
    files += [str(DATA / name) for name in ('query_CA.csv', 'v1.csv', 'v2.csv', 'od_demand.csv',
              'flight.csv', 'flight_capacity.csv', 'np_lean_workflow.py', 'SBLP_CA_Label/CA_answer.csv')]
    output.mkdir(parents=True)
    for name in files:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    (output / 'exp_config.yaml').write_text((ROOT / 'exp_config.yaml').read_text()
                                           + '\nartifact_protocol: ca-direct-full-v1\n', encoding='utf-8')
    for name in ('ca_lean_workflow.py', 'run_airnrm_ca.py', 'check_server.py', 'verify_package.py',
                 'setup.sh', 'requirements-server.txt', 'README.md'):
        shutil.copyfile(SOURCE / name, output / name)
    (output / NOTEBOOK).write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'Air-NRM-CA-LS',
                'profile': 'gpt-oss-20b', 'protocol': 'ca-direct-full-v1', 'sha256': {}}
    for p in sorted(output.rglob('*')):
        if p.is_file():
            manifest['sha256'][p.relative_to(output).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (output / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with tarfile.open(str(output) + '.tar.gz', 'w:gz') as tar:
        tar.add(output, arcname=output.name)
    with zipfile.ZipFile(str(output) + '.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                archive.write(p, str(Path(output.name) / p.relative_to(output)))
    print(f'Folder: {output}\nZIP: {output}.zip\nLinux: {output}.tar.gz')


if __name__ == '__main__':
    main()
