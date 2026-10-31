"""Offline tests: actual request attachments, provider usage, and ablation boundaries."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import leanopt_exp as lx
import build_ls_ablations as builder
from airnrm_api_files import ApiFileInput, DATA_DIR, FILE_NAMES
from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import install_np_workflow


def cfg():
    return lx.load_config('exp_config.yaml', model_profile='gpt-4.1',
                         method='Abl-FewShotOnly', dataset='Air-NRM-CA-LS')


def client():
    c = Mock()
    c.files.create.side_effect = [NS(id=f'file-{i}') for i in range(5)]
    c.responses.create.return_value = NS(
        status='completed', output_text='```python\nprint(1)\n```',
        model='gpt-4.1-2025-04-14', incomplete_details=None,
        usage=NS(input_tokens=1000, output_tokens=200, total_tokens=1200,
                 input_tokens_details=NS(cached_tokens=100),
                 output_tokens_details=NS(reasoning_tokens=0)))
    return c


class ApiFilesTests(unittest.TestCase):
    def test_full_bytes_attached_and_cached_not_just_id_text(self):
        c = client(); api = ApiFileInput(cfg(), client=c)
        # No target-data parsing allowed, including pandas in callers.
        with patch('pandas.read_csv', side_effect=AssertionError('local dataframe read')):
            api.answer('query one'); api.answer('query two')
        self.assertEqual(c.files.create.call_count, 5)
        for name, call in zip(FILE_NAMES, c.files.create.call_args_list):
            upload = call.kwargs['file']
            self.assertEqual(upload[0], name + '.txt')
            self.assertEqual(upload[1], (DATA_DIR / name).read_bytes())
            self.assertEqual(call.kwargs['purpose'], 'user_data')
        req = c.responses.create.call_args.kwargs
        self.assertNotIn('tools', req)
        self.assertEqual([x['file_id'] for x in req['input'][1]['content'] if x['type']=='input_file'],
                         [f'file-{i}' for i in range(5)])
        api.close(); self.assertEqual(c.files.delete.call_count, 5)

    def test_provider_usage_logged_once(self):
        c = client(); config = cfg(); api = ApiFileInput(config, client=c)
        rec = Mock(); token = lx._CURRENT_RECORD.set(rec)
        try: api.answer('query')
        finally: lx._CURRENT_RECORD.reset(token)
        calls = [x.args[0] for x in rec.add_call.call_args_list]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['total_tokens'], 1200)
        self.assertEqual(calls[0]['cached_prompt_tokens'], 100)
        self.assertEqual(calls[0]['token_source'], 'provider')
        self.assertGreater(calls[0]['cost_usd'], 0)

    def test_api_failure_is_logged(self):
        c=client(); c.responses.create.side_effect=RuntimeError('transport failed')
        config=cfg(); config.max_retries=1
        rec=Mock(); token=lx._CURRENT_RECORD.set(rec)
        rec.n_retries=0; rec.errors=[]
        try:
            with self.assertRaisesRegex(RuntimeError,'transport failed'):
                ApiFileInput(config,client=c).answer('query')
        finally: lx._CURRENT_RECORD.reset(token)
        self.assertFalse(rec.add_call.call_args.args[0]['ok'])

    def test_workflow_ca_keeps_example_without_fake_tools(self):
        nb=builder.load_src();template,fmt=builder.extract_shared_prompt_pieces(nb)
        out=builder.build_fewshot_only(nb,template,fmt)
        prompt=next(''.join(c['source']) for c in out['cells'] if 'PREFIX_CA = (' in ''.join(c['source']))
        ns={};exec(prompt,ns)
        self.assertNotIn('CSVQA', ns['PREFIX_CA'])
        self.assertNotIn('Action: Get', ns['PREFIX_CA'])
        self.assertIn(builder.EXEMPLAR_PROMPT, ns['PREFIX_CA'])
        self.assertNotIn('{{', ns['PREFIX_CA'])
        files=Mock();ns['API_FILES']=files
        exec(builder.FS_AGENT,ns);ns['Fewshot_Only_CA']('new query')
        files.answer.assert_called_once_with('new query', ns['PREFIX_CA'])
        for i,c in enumerate(out['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')

    def test_np_workflow_reads_no_target_data_locally(self):
        files=Mock()
        files.answer.return_value='answer'
        namespace={name:Mock(side_effect=AssertionError('custom tool called')) for name in
                   ('LoadFiles','CSVLoader','FAISS','EMBEDDINGS','Tool','initialize_agent','AgentType')}
        namespace.update(CFG=cfg(),lx=lx,API_FILES=files)
        with patch('pandas.read_csv',side_effect=AssertionError('dataframe read')):
            installed=install_np_workflow(namespace,ablation='fewshot_only')
            with (DATA_DIR/'query_NP_Flow.csv').open() as f: queries=list(csv.DictReader(f))
            for row in queries:
                self.assertEqual(installed['ProcessPolicyFlow'](row['Query']), 'answer')
        self.assertEqual(files.answer.call_count,15)
        prefix=files.answer.call_args.args[1]
        self.assertNotIn('CSVQA',prefix)
        self.assertNotIn('Action: CName',prefix)
        self.assertIn('Final Answer:',prefix)


if __name__=='__main__':unittest.main()
