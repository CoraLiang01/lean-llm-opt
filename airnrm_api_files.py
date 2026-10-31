"""API-native file input for Air-NRM ablations; no local data parsing/retrieval."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import leanopt_exp as lx

DATA_DIR = Path(__file__).resolve().parent / 'Test_Dataset/Air_NRM/large_scale'
FILE_NAMES = ('flight.csv', 'od_demand.csv', 'v1.csv', 'v2.csv', 'flight_capacity.csv')
FILE_RULES = '''The attached .csv.txt files contain the complete, unchanged CSV text.
Read the attachments through the API. Determine the current question's parameters
yourself; no custom CSVQA, retrieval, or local data-extraction tool is available.
Return the complete model and exactly one self-contained gurobipy Python block.
Embed all required numeric data in that block. Do not read external files, use
pandas, invent placeholder values, or abbreviate dictionaries with ellipses.
'''


class ApiFileInput:
    def __init__(self, cfg, role='data_agent', client=None, paths=None):
        self.cfg = cfg
        self.spec = cfg.spec(role)
        if self.spec.provider != 'openai':
            raise ValueError('API file-input ablations require an OpenAI profile')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get(self.spec.api_key_env),
                            base_url=self.spec.base_url or None, max_retries=0)
        self.client = client
        self.paths = list(paths) if paths is not None else [DATA_DIR / n for n in FILE_NAMES]
        self.cache = {}
        atexit.register(self.close)

    def _attachments(self):
        content, manifest = [], []
        for path in self.paths:
            path = Path(path)
            raw = path.read_bytes()  # byte-for-byte upload, no dataframe or filtering
            digest = hashlib.sha256(raw).hexdigest()
            key = (str(path.resolve()), digest)
            if key not in self.cache:
                uploaded = self.client.files.create(
                    file=(path.name + '.txt', raw, 'text/plain'), purpose='user_data')
                self.cache[key] = uploaded.id
            file_id = self.cache[key]
            content.extend([{'type': 'input_text', 'text': 'Current data file: ' + path.name},
                            {'type': 'input_file', 'file_id': file_id}])
            manifest.append(dict(name=path.name, sha256=digest, bytes=len(raw), file_id=file_id))
        return content, manifest

    def answer(self, query, instructions=''):
        attachments, manifest = self._attachments()
        rec = lx._CURRENT_RECORD.get()
        if rec is not None:
            rec.set(file_input_protocol='responses-input-file-text-v1', attached_files=manifest)
        content = [{'type': 'input_text', 'text': query}] + attachments
        request = dict(model=self.spec.model, temperature=self.spec.temperature,
                       top_p=self.spec.top_p, max_output_tokens=self.spec.max_tokens or 32768,
                       input=[{'role': 'system', 'content': instructions + '\n\n' + FILE_RULES},
                              {'role': 'user', 'content': content}])
        # Do not inherit Chat Completions-only kwargs such as seed or n.
        def call():
            from langchain_core.outputs import Generation, LLMResult
            run_id = uuid4()
            tracker = lx.TRACKER.bind(self.cfg)
            tracker.on_llm_start({}, [json.dumps(request, ensure_ascii=False)],
                                 run_id=run_id, invocation_params=request)
            try:
                response = self.client.responses.create(**request)
            except Exception as exc:
                tracker.on_llm_error(exc, run_id=run_id)
                raise
            usage = response.usage
            tokens = {} if usage is None else {
                'prompt_tokens': usage.input_tokens, 'completion_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
                'prompt_tokens_details': {'cached_tokens': getattr(usage.input_tokens_details, 'cached_tokens', 0)},
                'completion_tokens_details': {'reasoning_tokens': getattr(usage.output_tokens_details, 'reasoning_tokens', 0)},
            }
            text = response.output_text or ''
            tracker.on_llm_end(LLMResult(generations=[[Generation(text=text)]],
                llm_output={'token_usage': tokens, 'model_name': response.model}), run_id=run_id)
            if response.status != 'completed' or not text.strip():
                raise ValueError(f'Unusable API response: {response.status}; {response.incomplete_details}')
            return text
        return lx.call_with_retry(call, self.cfg)

    def close(self):
        for key, file_id in list(self.cache.items()):
            try:
                self.client.files.delete(file_id)
            except Exception as exc:
                print(f'Uploaded file cleanup failed ({file_id}): {type(exc).__name__}')
            else:
                del self.cache[key]


class FileInputAgent:
    """Small adapter preserving the notebook's invoke/output interface."""
    def __init__(self, files, instructions):
        self.files, self.instructions = files, instructions

    def invoke(self, values):
        return {'output': self.files.answer(values['input'], self.instructions)}
