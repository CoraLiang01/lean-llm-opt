"""Derive component removals only after the user-approved aggregate full gate passes."""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path

from check_0927_review import gate
from derive_0927_final_methods import source, put, replace_function, save, strings, REACT_HELPER, QUERY_ONLY_RAG
from evaluate_0927_optimization import namespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
RUNS = ROOT / "outputs/optimization_0927_20261006"
FULL_NAME = "LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb"
NAMES = {"rag_only": "Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb",
         "few_shot_only": "Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb",
         "examples_and_route": "LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def use_direct_nrm_formulation(book):
    """Few-shot NRM passes numerical formulation to codegen, without runtime injection."""
    cfg = source(book, 3).replace('"NRM": "planned"', '"NRM": "legacy"', 1)
    cfg = cfg.replace(
        '# Preserve the original route behavior: only NRM uses planned extraction.',
        '# Few-shot CSV routes generate self-contained code from numerical formulations.')
    cfg = cfg.replace('manual_few_shot_only_', 'manual_few_shot_only_nrm_direct_')
    put(book, 3, cfg)
    text = source(book, 15)
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_NRM_response')
    nrm = ast.get_source_segment(text, node)
    instruction = next(n.value.value for n in node.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'route_instruction' for t in n.targets))
    nrm = nrm.replace(instruction, '''Use the complete Python Observation and return a complete numerical formulation.
Define index sets, parameters, variables, objective, and constraints. Retrieved Information
must include every query-required product/resource identifier, objective coefficient,
resource-consumption coefficient, capacity, bound, and any additional parameter, with complete
vectors and matrices and explicit axis labels. Preserve exact identifiers, source order, values,
and matrix orientation. Apply only the selection rules stated in the query; do not invent,
aggregate, abbreviate, or silently omit required data. The formulation must contain all data
needed to generate self-contained code; no CSVQA_DATA or external CSV will be supplied to it.''', 1)
    nrm = nrm.replace('CSVQA_PLANNED_PROMPTS["NRM"]', 'CSVQA_LEGACY_SYSTEM_PROMPT')
    nrm = nrm.replace('CSVQA_TOOL_DESCRIPTIONS["NRM"]', 'CSVQA_LEGACY_TOOL_DESCRIPTION')
    nrm = nrm.replace(
        'Return a concise symbolic model and Data Mapping only; never enumerate data rows or repeat examples.',
        'Return a concise numerical model with all required identifiers and coefficients; omit explanations and repeated examples, never required data.')
    put(book, 15, replace_function(text, 'get_NRM_response', nrm))
    put(book, 6, '## 3. CSV loading and route modes\n\nPython supplies complete CSV Observation for modeling. NRM/RA/TP/AP/FLP use numerical formulations and self-contained code, without runtime data injection.')
    put(book, 14, '## 7. Route retrieval and NRM formulation\n\nRetain reference examples. NRM produces a complete numerical formulation from the Python Observation, then generates code directly like the other numerical routes.')
    put(book, 26, '## 13. Generate Python code\n\nNRM/RA/TP/AP/FLP generate self-contained code from the numerical formulation and original query. Complete source Observation is not appended to code-generation prompts. Others with CSV generates code in section 12.')


def main(version):
    assert gate(version)["passed"], "New ablation/LOTO work requires a complete improved full round"
    frozen = RUNS / version / "frozen_notebook.ipynb"
    staged = OUT / "delivery" / version
    staged.mkdir(parents=True, exist_ok=True)
    full = json.loads(frozen.read_text())
    config = source(full, 3)
    filename_node = next(n for n in ast.parse(config).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "NOTEBOOK_FILENAME" for t in n.targets))
    lines = config.splitlines(True)
    config = "".join(lines[:filename_node.lineno - 1]) + f'NOTEBOOK_FILENAME = "{FULL_NAME}"\n' + "".join(lines[filename_node.end_lineno:])
    directory_node = next(n for n in ast.parse(config).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'RESULTS_DIR' for t in n.targets))
    lines = config.splitlines(True)
    config = ''.join(lines[:directory_node.lineno-1]) + f'RESULTS_DIR = PROJECT_ROOT / "outputs/review_0927_20261006/manual_full_{version}"\n' + ''.join(lines[directory_node.end_lineno:])
    put(full, 3, config)
    put(full, 0, "# LEAN-LLM-OPT — 0927 通用最小修改版\n\n"
        f"完整全量验证：outputs/optimization_0927_20261006/{version}。"
        "保留 ReAct、原模型和执行机制；NRM 使用原 planned，其余 route 使用 legacy。"
        "数据校验、标识符映射和提示要求为通用规则，没有题号分支或参考答案参与生成。"
        "实验入口默认关闭；失败记录保留，新一轮使用新输出目录。\n")
    full["metadata"]["experiment_provenance"] = {"revision": version, "method": "full",
        "evaluated_frozen_notebook": str(frozen), "evaluated_frozen_sha256": sha(frozen),
        "baseline_archive": str(OUT / "baseline" / FULL_NAME),
        "delivery_changes": "Filename, output directory, documentation and provenance only; inference functions and constants match the evaluated full."}
    save(staged / FULL_NAME, full)
    full_sha = sha(staged / FULL_NAME)
    ns = namespace(frozen)
    rows = [r for r in ns["load_records"](RUNS / version / "automatic/results.csv") if r["benchmark"] == "main"]
    assert len(rows) == 101 and all(r.get("predicted_label") and r.get("assigned_route") for r in rows)
    cache = RUNS / version / "classification_main.csv"
    fields = ["problem_id", "query", "dataset_address", "predicted_label", "assigned_route", "experiment_mode", "forced_route"]
    buffer = __import__("io").StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8-sig")
    if cache.exists():
        assert cache.read_bytes() == data
    else:
        cache.write_bytes(data)
    guidance_node = next(n for n in ast.parse(source(full, 27)).body if isinstance(n, ast.FunctionDef) and n.name == "_generate_code")
    guidance = [n.value.args[0].value for n in ast.walk(guidance_node)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "append" and n.value.args and isinstance(n.value.args[0], ast.Constant)
        and isinstance(n.value.args[0].value, str) and "Mandatory result interface:" in n.value.args[0].value]
    assert len(guidance) == 1
    delivered = {FULL_NAME: full_sha}
    for method, name in NAMES.items():
        book = json.loads((OUT / "baseline" / name).read_text())
        book["cells"][:36] = copy.deepcopy(full["cells"][:36])
        cfg = source(full, 3).replace(f'NOTEBOOK_FILENAME = "{FULL_NAME}"', f'NOTEBOOK_FILENAME = "{name}"')
        cfg = cfg.replace(f'outputs/review_0927_20261006/manual_full_{version}', f'outputs/review_0927_20261006/manual_{method}_{version}')
        cfg = cfg.replace('EXPERIMENT_METHOD = "full"', f'EXPERIMENT_METHOD = "{method}"')
        cfg += (f'\nBASE_NOTEBOOK_PATH = PROJECT_ROOT / "{FULL_NAME}"\n'
                f'BASE_NOTEBOOK_SHA256 = "{full_sha}"\n'
                f'CLASSIFICATION_RESULTS_PATH = Path({str(cache)!r})\n'
                f'CLASSIFICATION_RESULTS_SHA256 = "{sha(cache)}"\n')
        if method == "examples_and_route":
            cfg += '\nLOTO_VARIANT = "examples_and_route"\nLOTO_REMOVE_ROUTE = True\nLOTO_BASE_NOTEBOOK = BASE_NOTEBOOK_PATH.name\nLOTO_BASE_SHA256 = BASE_NOTEBOOK_SHA256\n'
        put(book, 3, cfg)
        put(book, 0, f"# {method} — 最终全量的组件移除实验\n\n"
            f"派生自通过整体比较的 {version}；全量源码 SHA-256：{sha(frozen)}。"
            "共享模型、求解参数、匹配容差、数据及执行机制；结果全部保留，不按单题选择重跑。\n")
        if method != "examples_and_route":
            put(book, 29, source(book, 29).replace("invoke_classifier(query) if forced_route is None else {}",
                "classification_for_case(case) if forced_route is None else {}"))
            cache_source = source(book, 37).replace('    path = Path(classification_csv).resolve()',
                '    path = Path(classification_csv).resolve()\n    if hashlib.sha256(path.read_bytes()).hexdigest() != CLASSIFICATION_RESULTS_SHA256:\n        raise ValueError("Classification cache differs from the evaluated full round")')
            cache_source = cache_source.replace('        if class_to_workflow_route(label) != route:',
                '        expected_route = class_to_workflow_route(label) if case["dataset_address"] else "Others"\n        if expected_route != route:')
            cache_source = cache_source.replace('    if class_to_workflow_route(label) != case["cached_assigned_route"]:',
                '    expected_route = class_to_workflow_route(label) if normalize_data_address(case.get("dataset_address")) else "Others"\n    if expected_route != case["cached_assigned_route"]:')
            put(book, 37, cache_source)
        if method == "rag_only":
            put(book, 15, replace_function(source(book, 15), "retrieve_rag_examples", "def retrieve_rag_examples(route, query, k=1):\n    return []"))
            put(book, 25, replace_function(source(book, 25), "get_others_without_CSV_response", QUERY_ONLY_RAG))
        elif method == "few_shot_only":
            put(book, 8, "## 移除当前题的数据抽取计划\nPython 提供完整原始 Observation，不使用 LLM 做数据预处理。")
            put(book, 9, "# Current-case extraction planner/executor removed.")
            put(book, 10, "## 完整 Python Observation\n保持所有解析后的行、列与单元格。")
            put(book, 11, strings["DIRECT"])
            full_helper = next(n for n in ast.parse(source(full, 15)).body
                               if isinstance(n, ast.FunctionDef) and n.name == "formulate_with_csvqa")
            retained_guidance = "\n".join(ast.unparse(n) for n in full_helper.body
                if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == "prefix")
            assert retained_guidance
            react_helper = REACT_HELPER.replace('    return formulate_from_python_observation(query, observation, prefix, suffix)',
                "\n".join("    " + line for line in retained_guidance.splitlines()) +
                '\n    return formulate_from_python_observation(query, observation, prefix, suffix)')
            put(book, 15, replace_function(source(book, 15), "formulate_with_csvqa", react_helper))
            # Remove CURRENT tool instructions only; reference examples retain their content.
            for index in [15, 17, 19, 21, 23]:
                s = source(book, index)
                s = s.replace("Call CSVQA exactly once and return an ABSTRACT model.", "Use the complete Python Observation and return an ABSTRACT model.")
                s = s.replace("You MUST call CSVQA exactly once before the Final Answer and use all returned rows.", "Use the complete Python Observation before the Final Answer, retaining every query-required row.")
                s = s.replace("Call CSVQA at least once and return a complete numerical formulation.", "Use the complete Python Observation and return a complete numerical formulation.")
                s = s.replace("When you need to retrieve information from the CSV file, use the provided tool.", "The complete source CSV data is already in the Python Observation; no data tool is available.")
                s = s.replace("When CSVQA has applied a validated filter, use its returned\nrecords directly and retain that exact predicate in Data Mapping; never re-filter by a guessed\ncomplete label. For FALLBACK_FULL_DATA, state and implement the query-supported selection explicitly.", "Python has supplied all source records without filtering; state and implement the query-supported selection explicitly in Data Mapping.")
                put(book, index, s)
            s = source(book, 25).replace("schema = csv_schema_preview(dataset_address, query=user_query)",
                'source_observation = direct_source_observation(dataset_address, "Others")\n        schema = source_observation\n        code_schema = direct_source_schema(dataset_address)')
            s = s.replace("schema=schema,\n            abstract_plan=abstract_model_plan,", "schema=code_schema,\n            abstract_plan=abstract_model_plan,")
            s = s.replace('"observation": schema,', '"observation": source_observation,').replace('"status": "LEGACY_SCHEMA"', '"status": "PYTHON_FULL_CSV"')
            s = s.replace("hashlib.sha256(schema.encode()).hexdigest()", "hashlib.sha256(source_observation.encode()).hexdigest()")
            put(book, 25, s)
            direct = strings["DIRECT_CODE"].replace("    response = make_llm().invoke(", "    parts.append(" + repr(guidance[0]) + ")\n    response = make_llm().invoke(")
            direct = direct.replace("    route = normalize_route(route)", '''    route = normalize_route(route)
    if legacy_observation:
        raise ValueError("Complete Observation must not enter code generation")
    if data_payload:
        schema = json.loads(data_payload) if isinstance(data_payload, str) else data_payload
        if any("records" in table for table in schema["tables"]):
            raise ValueError("Code generation accepts only structural schema, never source records")''', 1)
            put(book, 27, replace_function(source(book, 27), "_generate_code", direct))
            s = source(book, 29).replace('        record["pipeline_stage"] = "code_generation"',
                '        code_schema = json.dumps(data_schema_only(payload), ensure_ascii=False) if payload else ""\n        record["pipeline_stage"] = "code_generation"')
            s = s.replace("route, query, data_payload=payload,", "route, query, data_payload=code_schema,")
            s = s.replace('legacy_observation=formulation.get("observation", "")\n                    if route == "RA" and mode == "legacy" else "",', 'legacy_observation="",')
            put(book, 29, s)
            s = source(book, 7)
            for fn in ["invoke_react_with_required_csvqa", "_build_profile"]:
                s = replace_function(s, fn, f'def {fn}(*args, **kwargs):\n    raise RuntimeError("CSVQA/extraction is removed in Few-shot Only")')
            put(book, 7, s)
            use_direct_nrm_formulation(book)
        else:
            s = source(book, 25).replace("documents = loader.load()", "documents = filter_loto_query_only_documents(loader.load())")
            s = s.replace('"prefix": prefix,\n            "suffix": suffix,', '"prefix": filter_loto_query_only_triggers(prefix),\n            "suffix": suffix,')
            put(book, 25, s)
        book["metadata"]["experiment_provenance"] = {"method": method, "revision": version,
            "full_delivered_sha256": full_sha, "full_evaluated_sha256": sha(frozen),
            "classification_cache": str(cache), "classification_cache_sha256": sha(cache),
            "scope": "Specified component removals only; shared inference/execution/scoring retained."}
        if method == "few_shot_only":
            book["metadata"]["experiment_provenance"]["nrm_flow"] = "User-requested numerical formulation -> self-contained code; no NRM runtime data injection. Requires a fresh evaluation."
        save(staged / name, book)
        delivered[name] = sha(staged / name)
    control = {"version": version, "staged_directory": str(staged), "delivered_sha256": delivered,
               "full_evaluated_sha256": sha(frozen), "classification_main_cache": str(cache),
               "classification_main_sha256": sha(cache), "gates": gate(version)}
    (OUT / "staged_delivery_control.json").write_text(json.dumps(control, ensure_ascii=False, indent=2))
    print(json.dumps(control, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    main(parser.parse_args().version)
