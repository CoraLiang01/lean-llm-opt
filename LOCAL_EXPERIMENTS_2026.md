# Local experiment code, July–October 2026

This code-only import is based on main commit 5613281bd (2026-10-28).
Existing project datasets and October notebooks remain in place.

| Task | Entry point |
|---|---|
| V2/101-instance accounting | `run_v2_101.py`, `run_v2_101.sh` |
| 101-instance SingleCall | `Test_Dataset/Large-scale-or/run_singlecall_cost_benchmark.py` |
| API-file ablations | `airnrm_api_files.py`, `run_airnrm_file_ablations_terminal.sh` |
| Air-NRM SingleCall/two-stage | `Test_Dataset/Air_NRM/large_scale/run_agentic_benchmark.py` |
| OneShot | `run_airnrm_fewshot_benchmark.py` |
| NP query/label builders and workflow | `Test_Dataset/Air_NRM/large_scale/build_np_queries.py`, `build_np_labels.py`, `run_np_lean.py` |
| Medium-scale tools | `run_airnrm_medium_resources.py`, `Test_Dataset/Air_NRM/medium_scale/` |
| Multi-file reference tools | `Test_Dataset/Large-scale-or/benchmark_dataset/reference/` |
| Gemini 3.0 resource estimates | `estimate_gemini_3pro.py`; see `ESTIMATE_GEMINI_3PRO_README.md` |

## Setup and data

Use the project's experiment environment and existing setup instructions.
The native Gemini SDK and SciPy are listed in `requirements-experiments.txt`.
Supply credentials at runtime.

The new CA/NP/medium workflows expect their matching local input
files and labels. Those additional data artifacts remain local for
this code-first import. Existing project inputs are retained. Provide the
matching local inputs before running the new workflows; preflight and builder
checks identify missing files.

The restored `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb` is the local V2 workflow;
main's `_1018` notebooks remain separate versions.

Startup scripts named `run_*.py` are code. Run directories, call logs, generated
answers, scoring tables, solution JSON, LP labels, figures and numeric usage
fixtures are excluded. Notebook execution outputs have been cleared.

## Offline estimator example

No model requests or run logs are required:

```bash
python3 -m unittest test_estimate_gemini_3pro
python3 estimate_gemini_3pro.py --total-tokens 810416 --api-cost 4.896942
```

With your own private logs, use `--run-dir /path/to/run` to estimate
thinking/cache scenarios from the per-call breakdown.
