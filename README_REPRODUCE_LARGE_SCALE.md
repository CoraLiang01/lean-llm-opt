# Reproducing the 101-case Large-scale-or experiments

The current pair of notebooks is [`LEAN_LLM_OPT_4.1_Large-scale.ipynb`](LEAN_LLM_OPT_4.1_Large-scale.ipynb) and [`LEAN_LLM_OPT_gpt_oss_20b_Large-scale.ipynb`](LEAN_LLM_OPT_gpt_oss_20b_Large-scale.ipynb). Both read `Test_Dataset/Large-scale-or/Large-scale-or-101.csv` and the unified `Large_Scale_Or_Files/RAG_Examples_All.csv` from the repository root. The older `*-Large-scale-or.ipynb` notebooks remain available but are not the entry points for this protocol.

The two implementations share the classification → type-specific retrieval/data extraction → formulation → code generation → Gurobi solve → objective comparison workflow. Their model-specific prompts and data-handling details are retained; they are **not** identical algorithms. The OSS notebook has no code-execution repair loop. Its ReAct agent may continue an incomplete answer before execution; that is not a post-execution repair.

## Requirements

- Python 3.12 and the focused packages in `requirements-large-scale.txt` (a separate environment is recommended). The repository-wide `requirements.txt` also covers unrelated experiments and pins older pandas/FAISS versions; use the focused file for this pair.
- A working Gurobi license. The notebook must be able to create and optimize a model.
- For OSS: an Ollama-compatible server with `gpt-oss:20b` and `nomic-embed-text` available. Set `OLLAMA_BASE_URL` if it is not at `http://127.0.0.1:11434`.
- For GPT-4.1: `OPENAI_API_KEY` with access to the model and its embedding API. Running the full benchmark incurs API charges.

From a fresh clone, install dependencies and check the input paths before any model call:

```bash
python -m pip install -r requirements-large-scale.txt
python run_large_scale.py --model oss --check
python run_large_scale.py --model gpt41 --check
```

Run one full 101-case automatic-route experiment into a **new** output directory:

```bash
python run_large_scale.py --model oss --output-dir runs/oss_repeat_1
# Optional, only if you intend to pay for GPT-4.1 evaluation:
python run_large_scale.py --model gpt41 --output-dir runs/gpt41_repeat_1
```

The runner never enables the 606 forced-route experiment. It refuses to reuse an existing output directory, so each invocation is independent. On completion, inspect `automatic/results.csv` and its summary in the specified output directory. Keep incomplete or failed records: they count as failures in the first-pass denominator. If an execution-infrastructure error is rerun, preserve and report the original result separately rather than silently replacing it.

For interactive use, start Jupyter in the repository root (or set `LEAN_PROJECT_ROOT` / `LEAN_LLM_OPT_ROOT` to that directory), open either notebook, and explicitly set `LEAN_RUN_AUTOMATIC=1` before running all cells. Automatic and forced-route execution are both off by default. Use `LEAN_RESULTS_DIR` to select an output directory. The OSS notebook also accepts `LEAN_ROW_START` and `LEAN_ROW_END` for a half-open row range; leave them unset for all 101 cases.

The reported solution metric is objective-value agreement after a successful optimization, using the notebook's relative and absolute tolerance of `1e-4`. It is not a claim of formulation equivalence or solution-transfer feasibility. To compare two runs, verify the 101 problem IDs, data file versions, model availability, complete exit status, and both first-pass and any infrastructure-retry counts.
