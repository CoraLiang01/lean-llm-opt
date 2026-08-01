# original_notebooks/

The **pre-instrumentation** notebooks, exactly as they were before the
`exp_config.yaml` + `leanopt_exp.py` refactor.

**Do not run these.** They contain hard-coded API key call sites, no budget cap,
no agent iteration limit, and no cost accounting. Run the notebooks in the
repository root instead — they carry the same file names and are the versions
everything else in the repo expects.

They are kept for two reasons:

1. **Reference / diff.** `PATCH_GUIDE.md` documents what changed and why; these
   files are the "before" side of that diff. `git diff` between a file here and
   its namesake at the root shows the whole instrumentation in one view.
2. **Regeneration.** `apply_instrumentation_patch.py` reads from this folder and
   writes the instrumented notebooks to the repository root:

   ```bash
   python apply_instrumentation_patch.py --check   # dry run, report only
   python apply_instrumentation_patch.py           # rewrite the root notebooks
   ```

   Verified: regenerating from this folder reproduces the current root
   notebooks byte-for-byte.

> **The root notebooks are generated files.** This folder plus
> `apply_instrumentation_patch.py` is the source of truth. If you hand-edit a
> notebook at the root and then run the patch script, your edit is overwritten —
> either port it into a rule in the script, or stop running the script.

## The seven pairs

| original (here) | instrumented (repository root) |
|---|---|
| `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb` | `../LEAN_LLM_OPT_4.1_Large-scale-or.ipynb` |
| `LEAN_LLM_OPT_4.1_Air_NRM.ipynb` | `../LEAN_LLM_OPT_4.1_Air_NRM.ipynb` |
| `LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb` | `../LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb` |
| `LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb` | `../LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb` |
| `Ablation_Study_Air_NRM_RAG_Only.ipynb` | `../Ablation_Study_Air_NRM_RAG_Only.ipynb` |
| `Ablation_Study_Air_NRM_Few-shot_Only.ipynb` | `../Ablation_Study_Air_NRM_Few-shot_Only.ipynb` |
| `Benchmark_Base_Model_Small_Scale.ipynb` | `../Benchmark_Base_Model_Small_Scale.ipynb` |

Same name, different folder — that is deliberate, so the pairing needs no
lookup table to read.

Note: `run_all_Generate_Label_Large_Scale_Or.ipynb` and
`uflp_txt_to_gurobipy_and_obj.ipynb` stay in the repository root and have no
counterpart here — they are label-generation utilities, not part of the
instrumented pipeline, and were never patched.
