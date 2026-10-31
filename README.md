### LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach

#### Project Overview
Large-scale optimization is a key backbone in modern business decision-making. However, the process of building these models is often labor-intensive and time-consuming. We address this by proposing a multi-agent framework LEAN-LLM-OPT, which takes a query (a problem description and associated datasets) as input and orchestrates a team of LLM agents to output the optimization formulation. LEAN-LLM-OPT innovatively applies few-shot learning to teach LLM agents how they could effectively apply reasoning and customized tools to build optimization models in our benchmark Large-scale-or and a Singapore Airlines choice-based revenue management use case.

This repository accompanies the paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027

#### Directory Structure

1. **Large_Scale_Or_Files/**  
   - Contains all required Ref-Data files for further experiments.

2. **Test_Dataset/**  
   - Contains test datasets, including:
     - Air_NRM (Singapore Airlines use case)  
     - Large-scale-or (101 benchmarks)  
     - Small-scale Benchmarks

3. **Results/**  
   - Contains all numerical results in the paper

4. **Ablation_Study_Air_NRM_Few-shot_Only.ipynb**  
   - A Jupyter notebook conducting ablation studies in a few-shot-only setting for the Air_NRM use case.
  
5. **Ablation_Study_Air_NRM_RAG_Only.ipynb**  
   - Another ablation study focusing on RAG-Only setting for the Air_NRM use case.

6. **LEAN_LLM_OPT_4.1_Air_NRM.ipynb**  
   - Implements the LEAN-LLM-OPT framework based on GPT-4.1 for the Air_NRM scenario.

7. **LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb**  
   - Implements the LEAN-LLM-OPT framework based on gpt-oss-20b for the Air_NRM scenario.

8. **LEAN_LLM_OPT_4.1_Large-scale-or.ipynb**  
   - Implements the LEAN-LLM-OPT framework based on GPT-4.1 for large-scale and small-scale experiments.

9. **LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb**  
   - Implements the LEAN-LLM-OPT framework based on gpt-oss-20b for large-scale and small-scale experiments.

10. **README.md**  
   - The current project description (this file).

11. **requirements.txt**  
   - Lists required Python packages.  
#### Additional Experiments

1. **Ablation_Study_gpt_oss_20b_Large-scale_RAG_Only.ipynb** and **Ablation_Study_gpt_oss_20b_Large-scale_Few-shot_Only.ipynb**
   - Component ablations of the gpt-oss-20b workflow on the 101 Large-scale-or instances, with the same two settings as the GPT-4.1 ablations.
   - Results: **[outputs/ablation_gpt_oss_20b_rag_only/automatic/](outputs/ablation_gpt_oss_20b_rag_only/automatic/)** and **[outputs/ablation_gpt_oss_20b_few_shot_only/automatic/](outputs/ablation_gpt_oss_20b_few_shot_only/automatic/)**.
   - Code: **[Ablation_Study_gpt_oss_20b_Large-scale_RAG_Only.ipynb](Ablation_Study_gpt_oss_20b_Large-scale_RAG_Only.ipynb)** and **[Ablation_Study_gpt_oss_20b_Large-scale_Few-shot_Only.ipynb](Ablation_Study_gpt_oss_20b_Large-scale_Few-shot_Only.ipynb)**.

2. **fallback mechanism/fallback.ipynb**
   - The audit-and-repair fallback: after the Primary workflow, it audits the formulation logic, the data usage and the code, repairs the first failing layer, re-executes the program in isolation and audits it again. It contains the shared definitions and a GPT-4.1 and a gpt-oss-20b backend.
   - Code: **[fallback mechanism/fallback.ipynb](fallback%20mechanism/fallback.ipynb)**.

3. **fallback mechanism/LEAN_LLM_OPT_4.1_Large-scale_with_Fallback.ipynb** and **fallback mechanism/LEAN_LLM_OPT_gpt_oss_20b_Large-scale_with_Fallback.ipynb**
   - Each Primary workflow followed by the fallback with the same model, run on the 36 structural variants in `benchmark_dataset`.
   - Results: **[outputs/variants_fallback_gpt41/](outputs/variants_fallback_gpt41/)** and **[outputs/variants_fallback_gpt_oss_20b/](outputs/variants_fallback_gpt_oss_20b/)**.
   - Code: **[fallback mechanism/LEAN_LLM_OPT_4.1_Large-scale_with_Fallback.ipynb](fallback%20mechanism/LEAN_LLM_OPT_4.1_Large-scale_with_Fallback.ipynb)** and **[fallback mechanism/LEAN_LLM_OPT_gpt_oss_20b_Large-scale_with_Fallback.ipynb](fallback%20mechanism/LEAN_LLM_OPT_gpt_oss_20b_Large-scale_with_Fallback.ipynb)**.

4. **Parameter-perturbation evaluation (303 numerical variants)**
   - The variants are in `Test_Dataset/Perturbation-Large-scale-or`. Each is solved end to end by the main workflows.
   - Results: **[outputs/perturbation_303/summary.csv](outputs/perturbation_303/summary.csv)**, **[outputs/perturbation_303/accuracy_by_seed.csv](outputs/perturbation_303/accuracy_by_seed.csv)** and **[outputs/perturbation_303/accuracy_by_problem_type.csv](outputs/perturbation_303/accuracy_by_problem_type.csv)**.

#### Installation
Prerequisites:
- Python 3.10 or higher
- Recommended environment: Conda or virtualenv

#### Usage Instructions
1. Clone the repository
```bash
gh repo clone CoraLiang01/lean-llm-opt
```
2. Navigate to the Project Directory & Install Dependencies
```python
cd lean-llm-opt
pip install -r requirements.txt
```

3. Run Jupyter Notebooks
- Open any of the provided notebooks (e.g., Ablation_Study_Air_NRM_Few-shot_Only.ipynb) to explore experiments or replicate results.

#### Acknowledgements
Special thanks to:
- Singapore Airlines: For providing simulated datasets and supporting the case study.

For inquiries, please contact:
- Kuo Liang: cora.liang1116@outlook.com
- Hanzhang Qin: hzgin@nus.edu.sg
- Ruihao Zhu: ruihao.zhu@cornell.edu

Local experiment extensions (July–October 2026): see [tools, data versions and offline checks](LOCAL_EXPERIMENTS_2026.md).
