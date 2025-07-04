### LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach

#### Overview
LEAN-LLM-OPT is an innovative framework that leverages Large Language Models (LLMs) for the automatic formulation of large-scale optimization models. This repository contains the implementation of the framework, benchmark datasets, and experimental results, along with case studies, including its application to Singapore Airlines Choice-Based Revenue Management.


This repository accompanies the paper:
"LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach"
Authors: Kuo Liang, Yuhang Lu, Jianming Mao, Shuyi Sun, Chunwei Yang, Congcong Zeng, Hanzhang Qin, Ruihao Zhu, Chung-Piaw Teo.


#### Project Structure
| File/Folder                  | Description                                      |
|------------------------------|--------------------------------------------------|
| Air_NRM_Files/               | Required Files for Codes Related to Air_NRM |
| Large_Scale_Or_Files/        | Required Files for Codes Related to Large_Scale_Or |
| Small_Scale_Test_Files/      | Required Files for Codes Related to Small Scale Test |
| Generate_Label_Air_NRM_CA.ipynb |Generate Labels for Air NRM CA    |
| Generate_Label_Air_NRM_NP_Flow.ipynb |Generate Labels for Air NRM NP-Flow|
| Generate_Label_Air_NRM_NP_NoFlow.ipynb |  Generate Labels for Air NRM NP-No Flow|
| Generate_Label_Large_Scale_Or.ipynb | Generate Labels for Large Scale Or |
| Lean_LLM_OPT_Ablation_Study.ipynb | Ablation Study  |
| Lean_LLM_OPT_Air_NRM_CA.ipynb |  Our LLM Method for Air NRM CA                          |
| Lean_LLM_OPT_Air_NRM_NP_Flow.ipynb |  Our LLM Method for Air NRM NP-Flow   |
| Lean_LLM_OPT_Air_NRM_NP_NoFlow.ipynb | Our LLM Method for Air NRM NP-No Flow           |
| Lean_LLM_OPT_Large_Scale_Or.ipynb | Our LLM Method for Large Scale Or   |
| LEAN-LLM-OPT.py              | Main Python Script for Optimization Modeling     |
| rag_main_front_fixChatbox.py | Chatbox Integration Script                       |
| requirements.txt             | Python Dependencies                             |
| Supplementary_Codes.ipynb    | Additional Functions                            |
| readme.md                    | Project Documentation                           |
| LEAN_OPT.pdf                 | Accompanying Paper                              |

#### Installation
Prerequisites:
- Python 3.8 or higher
- Recommended environment: Conda or virtualenv

Steps:
1. Clone the repository:
```bash
git clone https://github.com/CoraLiang01/LeanOpt.git
cd LeanOpt
```
3. Install required Python packages:
```python
pip install -r requirements.txt
```
#### Usage
Running the Notebooks:
1. Open the desired .ipynb file (e.g., Lean_LLM_OPT_Air_NRM_CA.ipynb).
2. Follow the instructions within the notebook to run experiments and tests.

Running the Main Script:
```python
python LEAN-LLM-OPT.py
```
#### Datasets
1. Large-Scale-OR:
A benchmark dataset containing diverse optimization problems across domains:

- Small-size instances: Up to 20 variables.
- Medium-size instances: 21–100 variables.
- Large-size instances: More than 100 variables.

2. Air-NRM:
Specialized datasets for aviation resource management:

- Air-NRM-CA: Cabin capacity allocation dataset (35 instances).
- Air-NRM-NP: Network planning dataset (63 instances).

#### Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/<name>).
3. Commit your changes (git commit -m 'Add some <name>').
4. Push to the branch (git push origin feature/<name>).
5. Open a pull request.

#### Acknowledgements
Special thanks to:
Singapore Airlines: For providing simulated datasets and supporting the case study.

For inquiries, contact:
Kuo Liang: cora.liang1116@outlook.com

#### Demo
Explore the live demo of LEAN-LLM-OPT: https://lean-opt-llm.streamlit.app/.
