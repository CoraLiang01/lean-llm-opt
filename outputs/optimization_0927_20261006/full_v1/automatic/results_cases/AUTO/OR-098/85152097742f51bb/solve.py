import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owner_names = df['Owner'].tolist()
if set(owner_names) != set(worker_cols):
    raise ValueError("Mismatch between 'Owner' values and worker columns. Each owner must correspond to a worker column.")
D = df.set_index('Owner')[worker_cols].apply(pd.to_numeric, errors='raise')
m = gp.Model('MutualAidWageFairness')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for k in worker_cols:
    income_coeff = D.loc[[owner for owner in owner_names if owner != k], k].sum()
    expenditure_terms = []
    for j in worker_cols:
        if j == k:
            continue
        coeff = D.loc[k, j]
        expenditure_terms.append((coeff, wage_vars[j]))
    expr = income_coeff * wage_vars[k] - gp.quicksum((coeff * var for (coeff, var) in expenditure_terms))
    m.addConstr(expr == 0, name=f'fairness_{k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()