import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
all_columns = list(df.columns)
if 'Owner' not in all_columns:
    raise KeyError("CSV must contain an 'Owner' column.")
worker_columns = [col for col in all_columns if col != 'Owner']
owner_names = df['Owner'].tolist()
for w in worker_columns:
    df[w] = df[w].astype(int)

def normalize(s):
    return s.strip()
owner_name_to_rowidx = {normalize(owner): idx for (idx, owner) in enumerate(owner_names)}
worker_to_owneridx = {}
for w in worker_columns:
    w_norm = normalize(w)
    if w_norm in owner_name_to_rowidx:
        worker_to_owneridx[w] = owner_name_to_rowidx[w_norm]
    else:
        raise ValueError(f"Worker '{w}' does not have a matching owner row.")
m = gp.Model('mutual_wage_balance')
w_vars = m.addVars(worker_columns, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_columns[0]
m.addConstr(w_vars[first_worker] == 60.0, name='wage_normalization')
for j in worker_columns:
    o_j_idx = worker_to_owneridx[j]
    income = gp.quicksum((df.at[o_idx, j] * w_vars[j] for o_idx in range(len(owner_names)) if o_idx != o_j_idx))
    payment = gp.quicksum((df.at[o_j_idx, k] * w_vars[k] for k in worker_columns))
    m.addConstr(income == payment, name=f'fairness_{j}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Worker daily wages (yuan):')
    for w in worker_columns:
        print(f'  {w}: {w_vars[w].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')