import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
workers = list(worker_columns)
num_workers = len(workers)
owners = df['Owner'].tolist()
num_owners = len(owners)
for w in workers:
    df[w] = df[w].astype(int)
owner_to_index = {owner: idx for (idx, owner) in enumerate(owners)}
worker_to_index = {worker: idx for (idx, worker) in enumerate(workers)}

def normalize(s):
    return s.strip().casefold()
owner_norm = [normalize(o) for o in owners]
worker_norm = [normalize(w) for w in workers]
worker_home_row = {}
for (j, wnorm) in enumerate(worker_norm):
    matches = [i for (i, onorm) in enumerate(owner_norm) if onorm == wnorm]
    if len(matches) != 1:
        raise ValueError(f"Worker '{workers[j]}' does not have exactly one matching owner row (found {len(matches)})")
    worker_home_row[workers[j]] = matches[0]
d = df[workers].to_numpy(dtype=int)
for (j, w) in enumerate(workers):
    total_days = d[:, j].sum()
    if total_days != 10:
        raise ValueError(f"Worker '{w}' has total contributed days {total_days}, expected 10.")
m = Model('mutual_payment_balance')
m.Params.OutputFlag = 0
wage_vars = {}
for (idx, w) in enumerate(workers):
    if idx == 0:
        wage_vars[w] = m.addVar(lb=60.0, ub=60.0, vtype=GRB.CONTINUOUS, name=f'w_{w}')
    else:
        wage_vars[w] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'w_{w}')
m.update()
for (k, w_k) in enumerate(workers):
    income = quicksum((d[i, k] * wage_vars[w_k] for i in range(num_owners) if i != worker_home_row[w_k]))
    expenditure = quicksum((d[worker_home_row[w_k], j] * wage_vars[workers[j]] for j in range(num_workers) if j != k))
    m.addConstr(income == expenditure, name=f'balance_{w_k}')
m.setObjective(0, GRB.MINIMIZE)
m.optimize()