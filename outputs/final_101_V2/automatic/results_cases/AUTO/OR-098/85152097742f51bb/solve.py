import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
owner_ids = df['Owner'].astype(str).tolist()
if len(worker_cols) != len(owner_ids):
    raise ValueError('Number of workers (columns) and owners (rows) do not match.')
workers = worker_cols
owners = owner_ids
for i, (w, o) in enumerate(zip(workers, owners)):
    if w != o:
        raise ValueError(f"Mismatch at position {i}: worker column '{w}' != owner row '{o}'.")
num_workers = len(workers)
days = df[worker_cols].to_numpy(dtype=float)
m = gp.Model('mutual_payment_balance')
wage_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for k_idx, k in enumerate(workers):
    income_coeff = 0.0
    expenditure_coeffs = {}
    income_sum = sum((days[i, k_idx] for i in range(num_workers) if i != k_idx))
    for j_idx, j in enumerate(workers):
        if j_idx != k_idx:
            expenditure_coeffs[j] = days[k_idx, j_idx]
    expr = income_sum * wage_vars[k] - gp.quicksum((expenditure_coeffs[j] * wage_vars[j] for j in expenditure_coeffs))
    m.addConstr(expr == 0, name=f'balance_{k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Daily wage rates (yuan per day):')
    for w in workers:
        print(f'{w}: {wage_vars[w].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')