import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owner_to_row = {str(owner): idx for idx, owner in enumerate(df['Owner'])}
worker_to_col = {worker: idx for idx, worker in enumerate(worker_cols)}
work_days = df[worker_cols].to_numpy(dtype=float)
m = gp.Model('mutual_payment_balance')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for k, worker_k in enumerate(worker_cols):
    owner_row_idx = owner_to_row.get(worker_k)
    if owner_row_idx is None:
        raise ValueError(f"Owner '{worker_k}' not found in Owner column.")
    income_from_others = 0.0
    for i in range(df.shape[0]):
        if i != owner_row_idx:
            income_from_others += work_days[i, k]
    payment_to_others = gp.LinExpr()
    for j, worker_j in enumerate(worker_cols):
        if j != k:
            payment_to_others += work_days[owner_row_idx, j] * wage_vars[worker_j]
    m.addConstr(income_from_others * wage_vars[worker_k] == payment_to_others, name=f'balance_{worker_k}')
m.setObjective(0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Daily wage rates (yuan):')
    for worker in worker_cols:
        print(f'{worker}: {wage_vars[worker].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')