import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owner_set = set(df['Owner'].str.strip())
worker_set = set([w.strip() for w in worker_cols])
if not worker_set.issubset(owner_set):
    missing = worker_set - owner_set
    raise ValueError(f'Each worker must appear as an Owner. Missing: {missing}')
owner_to_row = {owner.strip(): idx for (idx, owner) in enumerate(df['Owner'].str.strip())}
days_matrix = np.zeros((len(df), num_workers), dtype=int)
for (j, worker) in enumerate(worker_cols):
    days_matrix[:, j] = df[worker].astype(int).values
worker_to_rowidx = {worker: owner_to_row[worker] for worker in worker_cols}
for (j, worker) in enumerate(worker_cols):
    total_days = days_matrix[:, j].sum()
    if total_days != 10:
        raise ValueError(f'Worker {worker} has total work days {total_days}, expected 10.')
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
for (j, worker) in enumerate(worker_cols):
    own_row = worker_to_rowidx[worker]
    income_coeff = days_matrix[:, j].copy()
    income_coeff[own_row] = 0
    income_expr = wage_vars[worker] * income_coeff.sum()
    expense_expr = gp.quicksum((days_matrix[own_row, k] * wage_vars[worker_cols[k]] for k in range(num_workers) if k != j))
    m.addConstr(income_expr == expense_expr, name=f'balance_{worker}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for worker in worker_cols:
        print(f'{wage_vars[worker].VarName} {wage_vars[worker].X:.6f}')
else:
    print(f'Solver status: {m.status}')