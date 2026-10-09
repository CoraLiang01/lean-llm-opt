import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
owners = df['Owner'].tolist()
if len(owners) != len(worker_columns):
    raise ValueError(f'Number of owners ({len(owners)}) does not match number of workers ({len(worker_columns)}).')
owner_to_rowidx = {owner: idx for (idx, owner) in enumerate(owners)}
worker_to_colidx = {worker: idx for (idx, worker) in enumerate(worker_columns)}
owner_to_worker = {}
for (idx, owner) in enumerate(owners):
    if owner not in worker_columns:
        raise ValueError(f"Owner '{owner}' does not have a matching worker column.")
    owner_to_worker[owner] = owner
worker_to_owner = {worker: worker for worker in worker_columns}
work_days_numeric = df[worker_columns].astype(float).values
num_workers = len(worker_columns)
m = gp.Model('Mutual_Wage_Balance')
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for (k_idx, k) in enumerate(worker_columns):
    income = gp.quicksum((work_days_numeric[i, k_idx] * wage_vars[k] for i in range(num_workers) if i != k_idx))
    expense = gp.quicksum((work_days_numeric[k_idx, j] * wage_vars[worker_columns[j]] for j in range(num_workers) if j != k_idx))
    m.addConstr(income == expense, name=f'balance_{k}')
first_worker = worker_columns[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print(f'Fixed wage: {first_worker} = {wage_vars[first_worker].X:.2f} yuan/day')
    print('--- Daily wages for all workers ---')
    for worker in worker_columns:
        print(f'{worker}: {wage_vars[worker].X:.6f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')