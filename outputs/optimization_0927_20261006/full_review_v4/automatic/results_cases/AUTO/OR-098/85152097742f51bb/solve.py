import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
owner_ids = df['Owner'].tolist()
work_days_matrix = df[worker_columns].astype(int)
owner_id_to_idx = {owner: idx for (idx, owner) in enumerate(owner_ids)}
worker_id_to_idx = {worker: idx for (idx, worker) in enumerate(worker_columns)}
worker_totals = work_days_matrix.sum(axis=0)
if not np.all(worker_totals.values == 10):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_columns[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
for (j, worker_j) in enumerate(worker_columns):
    income = gp.quicksum((work_days_matrix.iloc[i, j] * wage_vars[worker_j] for (i, owner_i) in enumerate(owner_ids) if owner_i != worker_j))
    payment = gp.quicksum((work_days_matrix.loc[owner_j_idx, worker_k] * wage_vars[worker_k] for worker_k in worker_columns if worker_k != worker_j for owner_j_idx in [owner_id_to_idx[worker_j]]))
    m.addConstr(income == payment, name=f'balance_{worker_j}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found:')
    for worker in worker_columns:
        print(f'{worker}: {wage_vars[worker].X:.2f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')