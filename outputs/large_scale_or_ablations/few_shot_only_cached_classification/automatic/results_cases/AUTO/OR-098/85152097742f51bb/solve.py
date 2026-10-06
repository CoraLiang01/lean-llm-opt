import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owner_ids = df['Owner'].tolist()
if len(owner_ids) != num_workers:
    raise ValueError(f'Number of owners ({len(owner_ids)}) does not match number of workers ({num_workers}).')
owner_to_worker = {owner_ids[i]: worker_cols[i] for i in range(num_workers)}
worker_to_owner = {worker_cols[i]: owner_ids[i] for i in range(num_workers)}
work_days = df[worker_cols].to_numpy(dtype=float)
m = gp.Model('MutualAidWageBalance')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for i in range(num_workers):
    owner = owner_ids[i]
    worker_i = worker_cols[i]
    income = sum((work_days[k, i] for k in range(num_workers))) * wage_vars[worker_i]
    payment = gp.quicksum((work_days[i, j] * wage_vars[worker_cols[j]] for j in range(num_workers)))
    m.addConstr(income == payment, name=f'balance_{worker_i}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Daily wages (yuan):')
    for worker in worker_cols:
        print(f'{worker}: {wage_vars[worker].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')