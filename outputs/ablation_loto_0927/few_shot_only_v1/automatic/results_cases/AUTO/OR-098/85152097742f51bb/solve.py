import gurobipy as gp
import pandas as pd
import numpy as np
import re
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_names = [col for col in df.columns if col != 'Owner']
n_workers = len(worker_names)
worker_idx = {name: idx for (idx, name) in enumerate(worker_names)}
if not set(df['Owner']) == set(worker_names):
    raise ValueError('Mismatch between Owner names and worker columns.')
work_days = df.set_index('Owner').reindex(index=worker_names, columns=worker_names)
if work_days.isnull().any().any():
    raise ValueError('Missing data in work_days matrix.')
m = gp.Model('MutualAidWageBalance')
wage_vars = m.addVars(worker_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k in worker_names:
    income_coeff = work_days[k][k]
    lhs = gp.quicksum((float(work_days.loc[i, k]) for i in worker_names)) * wage_vars[k]
    rhs = gp.quicksum((float(work_days.loc[k, j]) * wage_vars[j] for j in worker_names))
    m.addConstr(lhs == rhs, name=f'fairness_{k}')
first_worker = worker_names[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Daily wages (yuan):')
    for name in worker_names:
        print(f'{name}: {wage_vars[name].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')