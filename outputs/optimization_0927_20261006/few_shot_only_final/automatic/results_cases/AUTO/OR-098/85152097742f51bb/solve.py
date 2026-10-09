import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owners = df['Owner'].tolist()
num_owners = len(owners)
days_df = df[worker_cols].apply(pd.to_numeric, errors='raise')
worker_total_days = days_df.sum(axis=0)
if not np.allclose(worker_total_days.values, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('MutualAidWageFairness')
wage_vars = m.addVars(worker_cols, vtype=gp.GRB.CONTINUOUS, name='')
for (i, worker_i) in enumerate(worker_cols):
    income = sum((days_df.iloc[k, i] * wage_vars[worker_i] for k in range(num_owners)))
    expenditure = sum((days_df.iloc[i, j] * wage_vars[worker_cols[j]] for j in range(num_workers)))
    m.addConstr(income - expenditure == 0, name=f'fair_{worker_i}')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='wage_normalization')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Daily wages (yuan):')
    for worker in worker_cols:
        print(f'{worker}: {wage_vars[worker].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')