import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
workers = list(worker_cols)
n_workers = len(workers)
homeowners = df['Owner'].astype(str).tolist()
if set(homeowners) != set(workers):
    raise ValueError('Mismatch between homeowners and workers identifiers.')
if homeowners != workers:
    raise ValueError('Order of homeowners and workers does not match. Please ensure the mapping is one-to-one and in the same order.')
days = df[worker_cols].copy()
days.index = homeowners
m = Model('mutual_wage_payment')
m.Params.OutputFlag = 0
wage_vars = {}
for (idx, j) in enumerate(workers):
    if idx == 0:
        wage_vars[j] = m.addVar(lb=60.0, ub=60.0, vtype=GRB.CONTINUOUS, name='w_' + j)
    else:
        wage_vars[j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='w_' + j)
m.update()
for i in workers:
    sum_days_hi = sum((days.loc[h, i] for h in homeowners if h != i))
    sum_days_ij_wj = sum((days.loc[i, j] * wage_vars[j] for j in workers if j != i))
    m.addConstr(sum_days_hi * wage_vars[i] - sum_days_ij_wj == 0, name='fair_' + i)
m.setObjective(0, GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
    result = {j: wage_vars[j].X for j in workers}
    for j in workers:
        print(f'{j}: {result[j]:.2f}')
elif m.status == GRB.INFEASIBLE:
    print('No feasible solution found.')
else:
    print(f'Gurobi ended with status {m.status}')