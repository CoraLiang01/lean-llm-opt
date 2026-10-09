import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_columns)
owner_ids = df['Owner'].tolist()
num_owners = len(owner_ids)
owner_to_rowidx = {owner_ids[i]: i for i in range(num_owners)}
if not set(owner_ids).issubset(set(worker_columns)):
    missing = set(owner_ids) - set(worker_columns)
    raise ValueError(f'Some owners are not listed as workers: {missing}')
d = pd.DataFrame(0.0, index=owner_ids, columns=worker_columns)
for j in worker_columns:
    d[j] = df[j].apply(lambda x: float(x.strip()) if isinstance(x, str) and x.strip() != '' else 0.0)
m = gp.Model('MutualAidWageBalance')
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_columns[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='wage_normalization')
for k in worker_columns:
    lhs = gp.quicksum((d.loc[i, k] * wage_vars[k] for i in owner_ids))
    rhs = gp.quicksum((d.loc[k, j] * wage_vars[j] for j in worker_columns))
    m.addConstr(lhs == rhs, name=f'balance_{k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Worker daily wages (yuan):')
    for j in worker_columns:
        print(f'{j}: {wage_vars[j].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')