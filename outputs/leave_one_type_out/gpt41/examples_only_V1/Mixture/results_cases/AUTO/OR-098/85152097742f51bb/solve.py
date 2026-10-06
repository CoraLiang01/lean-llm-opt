import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
owner_ids = df['Owner'].astype(str).tolist()
if len(worker_cols) != len(owner_ids):
    raise ValueError('Number of workers and owners do not match.')
worker_idx = {w: i for i, w in enumerate(worker_cols)}
owner_idx = {o: i for i, o in enumerate(owner_ids)}
days = df[worker_cols].to_numpy(dtype=float)
n = len(worker_cols)
if days.shape != (n, n):
    raise ValueError(f'Work days matrix shape mismatch: expected ({n},{n}), got {days.shape}')
col_sums = days.sum(axis=0)
if not np.allclose(col_sums, 10, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(worker_cols, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for k, worker_k in enumerate(worker_cols):
    income = days[:, k].sum() * wage_vars[worker_k]
    payment = gp.quicksum((days[k, j] * wage_vars[worker_cols[j]] for j in range(n)))
    m.addConstr(income == payment, name=f'fair_{worker_k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.FEASIBLE:
    print('Feasible wage rates found:')
    for w in worker_cols:
        print(f'{w}: {wage_vars[w].X:.6f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')