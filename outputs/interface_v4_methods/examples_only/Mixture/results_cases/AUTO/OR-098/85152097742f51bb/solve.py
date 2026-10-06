import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
num_owners = df.shape[0]
owner_names = df['Owner'].astype(str).tolist()
worker_names = [str(w) for w in worker_cols]
if set(owner_names) != set(worker_names):
    raise ValueError('Mismatch between owners and workers. Each owner must be a worker and vice versa.')
owner_idx = {name: idx for idx, name in enumerate(owner_names)}
worker_idx = {name: idx for idx, name in enumerate(worker_names)}
work_days = df[worker_cols].to_numpy(dtype=float)
worker_totals = work_days.sum(axis=0)
if not np.allclose(worker_totals, 10.0):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
first_worker = worker_cols[0]
fixed_wage = 60.0
m = gp.Model('MutualAidWageBalance')
w = m.addVars(worker_names, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstr(w[first_worker] == fixed_wage, name='fix_first_wage')
for i, owner in enumerate(owner_names):
    total_days_worked_by_i = work_days[:, worker_idx[owner]].sum()
    m.addConstr(total_days_worked_by_i * w[owner] == gp.quicksum((work_days[i, worker_idx[j]] * w[j] for j in worker_names)), name=f'balance_{owner}')
m.setObjective(0, gp.GRB.MINIMIZE)
m.optimize()