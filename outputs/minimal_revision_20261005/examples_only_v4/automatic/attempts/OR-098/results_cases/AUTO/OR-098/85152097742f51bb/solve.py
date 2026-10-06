import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
worker_to_idx = {worker: idx for (idx, worker) in enumerate(worker_cols)}
idx_to_worker = {idx: worker for (worker, idx) in worker_to_idx.items()}
owners = df['Owner'].tolist()
if set(owners) != set(worker_cols):
    raise ValueError('Mismatch between workers and owners: every worker must have a corresponding owner row.')
owner_to_row = {owner: idx for (idx, owner) in enumerate(owners)}
work_days = df[worker_cols].to_numpy(dtype=float)
col_sums = work_days.sum(axis=0)
if not np.allclose(col_sums, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('mutual_wage_balance')
w = m.addVars(worker_cols, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(w[first_worker] == 60.0, name='fix_first_wage')
for (k, worker_k) in enumerate(worker_cols):
    total_days_k = work_days[:, k].sum()
    own_days_k = work_days[k, k]
    left_coeff = total_days_k - own_days_k
    right_expr = gp.LinExpr()
    for (j, worker_j) in enumerate(worker_cols):
        if j != k:
            right_expr.add(work_days[k, j], w[worker_j])
    m.addConstr(left_coeff * w[worker_k] == right_expr, name='fairness_%s' % worker_k)
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for worker in worker_cols:
        print(f'{w[worker].VarName} {w[worker].X:.6f}')
else:
    print(f'Solver status: {m.status}')