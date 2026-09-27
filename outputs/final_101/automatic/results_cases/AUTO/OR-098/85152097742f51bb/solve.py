import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owners = df['Owner'].astype(str).tolist()
worker_idx = {w: idx for idx, w in enumerate(worker_cols)}
missing_owners = set(owners) - set(worker_cols)
if missing_owners:
    raise ValueError(f'Owner(s) not found in worker columns: {missing_owners}')
A = df[worker_cols].to_numpy(dtype=float)
m = gp.Model('MutualAidWageFairness')
wage = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for j, worker_j in enumerate(worker_cols):
    try:
        i_own = owners.index(worker_j)
    except ValueError:
        raise ValueError(f"Worker '{worker_j}' not found as an owner in the data.")
    total_days_j = A[:, j].sum()
    if not np.isclose(total_days_j, 10.0, atol=1e-06):
        raise ValueError(f"Worker '{worker_j}' does not have exactly 10 work days (has {total_days_j}).")
    sum_Aij = total_days_j
    Ajj = A[i_own, j]
    other_workers = [worker_k for k, worker_k in enumerate(worker_cols) if k != j]
    other_indices = [k for k in range(num_workers) if k != j]
    Ajk = A[i_own, other_indices]
    m.addConstr((sum_Aij - Ajj) * wage[worker_j] == gp.quicksum((Ajk[k] * wage[other_workers[k]] for k in range(len(other_workers)))), name=f'fair_{worker_j}')
first_worker = worker_cols[0]
m.addConstr(wage[first_worker] == 60.0, name='norm')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Daily wages for all workers (yuan per day):')
    for worker in worker_cols:
        print(f'{worker}: {wage[worker].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')