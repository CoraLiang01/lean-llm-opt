import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_columns)
owner_ids = df['Owner'].tolist()
num_owners = len(owner_ids)
owner_idx_to_id = {i: owner_ids[i] for i in range(num_owners)}
owner_id_to_idx = {owner_ids[i]: i for i in range(num_owners)}
worker_idx_to_id = {j: worker_columns[j] for j in range(num_workers)}
worker_id_to_idx = {worker_columns[j]: j for j in range(num_workers)}
d_matrix = np.zeros((num_owners, num_workers), dtype=float)
for j, worker in enumerate(worker_columns):
    d_matrix[:, j] = df[worker].astype(float).values
m = gp.Model('MutualAidWageBalance')
w_vars = m.addVars(worker_columns, vtype=gp.GRB.CONTINUOUS, name='')
for j, worker_j in enumerate(worker_columns):
    total_days_j = float(np.sum(d_matrix[:, j]))
    d_jk = d_matrix[j, :]
    balance_expr = total_days_j * w_vars[worker_j] - gp.quicksum((d_jk[k] * w_vars[worker_columns[k]] for k in range(num_workers)))
    m.addConstr(balance_expr == 0, name=f'balance_{worker_j}')
first_worker = worker_columns[0]
m.addConstr(w_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()