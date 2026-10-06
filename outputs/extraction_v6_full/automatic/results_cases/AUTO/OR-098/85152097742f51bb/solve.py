import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
worker_idx_to_name = {j: worker_cols[j] for j in range(num_workers)}
worker_name_to_idx = {name: j for j, name in worker_idx_to_name.items()}
work_days = df[worker_cols].to_numpy(dtype=float)
num_owners = work_days.shape[0]
if num_owners != num_workers:
    raise ValueError(f'Number of owners ({num_owners}) does not match number of workers ({num_workers}).')
worker_totals = work_days.sum(axis=0)
if not np.allclose(worker_totals, 10.0):
    raise ValueError('Each worker must contribute exactly 10 work days. Found: ' + str(worker_totals))
m = gp.Model('MutualAidWageFairness')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='wage')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name=f'FixWage_{first_worker}')
for k, worker_k in enumerate(worker_cols):
    income = work_days[:, k].sum() * wage_vars[worker_k]
    payment = gp.quicksum((work_days[k, j] * wage_vars[worker_cols[j]] for j in range(num_workers)))
    m.addConstr(income == payment, name=f'Fairness_{worker_k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()