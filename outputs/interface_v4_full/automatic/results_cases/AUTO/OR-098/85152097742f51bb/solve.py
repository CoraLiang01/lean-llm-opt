import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
num_owners = df.shape[0]
if num_workers != num_owners:
    raise ValueError(f'Number of workers ({num_workers}) does not match number of owners ({num_owners}).')
owner_to_worker = {i: worker_cols[i] for i in range(num_workers)}
worker_to_owner = {worker_cols[i]: i for i in range(num_workers)}
days = df[worker_cols].to_numpy(dtype=float)
worker_total_days = days.sum(axis=0)
if not np.allclose(worker_total_days, 10.0):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('Mutual_Payment_Wage_Balance')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k, worker_k in enumerate(worker_cols):
    income_days = np.sum(days[:, k]) - days[k, k]
    payment_expr = gp.LinExpr()
    for j, worker_j in enumerate(worker_cols):
        if j != k:
            payment_expr.addTerms(days[k, j], wage_vars[worker_j])
    m.addConstr(wage_vars[worker_k] * income_days - payment_expr == 0, name=f'balance_{worker_k}')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()