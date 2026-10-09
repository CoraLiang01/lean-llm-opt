import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
if num_workers != df.shape[0]:
    raise ValueError(f'Number of workers ({num_workers}) does not match number of owners ({df.shape[0]}).')
worker_idx_to_name = {i: worker_cols[i] for i in range(num_workers)}
worker_name_to_idx = {worker_cols[i]: i for i in range(num_workers)}
work_days = df[worker_cols].to_numpy(dtype=float)
col_sums = work_days.sum(axis=0)
if not np.allclose(col_sums, 10.0, atol=1e-06):
    raise ValueError("Each worker's total work days must sum to 10. Found: " + str(dict(zip(worker_cols, col_sums))))
m = gp.Model('mutual_wage_balance')
w = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k in worker_cols:
    k_idx = worker_name_to_idx[k]
    income_coeff = work_days[:, k_idx].copy()
    income_coeff[k_idx] = 0.0
    income = income_coeff.sum() * w[k]
    payment_coeff = work_days[k_idx, :].copy()
    payment_coeff[k_idx] = 0.0
    payment = gp.quicksum((payment_coeff[j] * w[worker_cols[j]] for j in range(num_workers) if j != k_idx))
    m.addConstr(income == payment, name='fair_' + k)
first_worker = worker_cols[0]
m.addConstr(w[first_worker] == 60.0, name='fix_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for j in worker_cols:
        print(f'{w[j].VarName} {w[j].X:.6f}')
else:
    print(f'Solver status: {m.status}')