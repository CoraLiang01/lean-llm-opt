import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
worker_to_idx = {worker: idx for idx, worker in enumerate(worker_cols)}
owners = df['Owner'].astype(str).tolist()
if set(owners) != set(worker_cols):
    raise ValueError('Mismatch between Owner names and worker columns.')
work_days = df[worker_cols].to_numpy(dtype=float)
m = gp.Model('Mutual_Payment_Wage_Balance')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
for k, worker_k in enumerate(worker_cols):
    income_coeff = np.sum(work_days[:, k])
    payment_coeffs = work_days[k, :]
    lhs = income_coeff * wage_vars[worker_k]
    rhs = gp.quicksum((payment_coeffs[j] * wage_vars[worker_cols[j]] for j in range(num_workers)))
    m.addConstr(lhs == rhs, name=f'balance_{worker_k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()