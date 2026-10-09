import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_columns = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_columns)
num_owners = df.shape[0]
owners = df['Owner'].astype(str).tolist()
if set(worker_columns) != set(owners):
    raise ValueError('Mismatch between worker columns and Owner rows. All workers must appear as owners.')
owner_to_row = {owner: idx for (idx, owner) in enumerate(owners)}
work_days = df[worker_columns].to_numpy(dtype=float)
worker_totals = work_days.sum(axis=0)
if not np.allclose(worker_totals, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days. Found: ' + str(dict(zip(worker_columns, worker_totals))))
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_columns[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for (k, worker_k) in enumerate(worker_columns):
    income_coeff = 0.0
    expense_expr = gp.LinExpr()
    for (i, owner_i) in enumerate(worker_columns):
        if i != k:
            income_coeff += work_days[i, k]
    for (j, worker_j) in enumerate(worker_columns):
        if j != k:
            expense_expr.addTerms(work_days[k, j], wage_vars[worker_j])
    m.addConstr(income_coeff * wage_vars[worker_k] == expense_expr, name='balance_' + worker_k)
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for worker in worker_columns:
        var = wage_vars[worker]
        print(f'{var.VarName} {var.X:.6f}')
else:
    print(f'Solver status: {m.status}')