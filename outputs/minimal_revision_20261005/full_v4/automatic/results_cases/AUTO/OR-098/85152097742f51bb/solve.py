import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owners = df['Owner'].astype(str).tolist()
workers = [str(w) for w in worker_cols]
if owners != workers:
    raise ValueError('Owner names (rows) and worker names (columns) do not match or are not in the same order.')
work_days = df[worker_cols].to_numpy(dtype=float)
col_sums = work_days.sum(axis=0)
if not np.allclose(col_sums, 10.0):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('mutual_payment_balance')
wage_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
for (k_idx, k) in enumerate(workers):
    income_coeff = work_days[:, k_idx].copy()
    income_coeff[k_idx] = 0.0
    income_expr = income_coeff.sum() * wage_vars[k]
    payment_expr = gp.LinExpr()
    for (j_idx, j) in enumerate(workers):
        if j_idx == k_idx:
            continue
        payment_expr.addTerms(work_days[k_idx, j_idx], wage_vars[j])
    m.addConstr(income_expr == payment_expr, name='balance_' + k)
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for j in workers:
        print(f'{wage_vars[j].VarName} {wage_vars[j].X:.6f}')
else:
    print(f'Solver status: {m.status}')