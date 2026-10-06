import gurobipy as gp
import pandas as pd
import numpy as np
import re
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
if 'Owner' not in df.columns:
    raise KeyError("Missing required column 'Owner' in work_days.csv")
workers = [col for col in df.columns if col != 'Owner']
n_workers = len(workers)
owners = df['Owner'].tolist()
if len(owners) != n_workers:
    raise ValueError(f'Number of owners ({len(owners)}) does not match number of workers ({n_workers}).')
owner_to_idx = {owner: idx for (idx, owner) in enumerate(owners)}
work_days = df[workers].to_numpy(dtype=float)
if work_days.shape != (n_workers, n_workers):
    raise ValueError(f'work_days matrix shape {work_days.shape} does not match expected ({n_workers}, {n_workers})')
col_sums = work_days.sum(axis=0)
if not np.allclose(col_sums, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total. Column sums: ' + str(col_sums))

def solve_wage_fairness(workers, work_days):
    m = gp.Model('mutual_payment_fairness')
    m.Params.MIPGap = 0.0001
    w = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for (i, owner) in enumerate(owners):
        income_coeff = work_days[:, i].sum()
        expenditure_expr = gp.LinExpr()
        for (j, worker_j) in enumerate(workers):
            expenditure_expr.add(work_days[i, j], w[worker_j])
        m.addConstr(income_coeff * w[workers[i]] - expenditure_expr == 0, name='fair_%d' % i)
    first_worker = workers[0]
    m.addConstr(w[first_worker] == 60.0, name='norm')
    m.setObjective(0.0, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_wage_fairness(workers, work_days)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')