import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_names = [col for col in df.columns if col != 'Owner']
owner_names = df['Owner'].tolist()
if set(owner_names) != set(worker_names):
    raise ValueError('Mismatch between owners and workers. Each owner must be a worker and vice versa.')
n_workers = len(worker_names)
work_days = df.set_index('Owner').astype(float)
worker_totals = work_days.sum(axis=0)
if not np.allclose(worker_totals.values, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')
m = gp.Model('mutual_payment_wages')
wage_vars = m.addVars(worker_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for i in worker_names:
    income_coeff = work_days[i].sum()
    expenditure_coeffs = work_days.loc[i]
    expr = income_coeff * wage_vars[i] - gp.quicksum((expenditure_coeffs[j] * wage_vars[j] for j in worker_names))
    m.addConstr(expr == 0, name=f'fair_{i}')
first_worker = worker_names[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='norm')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for j in worker_names:
        print(f'{wage_vars[j].VarName} {wage_vars[j].X}')
else:
    print(f'Solver status: {m.status}')