import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
worker_cols = [col for col in df.columns if col != 'Owner']
workers = list(worker_cols)
homeowners = list(df['Owner'].astype(str))
missing_workers = set(workers) - set(homeowners)
if missing_workers:
    raise ValueError(f'Workers not found as homeowners: {missing_workers}')
missing_owners = set(homeowners) - set(workers)
if missing_owners:
    raise ValueError(f'Homeowners not found as workers: {missing_owners}')
days = {}
for (idx, row) in df.iterrows():
    h = str(row['Owner'])
    for j in workers:
        days[h, j] = int(row[j])
for j in workers:
    total_days = sum((days[h, j] for h in homeowners))
    if total_days != 10:
        raise ValueError(f'Worker {j} has total work days {total_days}, expected 10.')
owner_row_idx = {str(row['Owner']): idx for (idx, row) in df.iterrows()}
m = gp.Model('mutual_wage_balance')
m.Params.MIPGap = 0.0001
wage_vars = m.addVars(workers, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for j in workers:
    income = sum((days[h, j] * wage_vars[j] for h in homeowners))
    payment = gp.LinExpr()
    for k in workers:
        payment.addTerms(days[j, k], wage_vars[k])
    m.addConstr(income == payment, name='fairness_' + j)
m.setObjective(0, GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in wage_vars.values():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')