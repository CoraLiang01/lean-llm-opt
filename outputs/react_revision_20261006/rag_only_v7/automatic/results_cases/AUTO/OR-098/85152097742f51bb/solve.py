import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_columns)
if df.shape[0] != num_workers:
    raise ValueError(f'Number of rows ({df.shape[0]}) does not match number of workers ({num_workers}).')
for (i, worker) in enumerate(worker_columns):
    owner_id = df.iloc[i]['Owner'].strip()
    if owner_id != worker:
        raise ValueError(f"Row {i} Owner '{owner_id}' does not match worker column '{worker}'.")
days = {}
for (i, owner) in enumerate(worker_columns):
    days[owner] = {}
    for (j, worker) in enumerate(worker_columns):
        val = df.iloc[i][worker]
        try:
            days[owner][worker] = int(val)
        except Exception:
            raise ValueError(f"Invalid integer in days[{owner}][{worker}]: '{val}'")
for worker in worker_columns:
    total_days = sum((days[owner][worker] for owner in worker_columns))
    if total_days != 10:
        raise ValueError(f"Worker '{worker}' contributed {total_days} days (expected 10).")
m = gp.Model('mutual_wage_balance')
m.Params.MIPGap = 0.0001
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=GRB.CONTINUOUS, name='')
first_worker = worker_columns[0]
wage_vars[first_worker].lb = 60.0
wage_vars[first_worker].ub = 60.0
for k in worker_columns:
    income = sum((days[owner][k] * wage_vars[k] for owner in worker_columns))
    expenditure = gp.quicksum((days[k][worker] * wage_vars[worker] for worker in worker_columns))
    m.addConstr(income == expenditure, name='')
m.setObjective(0, GRB.MINIMIZE)
m.optimize()